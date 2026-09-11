# SPDX-License-Identifier: Apache-2.0
"""Map saved graph call sequences to recorded 4096-wide reduction sources.

Static old-cache graph references are not a claim that every graph was loaded.
New graph filenames are restricted to the completed live binding receipts.
No generated Python is imported or GPU initialized.
"""

import argparse
import ast
import json
from pathlib import Path

from benchmarks.analyze_glm53_reduction_receipts import sha


def kernel_references(text):
    """Inspect actual module AST, excluding the duplicate compile-time docstring."""
    tree = ast.parse(text)
    references = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if (
            not isinstance(call.func, ast.Attribute)
            or call.func.attr != "triton"
            or not isinstance(call.func.value, ast.Name)
            or call.func.value.id != "async_compile"
        ):
            continue
        if len(call.args) < 2 or not isinstance(call.args[1], ast.Constant):
            raise ValueError("nonliteral generated Triton source")
        (target,) = node.targets
        if not isinstance(target, ast.Name) or target.id in references:
            raise ValueError("duplicate/nonliteral generated binding")
        references[target.id] = call.args[1].value
    return references


def call_sequence(text):
    tree = ast.parse(text)
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = ast.unparse(node.func)
        if name.startswith("torch.ops.vllm.") or (
            name.startswith("triton_") and name.endswith(".run")
        ):
            calls.append(
                dict(line=node.lineno, name=name, expression=ast.unparse(node))
            )
    return sorted(calls, key=lambda r: r["line"])


def kernel_fingerprint(text):
    functions = [
        n
        for n in ast.parse(text).body
        if isinstance(n, ast.FunctionDef) and n.name.startswith("triton_")
    ]
    if len(functions) != 1:
        raise ValueError("one generated kernel required")
    return ast.dump(functions[0])


def role(use):
    following = use["after"][0]["name"] if use["after"] else None
    if following == "torch.ops.vllm.moe_forward.default":
        return "post_attention_layernorm"
    if following in (
        "torch.ops.vllm.quixicore_fp8_block_linear.default",
        "torch.ops.vllm.quixicore_decode_linear.default",
    ):
        return "input_layernorm"
    if following is None and "mean_rms_norm" in use["call"]["name"]:
        return "final_mean_and_norm"
    raise ValueError("unrecognized normalization consumer")


def graph_signature(graph):
    uses = []
    for norm in graph["norms"]:
        for use in norm["uses"]:
            call = ast.parse(use["call"]["expression"], mode="eval").body
            # Match body, semantic consumer and exact ordered arguments; kernel
            # symbol suffixes and source filenames are compilation artifacts.
            uses.append(
                (
                    use["call"]["line"],
                    norm["rank"],
                    norm["body"],
                    role(use),
                    [ast.dump(a) for a in call.args],
                    [ast.dump(k) for k in call.keywords],
                )
            )
    return json.dumps([row[1:] for row in sorted(uses)])


def pair_graphs(graphs):
    arms = {arm: {} for arm in ("old", "new")}
    for graph in graphs:
        key = graph_signature(graph)
        if key in arms[graph["arm"]]:
            raise ValueError("ambiguous graph-role signature")
        arms[graph["arm"]][key] = graph
    if not arms["old"] or arms["old"].keys() != arms["new"].keys():
        raise ValueError("incomplete old/new graph-role correspondence")
    pairs, changes = [], {}
    for signature, old in arms["old"].items():
        new = arms["new"][signature]
        old_norms = sorted(old["norms"], key=lambda n: n["uses"][0]["call"]["line"])
        new_norms = sorted(new["norms"], key=lambda n: n["uses"][0]["call"]["line"])
        if len(old_norms) != len(new_norms):
            raise ValueError("normalization binding count changed")
        records = []
        for a, b in zip(old_norms, new_norms):
            if a["body"] != b["body"] or a["rank"] != b["rank"]:
                raise ValueError("normalization body/rank changed")
            record = dict(
                rank=a["rank"],
                old_source=a["source"],
                new_source=b["source"],
                old_config=a["config"],
                new_config=b["config"],
                roles=sorted({role(u) for u in a["uses"]}),
                body=a["body"],
            )
            records.append(record)
            if a["config"] != b["config"]:
                key = (a["rank"], a["source"])
                if key in changes and any(
                    changes[key][field] != record[field]
                    for field in ("old_config", "new_config", "roles", "body")
                ):
                    raise ValueError("inconsistent source choice across graphs")
                changes[key] = record
        pairs.append(dict(old_graph=old["path"], new_graph=new["path"], norms=records))
    return dict(pairs=pairs, unique_changed_sources=list(changes.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--serving-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("preserve prior output")
    manifest = json.loads(args.manifest.read_text())
    serving = json.loads(args.serving_audit.read_text())
    original = Path(manifest["original_namespace"])
    # Use only the old namespace's already-inventoried two-level Python sources.
    files = [
        ("old", original / name, digest)
        for name, digest in manifest["original_files"].items()
        if name.endswith(".py") and len(Path(name).parts) == 3
    ]
    for row in serving["graph_receipts"]:
        if sha(row["path"]) != row["sha256"]:
            raise ValueError("live receipt changed")
        document = json.loads(Path(row["path"]).read_text())
        files += [
            ("new", Path(g["module"]), g["sha256"])
            for g in document["snapshots"]["after"]["graphs"]
        ]
    inventory = manifest["inventory"]
    # Embedded function source includes surrounding whitespace; compare only AST.
    fingerprints = {}
    for row in inventory:
        text = Path(row["source"]).read_text()
        if sha(row["source"]) != row["source_sha256"]:
            raise ValueError("bound source changed")
        fingerprints.setdefault(kernel_fingerprint(text), []).append(row)
    graphs = []
    for arm, path, digest in files:
        if sha(path) != digest:
            raise ValueError(f"graph/source changed: {path}")
        text = path.read_text()
        references = kernel_references(text)
        if not references:
            continue
        sequence = call_sequence(text)
        norms = []
        for symbol, embedded in references.items():
            matches = fingerprints.get(kernel_fingerprint(embedded), [])
            for matched in matches:
                if matched["info"]["widths"] != [4096]:
                    continue
                if (matched["arm"] == "combo") != (arm == "old"):
                    continue
                uses = []
                for i, call in enumerate(sequence):
                    if call["name"] != symbol + ".run":
                        continue
                    uses.append(
                        dict(
                            call=call,
                            before=sequence[max(0, i - 2) : i],
                            after=sequence[i + 1 : i + 3],
                        )
                    )
                norms.append(
                    dict(
                        rank=matched["rank"],
                        symbol=symbol,
                        source=matched["source"],
                        config=matched["selected"]["config"],
                        body=matched["info"]["body_sha256"],
                        uses=uses,
                    )
                )
        if norms:
            graphs.append(
                dict(
                    arm=arm,
                    path=str(path),
                    sha256=digest,
                    sequence=sequence,
                    norms=norms,
                )
            )
    if (
        not graphs
        or not any(g["arm"] == "old" for g in graphs)
        or not any(g["arm"] == "new" for g in graphs)
    ):
        raise ValueError("missing old/new graph references")
    output = dict(
        status="complete",
        analyzer_sha256=sha(__file__),
        manifest_sha256=sha(args.manifest),
        serving_audit_sha256=sha(args.serving_audit),
        graphs=graphs,
        correspondence=pair_graphs(graphs),
        limitation=__doc__,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                output=str(args.output),
                sha256=sha(args.output),
                graphs=len(graphs),
                old_graphs=sum(g["arm"] == "old" for g in graphs),
                new_graphs=sum(g["arm"] == "new" for g in graphs),
            )
        )
    )


if __name__ == "__main__":
    main()
