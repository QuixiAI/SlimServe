# SPDX-License-Identifier: Apache-2.0
"""Map saved attention normalization boundaries without importing generated code.

Static graph correspondence is not proof of historical live launch coverage or
model causality. This does not clear the failed indexer LayerNorm accuracy gate.
"""

import argparse
import ast
import json
from collections import Counter
from pathlib import Path

from benchmarks.analyze_glm53_kda_choices import read_pinned
from benchmarks.analyze_glm53_norm_graph_roles import (
    call_sequence,
    kernel_fingerprint,
    kernel_references,
)
from benchmarks.analyze_glm53_quality_pair import require
from benchmarks.analyze_glm53_reduction_receipts import sha

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "perf/results/2026-09-10"
MAPPING = RESULTS / "runtime-control/norm-graph-role-pairs.json"
MAPPING_SHA = "320a6f39c4f92cd78459f21e23653e1e900870620da29bed923f5cba5dbe315e"
PROBE = RESULTS / "attention-norm-rank-private-probe/analysis.json"
PROBE_SHA = "0da742b0b8d7eb08651be0b32fff0b7c874aaa87781d01d47ce2ca3877e06bdb"
MANIFEST = RESULTS / "runtime-control/attention-rank-private-manifest.json"
ROLES = {512: "kv_a_rmsnorm", 1536: "q_a_rmsnorm", 128: "indexer_layernorm"}


def runtime_function(text):
    tree = ast.parse(text)
    functions = [
        method
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Runner"
        for method in node.body
        if isinstance(method, ast.FunctionDef) and method.name == "call"
    ]
    require(len(functions) == 1, "one actual Runner.call required")
    return functions[0]


def tensor_definition(function, name):
    """Bounded allocation/alias provenance, excluding deletion and docstrings."""
    definitions = {}
    for node in ast.walk(function):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    definitions.setdefault(target.id, []).append(node.value)

    def visit(symbol, ancestors):
        require(symbol not in ancestors, "cyclic buffer alias")
        values = definitions.get(symbol, [])
        require(len(values) == 1, f"ambiguous/missing buffer definition: {symbol}")
        value = values[0]
        result = dict(symbol=symbol, expression=ast.unparse(value))
        if isinstance(value, ast.Name):
            result["base"] = visit(value.id, ancestors | {symbol})
        elif (
            isinstance(value, ast.Call)
            and ast.unparse(value.func) == "reinterpret_tensor"
        ):
            require(isinstance(value.args[0], ast.Name), "nonliteral alias base")
            result["base"] = visit(value.args[0].id, ancestors | {symbol})
        return result

    return visit(name, set())


def target_calls(text, records, rank, arm):
    bindings = {}
    references = kernel_references(text)
    for record in records:
        if record["rank"] != rank or record["arm"] != arm:
            continue
        fingerprint = record["fingerprint"]
        symbols = [
            symbol
            for symbol, source in references.items()
            if kernel_fingerprint(source) == fingerprint
        ]
        require(len(symbols) <= 1, "ambiguous embedded target")
        if not symbols:
            continue
        (symbol,) = symbols
        require(symbol not in bindings, "duplicate target source")
        uses = [c for c in call_sequence(text) if c["name"] == symbol + ".run"]
        require(len(uses) == 1, "target must have exactly one runtime call")
        call = ast.parse(uses[0]["expression"], mode="eval").body
        require(
            len(call.keywords) == 1 and call.keywords[0].arg == "stream",
            "unexpected launch kwargs",
        )
        bindings[symbol] = dict(
            symbol=symbol,
            source=record["source"],
            source_sha256=record["source_sha256"],
            widths=record["info"]["widths"],
            config=record["selected"]["config"],
            args=[ast.unparse(a) for a in call.args],
            stream=ast.unparse(call.keywords[0].value),
            line=uses[0]["line"],
        )
    return list(bindings.values())


def compare_pair(old_text, new_text, records, rank):
    old = target_calls(old_text, records, rank, "combo")
    new = target_calls(new_text, records, rank, "split")
    if not old and not new:
        return None
    require(len(old) == 1 and len(new) == 3, "one combo/three split bindings required")
    (combo,) = old
    require(combo["widths"] == [512, 1536, 128], "wrong combo widths")
    require(len(combo["args"]) == 11, "wrong combo call arity")
    split = {tuple(item["widths"]): item for item in new}
    require(set(split) == {(512,), (1536,), (128,)}, "wrong split widths")
    c = combo["args"]
    require(c[8] == c[9] == c[10], "combo row counts disagree")
    expected = {
        512: [c[0], c[1], c[5], c[8], "512"],
        1536: [c[0], c[2], c[6], c[9], "1536"],
        128: [c[0], c[3], c[4], c[7], c[10], "128"],
    }
    for width, args in expected.items():
        require(
            split[width,]["args"] == args,
            f"{width} input/weight/output contract changed",
        )
        require(split[width,]["stream"] == combo["stream"], "launch stream changed")
    functions = [runtime_function(t) for t in (old_text, new_text)]
    definitions = []
    for function in functions:
        definitions.append(
            {name: tensor_definition(function, name) for name in (c[0], *c[5:8])}
        )
    require(
        definitions[0] == definitions[1], "input/output allocation or alias changed"
    )
    returns = [
        [ast.dump(n.value) for n in ast.walk(f) if isinstance(n, ast.Return)]
        for f in functions
    ]
    require(
        len(returns[0]) == 1 and returns[0] == returns[1],
        "graph return interface changed",
    )
    native_calls = [
        [
            row["expression"]
            for row in call_sequence(t)
            if row["name"].startswith("torch.ops.vllm.")
        ]
        for t in (old_text, new_text)
    ]
    require(native_calls[0] == native_calls[1], "native operation sequence changed")
    return dict(
        rank=rank,
        combo=combo,
        split=new,
        tensor_definitions=definitions[0],
        graph_return=ast.unparse(
            next(n for n in ast.walk(functions[0]) if isinstance(n, ast.Return)).value
        ),
        native_calls=native_calls[0],
        outputs={ROLES[width]: expected[width][-3] for width in ROLES},
        unqualified="Non-target pointwise fusion and runtime values are not tested.",
    )


def analyze():
    mapping = read_pinned(MAPPING, MAPPING_SHA)
    probe = read_pinned(PROBE, PROBE_SHA)
    require(
        probe["pairs"] == 120 and probe["numerical_pass"] is False,
        "historical probe status changed",
    )
    manifest = read_pinned(MANIFEST, probe["manifest_sha256"])
    receipts = {
        str(MAPPING): MAPPING_SHA,
        str(PROBE): PROBE_SHA,
        str(MANIFEST): probe["manifest_sha256"],
    }
    records = []
    for record in manifest["records"]:
        path = Path(record["source"])
        require(sha(path) == record["source_sha256"], "recorded kernel source changed")
        records.append({**record, "fingerprint": kernel_fingerprint(path.read_text())})
        receipts[str(path)] = record["source_sha256"]
    graph_receipts = {g["path"]: g["sha256"] for g in mapping["graphs"]}
    require(len(graph_receipts) == len(mapping["graphs"]), "duplicate graph paths")
    paired, skipped = [], 0
    for pair in mapping["correspondence"]["pairs"]:
        ranks = {n["rank"] for n in pair["norms"]}
        require(len(ranks) == 1, "mixed graph ranks")
        texts = []
        for key in ("old_graph", "new_graph"):
            path = Path(pair[key])
            require(sha(path) == graph_receipts[str(path)], "recorded graph changed")
            receipts[str(path)] = graph_receipts[str(path)]
            texts.append(path.read_text())
        result = compare_pair(*texts, records, ranks.pop())
        if result is None:
            skipped += 1
        else:
            paired.append(
                dict(old_graph=pair["old_graph"], new_graph=pair["new_graph"], **result)
            )
    require(
        Counter(p["rank"] for p in paired) == {r: 2 for r in range(4)},
        "incomplete attention graph pairs",
    )
    return dict(
        status="complete",
        analyzer_sha256=sha(__file__),
        receipts=receipts,
        pairs=paired,
        non_attention_pairs=skipped,
        historical_indexer_oracle_pass=False,
        production_qualified=False,
        limitation=__doc__,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "preserve prior analysis")
    result = analyze()
    with args.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(
        json.dumps(
            dict(
                output=str(args.output),
                sha256=sha(args.output),
                pairs=len(result["pairs"]),
                receipts=len(result["receipts"]),
            )
        )
    )


if __name__ == "__main__":
    main()
