# SPDX-License-Identifier: Apache-2.0
"""Numerical checks through actual AOT-held launchers, not model forwards."""

from pathlib import Path

from benchmarks.kernels import check_glm53_attention_norms as norms
from benchmarks.kernels.check_glm53_rmsnorm_geometry import write_new

PHASE_ORDER = [0, 0, 1, 1, 0]


def expected_bindings(graphs):
    return sorted(
        (
            {k: r[k] for k in ("graph", "symbol", "source")}
            for r in graphs["bindings"]
            if r["target"]
        ),
        key=lambda r: (r["graph"], r["symbol"], r["source"]),
    )


def expected_cases(manifest):
    return [
        r
        for r in manifest["qualified_leaf_cases"]
        if r["case"]["rank"] == manifest["rank"]
    ]


def audit_leaf(record, binding, entry, mode):
    norms.require(mode in ("control", "correction"), "invalid leaf mode")
    norms.require(
        record["status"] == "complete"
        and record["binding"] == binding
        and record["qualified_case"] == entry,
        "leaf binding/case identity differs",
    )
    qualified = norms.load_checked(Path(entry["path"]), entry["sha256"])
    norms.require(
        record["input_sha256"] == qualified["input_sha256"]
        and record["replay_guards_pass"] is True,
        "leaf input/replay guards differ",
    )
    norms.require(
        [o["phase"] for o in record["observations"]] == PHASE_ORDER,
        "missing leaf observations",
    )
    for observation in record["observations"]:
        reference = qualified["phases"][observation["phase"]]
        key = "candidate_sha256" if mode == "correction" else "baseline_sha256"
        norms.require(
            observation["outputs"] == reference[key],
            "bound leaf output differs from qualified binary",
        )
        if mode == "correction":
            norms.require(
                observation["selection_sha256"] == reference["selected_sha256"]
                and observation["arena_guards_pass"] is True,
                "bound selection or arena guards differ",
            )
        else:
            norms.require(
                "selection_sha256" not in observation,
                "control unexpectedly writes selection",
            )


def qualify_bindings(modules, manifest, graphs, output):
    import torch

    private = Path(manifest["private_namespace"])
    module_map = {str(Path(m.__file__).relative_to(private)): m for m in modules}
    weights = norms.load_weights()
    norms.require(
        {k: norms.tensor_sha(w) for k, w in zip(norms.WEIGHTS, weights)}
        == manifest["weight_sha256"],
        "leaf checkpoint weights changed",
    )
    bindings, cases = expected_bindings(graphs), expected_cases(manifest)
    norms.require(
        len(bindings) == 2 and len(cases) == 30, "two bindings/thirty cases required"
    )
    results = []
    for binding in bindings:
        tuner = vars(module_map[binding["graph"]])[binding["symbol"]]
        launch = vars(tuner)["run"]
        adapter = (
            getattr(launch, "__self__", None)
            if manifest["mode"] == "correction"
            else None
        )
        for entry in cases:
            case = entry["case"]
            record = dict(
                status="running", binding=binding, qualified_case=entry, observations=[]
            )
            path = output / f"leaf-{len(results):03d}.json"
            try:
                data = [
                    norms.packed_inputs(
                        case["rows"], case["seed"] + 100 * p, case["magnitude"]
                    )
                    for p in range(2)
                ]
                record["input_sha256"] = [norms.tensor_sha(x) for x in data]
                if adapter is not None:
                    adapter.verify()
                    adapter.storage.fill_(171)

                def observe(
                    phase, values, *, adapter=adapter, case=case, record=record
                ):
                    item = dict(
                        phase=phase, outputs=[norms.tensor_sha(t) for t in values]
                    )
                    if adapter is not None:
                        adapter.verify()
                        storage = adapter.storage.cpu()
                        rows = case["rows"]
                        norms.require(
                            torch.all(storage[0] == 171).item()
                            and torch.all(storage[rows + 1 :] == 171).item(),
                            "leaf arena guard or unused rows mutated",
                        )
                        item.update(
                            selection_sha256=norms.tensor_sha(storage[1 : rows + 1]),
                            arena_guards_pass=True,
                        )
                    record["observations"].append(item)

                norms.run_launches(
                    {"combo": launch}, *data, weights, observe_phase=observe
                )
                norms.require(
                    vars(tuner)["run"] is launch,
                    "live target dispatch changed during leaves",
                )
                record.update(status="complete", replay_guards_pass=True)
                audit_leaf(record, binding, entry, manifest["mode"])
            except BaseException as error:
                record.update(status="failed", error=repr(error))
                raise
            finally:
                write_new(path, record)
            results.append(dict(path=path.name, sha256=norms.sha(path)))
            if len(results) % 10 == 0:
                print(f"Bound leaves: {len(results)}/60", flush=True)
    report = dict(
        status="complete", bindings=bindings, cases=len(results), records=results
    )
    write_new(output / "bound-leaves.json", report)
    return dict(cases=len(results), sha256=norms.sha(output / "bound-leaves.json"))
