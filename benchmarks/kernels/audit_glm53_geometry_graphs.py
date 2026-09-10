# SPDX-License-Identifier: Apache-2.0
"""Read actual graph globals independently of intervention callback receipts."""

import ast
from pathlib import Path

from benchmarks.kernels.check_glm53_attention_norms import require
from slimserve.rmsnorm_diagnostic import expected_receipt, launcher_receipt, sha


def exported_call(source):
    """Resolve the actual export: direct call or instance-bound Runner.call.

    Follow explicit top-level bindings, not arbitrary nested methods named call
    or compile-time source strings. Returned names also bind the live method.
    """
    body = ast.parse(source).body

    def assignments(name):
        return [
            node
            for node in body
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
        ]

    functions = [n for n in body if isinstance(n, ast.FunctionDef) and n.name == "call"]
    aliases = assignments("call")
    if functions:
        require(len(functions) == 1 and not aliases, "ambiguous direct call export")
        return functions[0], None, None
    require(len(aliases) == 1, "one explicit graph call export required")
    alias = aliases[0].value
    require(
        isinstance(alias, ast.Attribute)
        and alias.attr == "call"
        and isinstance(alias.value, ast.Name),
        "unsupported graph call export",
    )
    instance_name = alias.value.id
    instances = assignments(instance_name)
    require(len(instances) == 1, "one exported runner instance required")
    constructor = instances[0].value
    require(
        isinstance(constructor, ast.Call) and isinstance(constructor.func, ast.Name),
        "explicit runner constructor required",
    )
    class_name = constructor.func.id
    classes = [n for n in body if isinstance(n, ast.ClassDef) and n.name == class_name]
    require(len(classes) == 1, "one exported runner class required")
    methods = [
        n
        for n in classes[0].body
        if isinstance(n, ast.FunctionDef) and n.name == "call"
    ]
    require(len(methods) == 1, "one exported runner call method required")
    return methods[0], class_name, instance_name


def verify_call_export(module, source):
    definition, class_name, instance_name = exported_call(source)
    call = module.call
    namespace = vars(module)
    if class_name is None:
        function = call
        require(getattr(call, "__self__", None) is None, "direct call became bound")
    else:
        instance = namespace.get(instance_name)
        cls = namespace.get(class_name)
        function = getattr(call, "__func__", None)
        require(
            instance is not None
            and type(instance) is cls
            and getattr(call, "__self__", None) is instance
            and function is vars(cls).get("call"),
            "live call is not its exported runner method",
        )
    code = getattr(function, "__code__", None)
    require(
        getattr(function, "__globals__", None) is namespace
        and code is not None
        and code.co_filename == module.__file__
        and code.co_firstlineno == definition.lineno,
        "live call definition differs from inventoried source/globals",
    )


def run_symbols(source):
    function, _, _ = exported_call(source)
    return {
        node.func.value.id
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run"
        and isinstance(node.func.value, ast.Name)
    }


def inventory(modules, manifest, rank, mode, observer):
    """Neither controller owners nor its graph_bindings determine this inventory.

    Inspect the call function's actual globals, the source's executable run calls,
    each selected config and each launcher's observed CUDA object. Include ALL
    bound Triton globals so a later control/geometry comparison can also reject
    unrelated changes. No model execution is performed.
    """
    require(mode in ("control", "geometry"), "invalid graph audit mode")
    private = Path(manifest["private_namespace"])
    original = Path(manifest["original_namespace"])
    expected_graphs = manifest["expected_graphs"][str(rank)]
    targets = {t["relative"]: t for t in manifest["targets"][str(rank)]}
    expected_bindings = {
        (
            str(Path(use["graph"]).relative_to(original)),
            use["symbol"],
            target["relative"],
        )
        for target in targets.values()
        for use in target["static_graph_uses"]
    }
    found_graphs, found_targets, objects, rows = set(), set(), {}, []
    for module in modules:
        call = getattr(module, "call", None)
        if not callable(call) or id(module) in objects:
            continue
        objects[id(module)] = module
        path = Path(module.__file__)
        require(
            path.resolve() == path and path.is_relative_to(private),
            "graph outside exact private namespace",
        )
        relative = str(path.relative_to(private))
        require(
            relative in expected_graphs and sha(path) == expected_graphs[relative],
            "unexpected or changed graph source",
        )
        namespace = vars(module)
        source_text = path.read_text()
        verify_call_export(module, source_text)
        referenced = run_symbols(source_text)
        found_graphs.add(relative)
        bound_symbols = set()
        for symbol, tuner in namespace.items():
            filename = getattr(tuner, "filename", None)
            if not isinstance(filename, str) or not hasattr(tuner, "launchers"):
                continue
            source = Path(filename)
            require(
                source.resolve() == source and source.is_relative_to(private),
                "bound source outside private namespace",
            )
            source_relative = str(source.relative_to(private))
            digest = sha(source)
            require(
                manifest["original_files"].get(source_relative) == digest,
                "bound source differs from original snapshot",
            )
            selected = launcher_receipt(tuner)
            results = tuner.compile_results
            require(
                len(selected) == len(results) == 1,
                "one resolved static launcher required",
            )
            observer.verify_launcher(results[0], tuner.launchers[0])
            binary = observer.digest(results[0])
            target = targets.get(source_relative)
            if target is not None:
                key = (relative, symbol, source_relative)
                require(
                    key in expected_bindings and symbol in referenced,
                    "unexpected or unused target graph binding",
                )
                require(
                    selected == expected_receipt(target["configs"][mode])
                    and binary == target["configs"][mode]["cubin_sha256"],
                    "graph selected wrong target config/binary",
                )
                found_targets.add(key)
            bound_symbols.add(symbol)
            record = observer.records[id(results[0].kernel)]
            rows.append(
                dict(
                    graph=relative,
                    graph_sha256=expected_graphs[relative],
                    symbol=symbol,
                    source=source_relative,
                    source_sha256=digest,
                    target=target is not None,
                    referenced=symbol in referenced,
                    selected=selected,
                    cubin_sha256=binary,
                    observed_binary_index=record["index"],
                    module_index=len(objects),
                )
            )
        require(
            referenced <= bound_symbols, "executable kernel call lacks a bound launcher"
        )
    require(found_graphs == set(expected_graphs), "incomplete actual graph inventory")
    require(found_targets == expected_bindings, "incomplete actual target bindings")
    require(
        {r["source"] for r in rows if r["target"]} == set(targets),
        "incomplete actual source coverage",
    )
    return dict(
        graphs=len(found_graphs), target_bindings=len(found_targets), bindings=rows
    )


def compare_unrelated(control, geometry):
    def stable(data):
        return sorted(
            [
                {
                    k: v
                    for k, v in row.items()
                    if k not in ("module_index", "observed_binary_index")
                }
                for row in data["bindings"]
                if not row["target"]
            ],
            key=lambda row: (row["graph"], row["symbol"], row["source"]),
        )

    require(
        control["graphs"] == geometry["graphs"]
        and control["target_bindings"] == geometry["target_bindings"],
        "control/geometry coverage differs",
    )
    require(
        stable(control) == stable(geometry),
        "non-target graph source/config/binary changed",
    )
