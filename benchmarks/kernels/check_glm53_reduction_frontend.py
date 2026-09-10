#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fresh GPU frontend/receipt qualification on vLLM's real RMSNorm IR lowering.

Three small graphs, not the full model or TP execution. The runner facade exercises
the exact startup recorder before/after capture. No cached-source intervention.
"""

import argparse
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

from benchmarks.kernels.check_glm53_cached_rmsnorm import (
    compare,
    make_inputs,
    oracle,
    sha,
    tensor_sha,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("new output required; preserve failed attempts")
    if subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip():
        parser.error("GPU workload already active")
    args.output.mkdir(parents=True)
    root = args.output.resolve() / "cache"
    os.environ["VLLM_CACHE_ROOT"] = str(root)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(root / "inductor")
    os.environ["TRITON_CACHE_DIR"] = str(root / "triton")
    os.environ["SLIMSERVE_GLM53_NATIVE_ORDER"] = "1"
    summary = dict(
        status="running",
        source_sha256=sha(__file__),
        checks=[],
        full_model=False,
        tensor_parallel_execution=False,
    )

    def save():
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    save()
    try:
        import torch
        from safetensors import safe_open

        from slimserve import reduction_receipts
        from slimserve.deterministic_reductions import diagnostic_plan
        from slimserve.registry import resolve
        from vllm import ir
        from vllm.compilation.passes.ir.lowering_pass import VllmIRLoweringPass
        from vllm.config import CompilationConfig

        torch.cuda.set_device(0)
        plan = diagnostic_plan(resolve("glm53-nvfp4-4", "rtx6000", 4, None))
        compilation = CompilationConfig(**plan.engine["compilation_config"])
        lowering = VllmIRLoweringPass(
            SimpleNamespace(
                compilation_config=compilation, model_config=None, device_config=None
            )
        )
        options = {
            **compilation.inductor_compile_config,
            "post_grad_custom_post_pass": lowering,
        }
        ir.ops.rms_norm.set_default(["native"])
        key = "model.language_model.layers.22.post_attention_layernorm.weight"
        index = json.loads((args.model / "model.safetensors.index.json").read_text())
        with safe_open(
            args.model / index["weight_map"][key], framework="pt", device="cpu"
        ) as f:
            weight = f.get_tensor(key)
        wg = weight.cuda()

        def inplace(x, w):
            x.copy_(ir.ops.rms_norm(x, w, 1e-5))
            return (x,)

        def triple(x, w):
            out = ir.ops.rms_norm(x, w, 1e-5)
            return out.clone(), out.clone(), out.clone()

        def independent(x, w):
            # Disjoint pointwise branches exercise horizontal-fusion eligibility,
            # which the original single-reduction graphs did not cover.
            return ir.ops.rms_norm(x, w, 1e-5), w + 1, w[:2048] * 2

        jobs = []
        for name, function in (
            ("inplace", inplace),
            ("triple", triple),
            ("independent", independent),
        ):
            compiled = torch.compile(
                function, fullgraph=True, dynamic=True, options=options
            )
            for rows in (1, 16, 640, 7616):
                original = make_inputs(rows, 530901, 1.0)
                changed = make_inputs(rows, 531001, 1.0)
                storage = torch.full(
                    (rows + 2, 4096), 123.0, dtype=torch.bfloat16, device="cuda"
                )
                x = storage[1:-1]
                x.copy_(original)
                first = [t.cpu() for t in compiled(x, wg)]
                x.copy_(original)
                second = [t.cpu() for t in compiled(x, wg)]
                assert all(torch.equal(a, b) for a, b in zip(first, second))
                jobs.append(
                    (name, rows, compiled, storage, x, original, changed, first)
                )
        summary.update(
            recorder_sha256=sha(reduction_receipts.__file__),
            weight_sha256=tensor_sha(weight),
            compiler_options=compilation.inductor_compile_config,
            lowering=lowering.selected_impls,
        )

        def capture_model():
            for name, rows, compiled, storage, x, original, changed, first in jobs:
                x.copy_(original)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    outputs = compiled(x, wg)
                x.copy_(original)
                graph.replay()
                assert all(torch.equal(t.cpu(), ref) for t, ref in zip(outputs, first))
                x.copy_(changed)
                graph.replay()
                actual = [t.cpu() for t in outputs]
                if name == "independent":
                    assert torch.equal(actual[1], weight + 1)
                    assert torch.equal(actual[2], weight[:2048] * 2)
                    assert torch.equal(first[1], actual[1])
                    assert torch.equal(first[2], actual[2])
                else:
                    assert all(torch.equal(t, actual[0]) for t in actual)
                assert not torch.equal(actual[0], first[0])
                assert torch.equal(wg.cpu(), weight)
                assert torch.all(storage[[0, -1]] == 123).item()
                if name != "inplace":
                    assert torch.equal(x.cpu(), changed)
                metrics = [
                    compare(first[0], oracle(original, weight)),
                    compare(actual[0], oracle(changed, weight)),
                ]
                assert all(m["max_bf16_ulp"] <= 1 for m in metrics)
                summary["checks"].append(
                    dict(
                        name=name,
                        rows=rows,
                        oracle=metrics,
                        repeat_graph_guards_mutation_pass=True,
                        original_output_sha256=tensor_sha(first[0]),
                        changed_output_sha256=tensor_sha(actual[0]),
                    )
                )
                save()
                print(json.dumps(dict(name=name, rows=rows, status="pass")), flush=True)
            return "captured"

        runner = SimpleNamespace(
            compilation_config=compilation,
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(hidden_size=4096, num_hidden_layers=45)
            ),
            parallel_config=SimpleNamespace(
                rank=0,
                tensor_parallel_size=4,
                pipeline_parallel_size=1,
                enable_expert_parallel=False,
            ),
            speculative_config=None,
            capture_model=capture_model,
        )
        reduction_receipts.install(runner)
        assert runner.capture_model() == "captured"
        (receipt,) = (root / "glm53-reduction-receipts").glob("*.json")
        summary.update(
            status="complete", receipt=str(receipt), receipt_sha256=sha(receipt)
        )
    except BaseException as error:
        summary.update(status="failed", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
