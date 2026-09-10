# SPDX-License-Identifier: Apache-2.0
import pytest

from benchmarks.analyze_glm53_norm_graph_roles import (
    call_sequence,
    graph_signature,
    kernel_fingerprint,
    kernel_references,
    pair_graphs,
)


def test_graph_inventory_ignores_compile_time_docstring():
    text = '''"""triton_fake.run(x)"""
triton_norm = async_compile.triton('triton_norm',
    'def triton_norm(x):\\n    return x\\n')
class Runner:
    def call(self, args):
        x = torch.ops.vllm.glm5_mhc_fused_post_pre.default(args)
        triton_norm.run(x, args, 4096)
        return torch.ops.vllm.moe_forward.default(hidden_states=x)
'''
    assert set(kernel_references(text)) == {"triton_norm"}
    assert [c["name"] for c in call_sequence(text)] == [
        "torch.ops.vllm.glm5_mhc_fused_post_pre.default",
        "triton_norm.run",
        "torch.ops.vllm.moe_forward.default",
    ]


def test_kernel_fingerprint_ignores_harness_imports_not_metadata():
    source = "@heuristic(flag=False)\ndef triton_norm(x):\n    return x\n"
    assert kernel_fingerprint(source) == kernel_fingerprint(
        "import torch\n" + source + "def get_args():\n    pass\n"
    )
    assert kernel_fingerprint(source) != kernel_fingerprint(
        source.replace("False", "True")
    )
    with pytest.raises(ValueError):
        kernel_fingerprint(source + source)


def test_dynamic_and_duplicate_binding_rejected():
    for source in (
        "x = async_compile.triton('x', computed_source)",
        (
            "x = async_compile.triton('x', 'source')\n"
            "x = async_compile.triton('x', 'source')"
        ),
    ):
        with pytest.raises(ValueError):
            kernel_references(source)


def test_graph_pair_uses_body_arguments_and_consumer_not_symbol_name():
    import copy

    old = dict(
        arm="old",
        path="old.py",
        norms=[
            dict(
                rank=0,
                body="body",
                source="a.py",
                config={"R0_BLOCK": 4096},
                uses=[
                    dict(
                        call=dict(
                            line=1,
                            name="triton_rms_norm_0.run",
                            expression="triton_rms_norm_0.run(x, w, y, n, 4096)",
                        ),
                        before=[],
                        after=[dict(name="torch.ops.vllm.moe_forward.default")],
                    )
                ],
            )
        ],
    )
    new = copy.deepcopy(old)
    new.update(arm="new", path="new.py")
    new["norms"][0].update(source="b.py", config={"R0_BLOCK": 1024})
    new["norms"][0]["uses"][0]["call"].update(
        line=8,
        name="triton_rms_norm_1.run",
        expression="triton_rms_norm_1.run(x, w, y, n, 4096)",
    )
    assert graph_signature(old) == graph_signature(new)
    result = pair_graphs([old, new])
    assert len(result["unique_changed_sources"]) == 1
    assert result["pairs"][0]["norms"][0]["roles"] == ["post_attention_layernorm"]
    with pytest.raises(ValueError, match="ambiguous"):
        pair_graphs([old, old, new])
    new["norms"][0]["uses"][0]["call"]["expression"] = (
        "triton_rms_norm_1.run(x, other_weight, y, n, 4096)"
    )
    with pytest.raises(ValueError, match="correspondence"):
        pair_graphs([old, new])
