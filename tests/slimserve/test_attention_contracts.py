# SPDX-License-Identifier: Apache-2.0
import ast

import pytest

from benchmarks import analyze_glm53_attention_contracts as contracts


def fixtures():
    records, bindings = [], {"combo": [], "split": []}
    for arm, widths in (
        ("combo", [512, 1536, 128]),
        ("split", [512]),
        ("split", [1536]),
        ("split", [128]),
    ):
        symbol = "triton_" + arm + "_" + str(widths[0])
        source = f"def {symbol}():\n    return {widths!r}\n"
        bindings[arm].append(f"{symbol} = async_compile.triton({symbol!r}, {source!r})")
        records.append(
            dict(
                rank=0,
                arm=arm,
                fingerprint=contracts.kernel_fingerprint(source),
                source=symbol + ".py",
                source_sha256="fixture",
                info=dict(widths=widths),
                selected=dict(config={}),
            )
        )
    common = """
class Runner:
    def call(self, args):
        projection = torch.ops.vllm.quixicore_decode_linear.default(h, w, None)
        packed = projection
        kv = empty_strided_cuda((rows, 512), (512, 1), torch.bfloat16)
        q = empty_strided_cuda((rows, 1536), (1536, 1), torch.bfloat16)
        storage = empty_strided_cuda((rows, 256), (256, 1), torch.bfloat16)
        k = reinterpret_tensor(storage, (rows, 128), (256, 1), 0)
CALLS
        projected = torch.ops.vllm.quixicore_decode_linear.default(q, q_weight, None)
        return (kv, projected, storage)
"""
    calls = {
        "combo": (
            "        triton_combo_512.run(packed, wkv, wq, wk, bk, kv, q, k, "
            "rows, rows, rows, stream=stream0)"
        ),
        "split": "\n".join(
            (
                (
                    "        triton_split_512.run(packed, wkv, kv, "
                    "rows, 512, stream=stream0)"
                ),
                (
                    "        triton_split_1536.run(packed, wq, q, "
                    "rows, 1536, stream=stream0)"
                ),
                (
                    "        triton_split_128.run(packed, wk, bk, k, "
                    "rows, 128, stream=stream0)"
                ),
            )
        ),
    }
    texts = []
    for arm in ("combo", "split"):
        body = "\n".join(bindings[arm]) + common.replace("CALLS", calls[arm])
        # Real artifacts embed a second copy as a compile-time string. It is not
        # an executable definition, call or return and must not double coverage.
        texts.append(repr(body) + "\n" + body)
    return texts, records


def test_complete_attention_boundary_mapping():
    texts, records = fixtures()
    result = contracts.compare_pair(*texts, records, 0)
    assert result["outputs"] == dict(
        kv_a_rmsnorm="kv", q_a_rmsnorm="q", indexer_layernorm="k"
    )
    assert result["tensor_definitions"]["k"]["base"]["symbol"] == "storage"
    assert len(result["native_calls"]) == 2
    assert len(result["split"]) == 3


@pytest.mark.parametrize(
    "before,after,error",
    [
        ("packed, wkv, kv", "packed, wq, kv", "512 input/weight/output"),
        ("packed, wk, bk, k", "packed, bk, wk, k", "128 input/weight/output"),
        ("packed, wq, q", "packed, wq, kv", "1536 input/weight/output"),
        ("rows, 128, stream=stream0", "rows, 128, stream=stream1", "launch stream"),
        ("(256, 1), 0)", "(128, 1), 0)", "allocation or alias"),
        (
            "return (kv, projected, storage)",
            "return (q, projected, storage)",
            "return interface",
        ),
        (
            "default(q, q_weight, None)",
            "default(kv, q_weight, None)",
            "native operation",
        ),
    ],
)
def test_changed_boundary_rejected(before, after, error):
    texts, records = fixtures()
    with pytest.raises(ValueError, match=error):
        contracts.compare_pair(texts[0], texts[1].replace(before, after), records, 0)


def test_ambiguous_buffer_definition_rejected():
    function = ast.parse("def call():\n    x = a\n    x = b\n").body[0]
    with pytest.raises(ValueError, match="ambiguous"):
        contracts.tensor_definition(function, "x")


def test_real_retained_graph_contracts():
    if not contracts.MAPPING.exists():
        pytest.skip("local campaign evidence is not installed")
    result = contracts.analyze()
    assert len(result["pairs"]) == 8
    assert result["non_attention_pairs"] == 20
    assert result["historical_indexer_oracle_pass"] is False
    assert result["production_qualified"] is False
