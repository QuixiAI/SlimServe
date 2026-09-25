# SPDX-License-Identifier: Apache-2.0
"""A checkpoint's advertised KV quantization must not set the engine-wide cache
dtype of a hybrid model. GLM-5.3-Flash's cache holds KDA recurrent states,
pooled indexer rows and (with DFlash2) a drafter's sliding windows beside the
MLA latent; only the latent has an fp8 path, selected per layer through
glm5_next_main_kv_fp8. nvidia's ModelOpt export declares kv_cache_quant_algo
FP8 and the resolver turned the whole cache fp8 (2026-09-22)."""
from types import SimpleNamespace

from vllm.utils.torch_utils import resolve_kv_cache_dtype_string

MODELOPT_FP8_KV = {"quant_method": "modelopt", "quant_algo": "NVFP4", "kv_cache_quant_algo": "FP8"}


def _model(is_hybrid: bool, quant_cfg=MODELOPT_FP8_KV):
    return SimpleNamespace(is_hybrid=is_hybrid, hf_config=SimpleNamespace(quantization_config=quant_cfg))


def test_hybrid_model_keeps_auto_despite_checkpoint_kv_algo():
    assert resolve_kv_cache_dtype_string("auto", _model(is_hybrid=True)) == "auto"


def test_dense_model_still_honors_checkpoint_kv_algo():
    assert resolve_kv_cache_dtype_string("auto", _model(is_hybrid=False)) == "fp8_e4m3"


def test_explicit_operator_dtype_always_wins():
    assert resolve_kv_cache_dtype_string("fp8", _model(is_hybrid=True)) == "fp8"
    assert resolve_kv_cache_dtype_string("bfloat16", _model(is_hybrid=False)) == "bfloat16"


def test_model_without_quant_config_is_auto():
    assert resolve_kv_cache_dtype_string("auto", _model(is_hybrid=False, quant_cfg=None)) == "auto"
