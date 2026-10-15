"""Run the runner's CLI with our precision policy shimmed in: weights cast to fp16 at load, fp16 operands at
linear/sdpa, fp32 glue and fp32 model boundary (LTX_DIT_DTYPE=fp32)."""
import os, sys, mlx.core as mx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import operand_cast; operand_cast.install(mx.float16)
from mlx.utils import tree_map
from ltx_pipelines_mlx import _base
_orig = _base.BasePipeline._load_transformer_with_optional_streaming
def _cast(self, path):
    m = _orig(self, path); m.update(tree_map(lambda p: p.astype(mx.float16) if p.dtype == mx.bfloat16 else p, m.parameters())); mx.eval(m.parameters()); return m
_base.BasePipeline._load_transformer_with_optional_streaming = _cast
from ltx_pipelines_mlx.cli import main
sys.argv = ["ltx-2-mlx"] + sys.argv[1:]; main()
