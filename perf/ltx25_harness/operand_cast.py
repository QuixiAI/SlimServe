"""Emulate 'fp16/bf16 MMA operands, fp32 glue' inside the MLX runner: cast activations to the weight
dtype at every nn.Linear and to the chosen dtype at every sdpa; everything else (AdaLN, residual, norms,
RoPE, gating) stays whatever the runner makes it (fp32)."""
import mlx.core as mx, mlx.nn as nn
def install(op_dtype):
    _lin = nn.Linear.__call__
    def lin(self, x):
        w = self["weight"]
        if x.dtype != w.dtype: x = x.astype(w.dtype)
        if "bias" in self:
            b = self["bias"]; return mx.addmm(b.astype(w.dtype) if b.dtype != w.dtype else b, x, w.T)
        return x @ w.T
    nn.Linear.__call__ = lin
    _sdpa = mx.fast.scaled_dot_product_attention
    def sdpa(q, k, v, *a, **kw):
        if q.dtype != op_dtype: q, k, v = q.astype(op_dtype), k.astype(op_dtype), v.astype(op_dtype)
        m = kw.get("mask")
        if isinstance(m, mx.array) and m.dtype != mx.bool_ and m.dtype != op_dtype: kw["mask"] = m.astype(op_dtype)
        return _sdpa(q, k, v, *a, **kw)
    mx.fast.scaled_dot_product_attention = sdpa
