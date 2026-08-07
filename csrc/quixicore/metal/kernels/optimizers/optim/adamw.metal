#include "tk.metal"
#include <metal_stdlib>

using namespace metal;
using namespace mittens;

// ---------------------------------------------------------------------------
// AdamW optimizer step (decoupled weight decay), elementwise over the flattened param:
//   m = b1*m + (1-b1)*g;  v = b2*v + (1-b2)*g^2
//   mhat = m/(1-b1^t);    vhat = v/(1-b2^t)          (bc1=1-b1^t, bc2=1-b2^t from the host)
//   p   -= lr*( mhat/(sqrt(vhat)+eps) + wd*p )        == p*(1-lr*wd) - lr*mhat/(sqrt(vhat)+eps)
// param/grad are T (fp16/bf16/fp32); the moment state m,v is fp32. Produces new param/m/v (the
// MLX/torch host assigns them back), so it is a pure functional kernel. One thread per element.
// ---------------------------------------------------------------------------

template <typename T>
kernel void adamw(device const T     *param     [[buffer(0)]],
                  device const T     *grad      [[buffer(1)]],
                  device const float *m_in      [[buffer(2)]],
                  device const float *v_in      [[buffer(3)]],
                  device T           *param_out [[buffer(4)]],
                  device float       *m_out     [[buffer(5)]],
                  device float       *v_out     [[buffer(6)]],
                  constant float &lr    [[buffer(7)]],
                  constant float &beta1 [[buffer(8)]],
                  constant float &beta2 [[buffer(9)]],
                  constant float &eps   [[buffer(10)]],
                  constant float &wd    [[buffer(11)]],
                  constant float &bc1   [[buffer(12)]],   // 1 - beta1^t
                  constant float &bc2   [[buffer(13)]],   // 1 - beta2^t
                  constant uint  &n     [[buffer(14)]],
                  uint gid [[thread_position_in_grid]]) {
    if (gid >= n) { return; }
    const float g = float(grad[gid]);
    const float m = beta1 * m_in[gid] + (1.0f - beta1) * g;
    const float v = beta2 * v_in[gid] + (1.0f - beta2) * g * g;
    const float mhat = m / bc1;
    const float vhat = v / bc2;
    const float p = float(param[gid]);
    param_out[gid] = T(p - lr * (mhat / (metal::sqrt(vhat) + eps) + wd * p));
    m_out[gid] = m;
    v_out[gid] = v;
}

template <typename T>
kernel void adamw_masked(device const T     *param     [[buffer(0)]],
                         device const T     *grad      [[buffer(1)]],
                         device const float *m_in      [[buffer(2)]],
                         device const float *v_in      [[buffer(3)]],
                         device T           *param_out [[buffer(4)]],
                         device float       *m_out     [[buffer(5)]],
                         device float       *v_out     [[buffer(6)]],
                         constant float &lr        [[buffer(7)]],
                         constant float &beta1     [[buffer(8)]],
                         constant float &beta2     [[buffer(9)]],
                         constant float &eps       [[buffer(10)]],
                         constant float &wd        [[buffer(11)]],
                         constant float &bc1       [[buffer(12)]],
                         constant float &bc2       [[buffer(13)]],
                         constant uint  &n         [[buffer(14)]],
                         device const uchar *mask  [[buffer(15)]],
                         constant uint &seg_size   [[buffer(16)]],
                         constant int &mask_mode   [[buffer(17)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= n) { return; }
    const bool active = mask[gid / seg_size] != 0;
    if (!active && mask_mode == 0) {
        param_out[gid] = param[gid];
        m_out[gid] = m_in[gid];
        v_out[gid] = v_in[gid];
        return;
    }
    const float g = float(grad[gid]);
    const float m = beta1 * m_in[gid] + (1.0f - beta1) * g;
    const float v = beta2 * v_in[gid] + (1.0f - beta2) * g * g;
    const float mhat = m / bc1;
    const float vhat = v / bc2;
    const float p = float(param[gid]);
    const float decay = active ? wd : 0.0f;
    param_out[gid] = T(p - lr * (mhat / (metal::sqrt(vhat) + eps) + decay * p));
    m_out[gid] = m;
    v_out[gid] = v;
}

#define instantiate_adamw(type_name, T)                                          \
  template [[host_name("adamw_" #type_name)]] [[kernel]] void                     \
  adamw<T>(device const T *param [[buffer(0)]], device const T *grad [[buffer(1)]], \
    device const float *m_in [[buffer(2)]], device const float *v_in [[buffer(3)]], \
    device T *param_out [[buffer(4)]], device float *m_out [[buffer(5)]],          \
    device float *v_out [[buffer(6)]], constant float &lr [[buffer(7)]],          \
    constant float &beta1 [[buffer(8)]], constant float &beta2 [[buffer(9)]],     \
    constant float &eps [[buffer(10)]], constant float &wd [[buffer(11)]],        \
    constant float &bc1 [[buffer(12)]], constant float &bc2 [[buffer(13)]],       \
    constant uint &n [[buffer(14)]], uint gid [[thread_position_in_grid]]);

instantiate_adamw(float32, float)
instantiate_adamw(float16, half)
instantiate_adamw(bfloat16, bf16)

#define instantiate_adamw_masked(type_name, T)                                      \
  template [[host_name("adamw_masked_" #type_name)]] [[kernel]] void                \
  adamw_masked<T>(device const T *param [[buffer(0)]], device const T *grad [[buffer(1)]], \
    device const float *m_in [[buffer(2)]], device const float *v_in [[buffer(3)]], \
    device T *param_out [[buffer(4)]], device float *m_out [[buffer(5)]],           \
    device float *v_out [[buffer(6)]], constant float &lr [[buffer(7)]],            \
    constant float &beta1 [[buffer(8)]], constant float &beta2 [[buffer(9)]],       \
    constant float &eps [[buffer(10)]], constant float &wd [[buffer(11)]],          \
    constant float &bc1 [[buffer(12)]], constant float &bc2 [[buffer(13)]],         \
    constant uint &n [[buffer(14)]], device const uchar *mask [[buffer(15)]],       \
    constant uint &seg_size [[buffer(16)]], constant int &mask_mode [[buffer(17)]], \
    uint gid [[thread_position_in_grid]]);

instantiate_adamw_masked(float32, float)
instantiate_adamw_masked(float16, half)
instantiate_adamw_masked(bfloat16, bf16)
