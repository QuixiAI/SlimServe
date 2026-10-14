# ruff: noqa: E501
"""N11: the res_2s sampler against upstream's res2s_audio_video_denoising_loop.

The loop is pure math around a denoiser, so both sides run the same synthetic
denoiser (x0 = 0.5 tanh(x W) + 0.3 x with one fixed W per modality) on the
same initial latents and the same recorded noise; upstream in float64 on the
CPU (its own choice there), ours in fp32. No model, no GPU:

  python n11_res2s_parity.py ref OUT.npz      (conda vllm-mlx)
  python n11_res2s_parity.py ours OUT.npz REF.npz  (venv-slimserve)

Schedules: the 15-step dev schedule at the 1536-token shift ending in 0 (the
HQ stage 1) and STAGE_2_DISTILLED_SIGMAS (the HQ stage 2)."""

import sys

import numpy as np
from paths import REPO, SCRATCH, UP

STUB = f"{SCRATCH}/n11"
NV, NA, C = 96, 12, 128
SEED = 3


def inputs():
    rng = np.random.default_rng(21)
    return (
        rng.standard_normal((1, NV, C)).astype(np.float32),
        rng.standard_normal((1, NA, C)).astype(np.float32),
        (rng.standard_normal((C, C)) / np.sqrt(C)).astype(np.float32),
        (rng.standard_normal((C, C)) / np.sqrt(C)).astype(np.float32),
    )


def schedules():
    from slimserve.video.ltx25 import sampling

    return {
        "stage1": sampling.ltx2_schedule(15, 1536),
        "stage2": list(sampling.STAGE_2_DISTILLED_SIGMAS),
    }


def ref(out):
    import torch

    sys.path[:0] = [f"{UP}/ltx-core/src", f"{UP}/ltx-pipelines/src", STUB]
    import oiio_stub  # noqa: F401
    from ltx_core.types import LatentState
    from ltx_pipelines.utils.samplers import res2s_audio_video_denoising_loop

    sys.path.append(REPO)
    v, a, wv, wa = inputs()
    wv_t, wa_t = torch.from_numpy(wv).double(), torch.from_numpy(wa).double()

    class Result:
        def __init__(self, x):
            self.denoised = x

    def denoiser(transformer, video_state, audio_state, sigmas, step_index):
        xv = video_state.latent.double()
        xa = audio_state.latent.double()
        return Result(0.5 * torch.tanh(xv @ wv_t) + 0.3 * xv), Result(
            0.5 * torch.tanh(xa @ wa_t) + 0.3 * xa
        )

    drawn = []
    real_randn = torch.randn

    def randn(*args, **kw):
        n = real_randn(*args, **kw)
        drawn.append(n.detach().cpu().double().numpy())
        return n

    torch.randn = randn
    res = {}
    for name, sig in schedules().items():
        drawn.clear()

        def state(x):
            t = torch.from_numpy(x).float()
            return LatentState(
                latent=t.clone(),
                clean_latent=t.clone(),
                denoise_mask=torch.ones(1, t.shape[1], 1),
                positions=torch.zeros(1, 3, t.shape[1], 2),
            )

        vs, as_ = res2s_audio_video_denoising_loop(
            sigmas=torch.tensor(sig, dtype=torch.float32),
            video_state=state(v),
            audio_state=state(a),
            transformer=None,
            denoiser=denoiser,
            noise_seed=SEED,
            model_dtype=torch.float32,
        )
        res[f"{name}_video"] = vs.latent.float().numpy()
        res[f"{name}_audio"] = as_.latent.float().numpy()
        for i, d in enumerate(drawn):
            res[f"{name}_noise{i}"] = d
        print(name, "draws", len(drawn))
    np.savez(out, **res)


def ours(out, ref_path):
    import mlx.core as mx

    from slimserve.video.ltx25 import sampling

    v, a, wv, wa = inputs()
    wv_m, wa_m = mx.array(wv), mx.array(wa)

    def denoise(video, audio, vx, ax, sigma, step=None):
        return 0.5 * mx.tanh(vx @ wv_m) + 0.3 * vx, 0.5 * mx.tanh(ax @ wa_m) + 0.3 * ax

    r = np.load(ref_path)
    res = {}
    for name, sig in schedules().items():
        draws = [
            r[f"{name}_noise{i}"]
            for i in range(len([k for k in r if k.startswith(f"{name}_noise")]))
        ]
        queue = list(draws)

        def normal(shape, *args, key=None, queue=queue, **kw):
            d = queue.pop(0)
            assert tuple(d.shape) == tuple(shape), (d.shape, shape)
            return mx.array(d.astype(np.float32))

        real = mx.random.normal
        mx.random.normal = normal
        try:
            state = lambda x: sampling.LatentState(  # noqa: E731
                latent=mx.array(x),
                clean=mx.array(x),
                denoise_mask=mx.ones((1, x.shape[1], 1)),
                positions=mx.zeros((1, x.shape[1], 3)),
            )
            vx, ax = sampling.res2s_loop(denoise, state(v), state(a), sig, SEED)
            mx.eval(vx, ax)
        finally:
            mx.random.normal = real
        assert not queue, f"{len(queue)} upstream draws unused"
        for k, x, y in (
            ("video", vx, r[f"{name}_video"]),
            ("audio", ax, r[f"{name}_audio"]),
        ):
            x = np.array(x).astype(np.float64)
            rel = np.linalg.norm(x - y) / np.linalg.norm(y)
            print(f"{name} {k}: rel-L2 {rel:.2e} max-abs {np.abs(x - y).max():.2e}")
            res[f"{name}_{k}"] = x
    np.savez(out, **res)


if __name__ == "__main__":
    {"ref": ref, "ours": ours}[sys.argv[1]](*sys.argv[2:])
