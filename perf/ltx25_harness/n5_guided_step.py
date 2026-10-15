"""N5 gate: one guided dev step (4 passes + guider) on identical inputs, runner vs engine.
  capture PACK OUT.npz   runner (dev pack): first guided step's inputs, per-pass x0 and guided x0
  ours OUT.npz           engine on those inputs; prints per-pass and guided rel-L2 / cos
One process per mode. Usage: PYTHONPATH=<worktree> python n5_guided_step.py <mode> ..."""
import sys, numpy as np, mlx.core as mx
PROMPT = "A red fox trotting through a snowy pine forest at dawn, soft golden light, gentle camera dolly forward, birds chirping"
f32 = lambda x: np.array(x.astype(mx.float32))
if sys.argv[1] == "capture":
    from ltx_pipelines_mlx.ti2vid_two_stages import TI2VidTwoStagesPipeline
    from ltx_core_mlx.model.transformer.model import X0Model
    from ltx_core_mlx.components.guiders import MultiModalGuider
    rec = {"passes": [], "guided": []}
    class Done(Exception): pass
    _x0 = X0Model.__call__
    def x0(self, **kw):
        v, a = _x0(self, **kw); mx.eval(v, a)
        if not rec["passes"]:
            rec["in"] = {k: f32(kw[k]) for k in ("video_latent", "audio_latent", "sigma", "video_positions", "audio_positions", "video_keyframes_mask")}
        rec["passes"].append((f32(kw["video_text_embeds"]), f32(kw["audio_text_embeds"]), f32(v), f32(a)))
        return v, a
    _calc = MultiModalGuider.calculate
    def calc(self, *a):
        y = _calc(self, *a); mx.eval(y); rec["guided"].append(f32(y))
        if len(rec["guided"]) == 2: raise Done()
        return y
    X0Model.__call__ = x0; MultiModalGuider.calculate = calc
    pipe = TI2VidTwoStagesPipeline(sys.argv[2], low_memory=True)
    try: pipe.generate_two_stage(PROMPT, height=512, width=768, num_frames=121, frame_rate=24.0, seed=42, stage1_steps=30)
    except Done: pass
    out = {f"in_{k}": v for k, v in rec["in"].items()}
    for i, name in enumerate(("cond", "neg", "ptb", "mod")):
        vt, at, v, a = rec["passes"][i]; out[f"{name}_video"], out[f"{name}_audio"] = v, a
        if name in ("cond", "neg"): out[f"{name}_vtext"], out[f"{name}_atext"] = vt, at
    out["guided_video"], out["guided_audio"] = rec["guided"]
    np.savez(sys.argv[3], **out); print("[n5] captured", len(rec["passes"]), "passes, sigma", out["in_sigma"])
else:
    from slimserve.video.ltx25 import sampling
    from slimserve.video.ltx25.pipeline import LTX25Engine, GuidedDenoiser
    r = np.load(sys.argv[2]); m = lambda k: mx.array(r[k])
    eng = LTX25Engine(variant="dev"); dit = eng.load_dit()
    kf = m("in_video_keyframes_mask")
    video = sampling.LatentState(m("in_video_latent"), m("in_video_latent"), mx.ones((1, kf.shape[1], 1)), m("in_video_positions"), kf)
    audio = sampling.LatentState(m("in_audio_latent"), m("in_audio_latent"), mx.ones((1, r["in_audio_latent"].shape[1], 1)), m("in_audio_positions"))
    sigma = float(r["in_sigma"][0])
    def cmp(name, x, y):
        x, y = np.array(x).astype(np.float64), y.astype(np.float64)
        print(f"[n5] {name:22s} rel-L2 {np.linalg.norm(x-y)/np.linalg.norm(y):.4f} cos {(x*y).sum()/np.linalg.norm(x)/np.linalg.norm(y):.6f}", flush=True)
    for batched in (True, False):
        g = GuidedDenoiser(dit, (m("cond_vtext"), m("cond_atext")), (m("neg_vtext"), m("neg_atext")), sampling.Guidance(cfg=3.0), sampling.Guidance(cfg=7.0), batched)
        if batched:
            v0, a0 = g._forward(video, audio, video.latent, audio.latent, sigma, slice(0, 4)); mx.eval(v0, a0)
            for i, name in enumerate(("cond", "neg", "ptb", "mod")):
                cmp(f"{name} video", v0[i:i+1], r[f"{name}_video"]); cmp(f"{name} audio", a0[i:i+1], r[f"{name}_audio"])
            cmp("runner ptb vs its cond", mx.array(r["ptb_video"]), r["cond_video"]); cmp("runner mod vs its cond", mx.array(r["mod_video"]), r["cond_video"])
        gv, ga = g(video, audio, video.latent, audio.latent, sigma); mx.eval(gv, ga)
        cmp(f"guided video batched={batched}", gv, r["guided_video"]); cmp(f"guided audio batched={batched}", ga, r["guided_audio"])
