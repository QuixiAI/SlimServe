"""N2 audio gate: engine audio path (official checkpoint) vs the runner (dgrauet bf16 pack) on the same
real final audio latent ((1, 8, T, 16), e.g. gate/out/ref_bf16.audio.npy from a 49-frame distilled run).
Reports mel and waveform SNR / max|diff| for: engine vs runner as shipped; engine with the runner's three
deviations from upstream patched in (tanh on the 16 kHz stage, +1e-9 under the STFT sqrt, its edge handling
in the anti-aliased activation) vs runner; plus a mux check of the result.
Usage: PYTHONPATH=<worktree> python n2_audio_parity.py [LATENT.npy] [RUNNER_PACK_DIR] [OUT_DIR]"""
import json, os, subprocess, sys, time, numpy as np, mlx.core as mx
from mlx.utils import tree_map
from slimserve.video.ltx25 import audio, mux
from paths import BF16_PACK, SCRATCH
LAT = sys.argv[1] if len(sys.argv) > 1 else f"{SCRATCH}/gate/out/ref_bf16.audio.npy"
PACK = sys.argv[2] if len(sys.argv) > 2 else BF16_PACK
OUT = sys.argv[3] if len(sys.argv) > 3 else f"{SCRATCH}/n2_audio"
os.makedirs(OUT, exist_ok=True)
lat = mx.array(np.load(LAT)); T = lat.shape[2]
def snr(x, ref, name):
    x, ref = np.asarray(x, np.float64), np.asarray(ref, np.float64)
    assert x.shape == ref.shape, (name, x.shape, ref.shape)
    d = x - ref; s = 10 * np.log10((ref ** 2).sum() / max((d ** 2).sum(), 1e-30))
    print(f"[audio] {name:58s} SNR {s:7.2f} dB  max|diff| {np.abs(d).max():.3e}", flush=True); return s

# ---- runner reference ----
from ltx_pipelines_mlx.utils.blocks import AudioDecoder as RunnerDecoder
rd = RunnerDecoder(PACK); rdec, rvoc = rd.load()
r_mel_bf16 = rdec.decode(lat); mx.eval(r_mel_bf16)
rdec.update(tree_map(lambda p: p.astype(mx.float32), rdec.parameters()))
r_mel = rdec.decode(lat); mx.eval(r_mel)
r_wav_shipped = np.array(rvoc(r_mel_bf16).astype(mx.float32))   # what the runner ships: bf16 VAE -> fp32 vocoder
r_wav = np.array(rvoc(r_mel))                                    # same mel dtype as the engine
r_mel, r_mel_bf16 = np.array(r_mel), np.array(r_mel_bf16.astype(mx.float32))

# ---- engine ----
t0 = time.perf_counter(); dec = audio.AudioDecoder().load(); t_load = time.perf_counter() - t0
tokens = audio.patchify(lat)
mel = audio.decode_mel(dec.w, audio.unpatchify(tokens)); mx.eval(mel); e_mel = np.array(mel)
mx.reset_peak_memory(); t0 = time.perf_counter(); wav, sr = dec.decode(tokens); t_dec = time.perf_counter() - t0
t0 = time.perf_counter(); wav, sr = dec.decode(tokens); t_dec2 = time.perf_counter() - t0
print(f"[audio] engine load {t_load:.2f} s decode {t_dec:.2f} s (warm {t_dec2:.2f} s) peak {mx.get_peak_memory()/2**30:.2f} GiB  "
      f"tokens {T} -> mel {e_mel.shape} -> wav {wav.shape} @ {sr} Hz ({wav.shape[1]/sr:.3f} s)", flush=True)
assert e_mel.shape == (1, 2, 4 * T - 3, 64) and wav.shape == (2, (4 * T - 3) * 480) and sr == 48000
snr(e_mel, r_mel, "mel: engine fp32 vs runner VAE upcast to fp32")
snr(r_mel_bf16, r_mel, "mel: runner bf16 VAE (as shipped) vs runner fp32")
snr(wav, r_wav[0], "wav: engine (upstream semantics) vs runner, same mel dtype")
snr(wav, r_wav_shipped[0], "wav: engine vs runner as shipped (bf16 VAE)")

# ---- engine with the runner's deviations patched in: isolates port errors from semantic differences ----
def runner_upsample(x, kernel, ratio, pad, left, right):
    if ratio != 2: return _orig(x, kernel, ratio, pad, left, right)
    n, t, _ = x.shape; k = kernel.shape[1]
    u = mx.concatenate([x, mx.zeros_like(x)], axis=2).reshape(n, 2 * t, 1)
    return mx.conv1d(audio._edge_pad(u, k // 2, k // 2 - 1), kernel) * 2.0
_orig, audio._sinc_upsample = audio._sinc_upsample, runner_upsample
voc = dec.vocoder; voc.base_cfg = dict(voc.base_cfg, use_tanh_at_final=True)
def log_mel_eps(wavf):
    n_fft, hop = voc.bwe_cfg["n_fft"], voc.bwe_cfg["hop_length"]
    spec = mx.conv1d(mx.pad(wavf[:, :, None], [(0, 0), (n_fft - hop, 0), (0, 0)]), voc.w["vocoder.mel_stft.stft_fn.forward_basis"], stride=hop)
    b = n_fft // 2 + 1
    return mx.log(mx.maximum(mx.sqrt(mx.square(spec[..., :b]) + mx.square(spec[..., b:]) + 1e-9) @ voc.w["vocoder.mel_stft.mel_basis"].T, 1e-5))
voc._log_mel = log_mel_eps
wav_c, _ = dec.decode(tokens)
gate = snr(wav_c, r_wav[0], "wav: engine + runner deviations vs runner (port gate)")
audio._sinc_upsample = _orig

# ---- mux ----
frames = np.random.default_rng(0).integers(0, 255, (round(wav.shape[1] / sr * 24) , 64, 96, 3), dtype=np.uint8)
mp4 = f"{OUT}/mux_check.mp4"; mux.write_mp4(mp4, frames, 24.0, wav, sr); mux.write_wav(f"{OUT}/engine.wav", wav, sr)
mux.write_wav(f"{OUT}/runner.wav", r_wav_shipped[0], sr)
info = json.loads(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-of", "json", mp4], capture_output=True, text=True, check=True).stdout)
for s in info["streams"]:
    print(f"[audio] mux {s['codec_type']}: {s['codec_name']} {s.get('pix_fmt', '')} {s.get('r_frame_rate', '')} frames={s.get('nb_read_frames', s.get('nb_frames'))} "
          f"sr={s.get('sample_rate', '')} ch={s.get('channels', '')} dur={float(s['duration']):.3f}", flush=True)
v = next(s for s in info["streams"] if s["codec_type"] == "video"); a = next(s for s in info["streams"] if s["codec_type"] == "audio")
assert v["codec_name"] == "h264" and int(v["nb_read_frames"]) == len(frames) and v["r_frame_rate"] == "24/1" and a["codec_name"] == "aac"
print(f"[audio] PASS port gate {gate:.1f} dB" if gate > 60 else f"[audio] FAIL port gate {gate:.1f} dB", flush=True)
