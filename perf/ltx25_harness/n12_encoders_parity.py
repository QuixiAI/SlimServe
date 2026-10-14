"""N12: the source encoders the editing pipelines need, against upstream on the
CPU in fp32 (ltx_core / ltx_pipelines, PyAV for the media):

  audio-ref OUT.npz CLIP.mp4      PyAV decode_audio_from_file, torchaudio resample
                                  to 16 kHz, AudioProcessor mel, AudioEncoder
  audio-ours OUT.npz REF.npz CLIP our media.read_audio / resample_sinc /
                                  AudioDecoder(encoder=True).encode; compares the
                                  waveform, the 16 kHz waveform, the log-mel and
                                  the tokens
  video-ref OUT.npz CLIP.mp4      PyAV decode_video_from_file, video_preprocess,
                                  VideoEncoder.tiled_encode with a small
                                  TileSizeConfig (frames 24/16, 256/64 px) so a
                                  33-frame 768x512 clip is 3 x 3 x 2 tiles
  video-ours OUT.npz REF.npz CLIP our media.read_frames, image.conditioning
                                  normalization, VideoVAE.encode_tiled; compares
                                  the frames and the latent (and the untiled
                                  encode for scale)
Usage: python n12_encoders_parity.py <mode> ...   (through gpu_run.py)"""

import sys

import numpy as np
from paths import MODELS as ROOT
from paths import SCRATCH, UP

STUB = f"{SCRATCH}/n11"  # oiio_stub
AUDIO_VAE = f"{ROOT}/vae/ltx-2.5-audio-vae-bf16.safetensors"
VIDEO_VAE = f"{ROOT}/vae/ltx-2.5-video-vae-conv-bf16.safetensors"
TILES = ((24, 16), (256, 64))  # (frames tile, overlap), (pixels tile, overlap)


def _upstream():
    sys.path[:0] = [f"{UP}/ltx-core/src", f"{UP}/ltx-pipelines/src", STUB]
    import oiio_stub  # noqa: F401


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12)


def _report(name, a, b):
    gap = np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).max()
    print(f"{name:34s} rel-L2 {_rel(a, b):.2e} max-abs {gap:.2e}")


def audio_ref(out, clip):
    _upstream()
    import torch
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.model.audio_vae.audio_vae import encode_audio
    from ltx_core.model.audio_vae.model_configurator import AudioEncoderConfigurator
    from ltx_core.model.audio_vae.ops import AudioProcessor
    from ltx_pipelines.utils.blocks import AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER
    from ltx_pipelines.utils.media_io.decode import decode_audio_from_file

    audio = decode_audio_from_file(clip, torch.device("cpu"))
    enc = (
        SingleGPUModelBuilder(
            model_path=AUDIO_VAE,
            model_class_configurator=AudioEncoderConfigurator,
            model_sd_ops=AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
        )
        .build(device=torch.device("cpu"), dtype=torch.float32)
        .eval()
    )
    proc = AudioProcessor(enc.sample_rate, enc.mel_bins, enc.mel_hop_length, enc.n_fft)
    with torch.inference_mode():
        wav16 = proc.resample_audio(audio).waveform
        mel = proc.waveform_to_mel(audio)
        latent = encode_audio(audio, enc, proc)
    np.savez(
        out,
        waveform=audio.waveform[0].numpy(),
        rate=audio.sampling_rate,
        wav16=wav16[0].numpy(),
        mel=mel.numpy(),
        latent=latent.float().numpy(),
    )
    print("saved", audio.waveform.shape, audio.sampling_rate, wav16.shape, mel.shape)
    print("latent", latent.shape)


def audio_ours(out, ref_path, clip):
    import mlx.core as mx

    from slimserve.video.ltx25 import audio as audio_mod
    from slimserve.video.ltx25 import media

    r = np.load(ref_path)
    info = media.probe(clip)
    wav, rate = media.read_audio(clip, info)
    assert rate == int(r["rate"]), (rate, r["rate"])
    ref = r["waveform"]
    print("waveform samples ours", wav.shape[1], "ref", ref.shape[1])
    # PyAV hands upstream the AAC priming frame (packet pts -1024, skip-samples
    # side data) as a frame at negative time and upstream trims by that time,
    # so its stream starts `shift` samples into the true timeline; align here.
    n = min(wav.shape[1], ref.shape[1]) - 2048
    shift = min(range(0, 2048), key=lambda k: _rel(wav[:, k : k + n], ref[:, :n]))
    print("upstream's stream starts", shift, "samples into ours")
    _report("waveform (aligned)", wav[:, shift : shift + n], ref[:, :n])
    wav16 = media.resample_sinc(wav[:, shift : shift + ref.shape[1]], rate, 16000)
    m = min(wav16.shape[1], r["wav16"].shape[1])
    _report("16 kHz waveform (aligned)", wav16[:, :m], r["wav16"][:, :m])
    dec = audio_mod.AudioDecoder(encoder=True).load()
    # the same input: the front end alone
    mel = audio_mod.waveform_to_mel(r["wav16"], dec.cfg)
    _report("log-mel on upstream's 16 kHz", mel, r["mel"])
    same_mel = audio_mod.patchify(audio_mod.encode_mel(dec.w, mx.array(r["mel"])))
    mx.eval(same_mel)
    _report(
        "encoder on upstream's mel",
        np.array(same_mel),
        np.array(audio_mod.patchify(mx.array(r["latent"]))),
    )
    tokens = dec.encode(wav[:, shift : shift + ref.shape[1]], rate)
    mx.eval(tokens)
    ref_tokens = np.array(audio_mod.patchify(mx.array(r["latent"])))
    _report("tokens (aligned, end to end)", np.array(tokens), ref_tokens)
    np.savez(out, tokens=np.array(tokens), shift=shift)


def video_ref(out, clip):
    _upstream()
    import torch
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.model.video_vae.model_configurator import VideoEncoderConfigurator
    from ltx_core.tiling import DimensionSizeConfig, TileSizeConfig
    from ltx_pipelines.utils.blocks import VAE_ENCODER_COMFY_KEYS_FILTER
    from ltx_pipelines.utils.media_io.decode import (
        decode_video_from_file,
        get_videostream_metadata,
        video_preprocess,
    )

    shape = get_videostream_metadata(clip)
    frames_u8 = []
    for f in decode_video_from_file(clip, torch.device("cpu")):
        frames_u8.append(f[0].numpy())
    frames_u8 = np.stack(frames_u8)
    pixels = video_preprocess(
        (torch.from_numpy(f)[None] for f in frames_u8),
        shape.height,
        shape.width,
        torch.float32,
        torch.device("cpu"),
    )
    enc = (
        SingleGPUModelBuilder(
            model_path=VIDEO_VAE,
            model_class_configurator=VideoEncoderConfigurator,
            model_sd_ops=VAE_ENCODER_COMFY_KEYS_FILTER,
        )
        .build(device=torch.device("cpu"), dtype=torch.float32)
        .eval()
    )
    cfg = TileSizeConfig(
        frames=DimensionSizeConfig(*TILES[0]),
        height=DimensionSizeConfig(*TILES[1]),
        width=DimensionSizeConfig(*TILES[1]),
    )
    with torch.inference_mode():
        tiled = enc.tiled_encode(pixels, cfg)
        plain = enc(pixels)
    np.savez(
        out,
        frames=frames_u8,
        pixels=pixels.numpy(),
        tiled=tiled.float().numpy(),
        plain=plain.float().numpy(),
    )
    print("saved", frames_u8.shape, tiled.shape)


def video_ours(out, ref_path, clip):
    import mlx.core as mx

    from slimserve.video.ltx25 import image as image_mod
    from slimserve.video.ltx25 import media
    from slimserve.video.ltx25.vae import Tiling, VideoVAE

    r = np.load(ref_path)
    info = media.probe(clip)
    frames = media.read_frames(clip, info)
    print("frames ours", frames.shape, "ref", r["frames"].shape)
    diff = np.abs(frames.astype(np.int16) - r["frames"].astype(np.int16))
    print(f"frames: {np.mean(diff > 0) * 100:.3f}% pixels differ, max {diff.max()}")
    pixels = np.stack(
        [
            image_mod.resize_and_center_crop(
                f.astype(np.float32), info.height, info.width
            )
            for f in frames
        ]
    )
    pixels = mx.array((pixels / 127.5 - 1.0).transpose(3, 0, 1, 2))[None]
    _report("pixels", np.array(pixels), r["pixels"])
    vae = VideoVAE().load()
    tiled = vae.encode_tiled(
        mx.array(r["pixels"]), Tiling(spatial=TILES[1], temporal=TILES[0])
    )
    mx.eval(tiled)
    _report("tiled encode (upstream pixels)", np.array(tiled), r["tiled"])
    plain = vae.encode(mx.array(r["pixels"]))
    mx.eval(plain)
    _report("untiled encode (upstream pixels)", np.array(plain), r["plain"])
    _report("tiled vs untiled, upstream", r["tiled"], r["plain"])
    ours_e2e = vae.encode_tiled(pixels, Tiling(spatial=TILES[1], temporal=TILES[0]))
    mx.eval(ours_e2e)
    _report("tiled encode end to end", np.array(ours_e2e), r["tiled"])
    np.savez(out, tiled=np.array(ours_e2e))


if __name__ == "__main__":
    {
        "audio-ref": audio_ref,
        "audio-ours": audio_ours,
        "video-ref": video_ref,
        "video-ours": video_ours,
    }[sys.argv[1]](*sys.argv[2:])
