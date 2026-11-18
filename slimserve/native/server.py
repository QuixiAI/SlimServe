"""OpenAI-compatible server for the q38 native engine.

    python -m slimserve.native.server --model <hf dir or id> \\
        --q38-pack weights.q38w [--q38-dspine-pack drafter.q38w | --q38-mtp-pack mtp.q38w] [vLLM front-end flags]

The whole vLLM front end is reused unchanged (chat template, tool-call and
reasoning parsers, streaming, the Chat / Completions / Responses endpoints).
Only the engine behind `EngineClient` is different: instead of vLLM's
scheduler and model executor, requests go to the q38 library (native/q38),
which runs Qwen3.8-27B on Apple Metal with its own speculative decoding (the
model's MTP head) and prefix snapshots.

Concurrent requests share the engine: every pass over the weights carries
rows of all running requests (decode rows, MTP drafts while few requests
decode, and prompt chunks), up to --q38-max-streams at once; further requests
wait for a free stream. Prefix caching is always on: every prompt end and
every chat turn boundary leaves a snapshot, a finished conversation's sequence
stays alive, and a prompt resumes from the longest of those that is a prefix
of it.

Images and videos are supported: the vision tower runs in this process
(PyTorch, slimserve/native/vision.py) and the engine takes its rows as input
embeddings.

Not supported, and rejected or ignored explicitly rather than silently:
logprobs (rejected), n > 1 (rejected), structured
outputs / logit bias / repetition penalties / min_p (ignored with a warning).
"""
from __future__ import annotations

import asyncio
import os
import queue
import random
import signal
import threading
import time
from collections.abc import AsyncGenerator, Iterable, Mapping
from typing import Any

import uvloop

from vllm.config import VllmConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.api_server import build_and_serve, setup_server
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.renderers import renderer_from_config
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason
from vllm.v1.engine.exceptions import EngineDeadError
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.engine.output_processor import OutputProcessor, RequestOutputCollector
from vllm.v1.metrics.stats import PrefillStats

from slimserve.native.q38 import Engine, Pool

logger = init_logger(__name__)

_SUPPORTED_TASKS = ("generate",)


class Q38EngineClient(EngineClient):
    """`EngineClient` over one q38 engine. All engine calls run on one worker thread."""

    # The worker thread owns the engine, the pool and the vision tower. Requests
    # talk to it through `_cmds` and get their tokens back on an asyncio queue.

    def __init__(self, vllm_config: VllmConfig, args) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.renderer = renderer_from_config(vllm_config)
        self.input_processor = InputProcessor(vllm_config, self.renderer)
        self.output_processor = OutputProcessor(self.renderer.tokenizer, log_stats=False, stream_interval=1)
        self._cmds: queue.SimpleQueue = queue.SimpleQueue()
        self._streams: dict[int, tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = {}
        self._free: asyncio.Semaphore | None = None
        self._dead: BaseException | None = None
        self._stopped = False
        self._boundary = self._turn_boundary_token()
        self._warned: set[str] = set()
        self._stats = {"requests": 0, "prompt_tokens": 0, "cached_tokens": 0, "output_tokens": 0,
                       "decode_s": 0.0, "prefill_s": 0.0}

        here = os.path.dirname(os.path.abspath(__file__))
        build = args.q38_build_dir or os.path.join(here, "..", "..", "native", "q38", "build")
        build = os.path.abspath(build)
        metallibs = ":".join(os.path.join(build, n) for n in ("q38_kernels.metallib", "q38_native.metallib"))
        self._engine = Engine(
            os.path.join(build, "libq38.dylib"), os.path.expanduser(args.q38_pack), metallibs,
            mtp_pack=os.path.expanduser(args.q38_mtp_pack) if args.q38_mtp_pack else None,
            dspine_pack=os.path.expanduser(args.q38_dspine_pack) if args.q38_dspine_pack else None,
            max_ctx=self.model_config.max_model_len)
        self._pool = Pool(self._engine, max_streams=args.q38_max_streams, depth=args.q38_draft_depth,
                          adaptive=args.q38_draft_adaptive,
                          max_cache_bytes=int(args.q38_snapshot_gb * (1 << 30)),
                          big_every=args.q38_big_prefill_every)
        logger.info("q38 engine ready: context %d, %d concurrent streams, drafting %s",
                    self.model_config.max_model_len, args.q38_max_streams,
                    "DSpine blocks" if self._engine.has_dspine
                    else f"MTP depth {args.q38_draft_depth}" if self._engine.has_mtp else "off")
        self._vision = None  # loaded on the first image request
        self._grids: dict[str, tuple[int, int]] = {}
        self._worker = threading.Thread(target=self._run, name="q38", daemon=True)
        self._worker.start()
        self._model_dir = self.model_config.model

    # ---- helpers ----

    def _turn_boundary_token(self) -> int | None:
        """Token id that opens a chat turn; snapshots are left just before each one."""
        try:
            ids = self.renderer.tokenizer.encode("<|im_start|>", add_special_tokens=False)
        except Exception:  # tokenizer without the ChatML marker
            return None
        return ids[0] if len(ids) == 1 else None

    def _warn_once(self, key: str, msg: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            logger.warning(msg)

    def _snapshot_marks(self, ids: list[int]) -> list[int]:
        """Prompt lengths worth a snapshot: the last few turn starts."""
        if self._boundary is None:
            return []
        marks = [i for i, t in enumerate(ids) if t == self._boundary and i > 0]
        return marks[-3:]

    def _check(self, params: SamplingParams, request: EngineCoreRequest) -> None:
        for f in request.mm_features or ():
            if f.modality not in ("image", "video"):
                raise ValueError(f"the q38 native engine accepts image and video inputs only, not {f.modality}")
        if params.n != 1:
            raise ValueError("the q38 native engine supports n = 1 only")
        if params.logprobs is not None or params.prompt_logprobs is not None:
            raise ValueError("the q38 native engine does not return logprobs")
        if params.structured_outputs is not None:
            self._warn_once("so", "structured outputs are not enforced by the q38 native engine; ignored")
        if params.repetition_penalty not in (None, 1.0) or params.presence_penalty or params.frequency_penalty:
            self._warn_once("pen", "repetition / presence / frequency penalties are not applied by the q38 engine")
        if params.logit_bias or params.min_p:
            self._warn_once("bias", "logit_bias and min_p are not applied by the q38 engine")

    def _plan_images(self, ids: list[int], features):
        """Worker thread: encode the request's images and build the engine inputs."""
        from slimserve.native import vision
        if self._vision is None:
            path = self._model_dir
            if not os.path.isdir(path):
                from huggingface_hub import snapshot_download
                path = snapshot_download(path, revision=self.model_config.revision)
            self._vision = vision.VisionTower(path)
            logger.info("q38 vision tower loaded on %s", self._vision.device)
        images = []
        for f in features:
            rows = self._vision.cached(f.identifier)
            if f.data is None and rows is None:
                raise ValueError("image data missing for an image this engine has not seen")
            if f.data is not None:
                d = f.data.get_data()
                video = f.modality == "video"
                gkey, pkey = ("video_grid_thw", "pixel_values_videos") if video else ("image_grid_thw", "pixel_values")
                grid = d[gkey].reshape(-1).tolist()
                if rows is None:
                    rows = self._vision.encode(f.identifier, d[pkey], d[gkey])
                h, w = grid[1] // self._vision.merge, grid[2] // self._vision.merge
                # a video's rows come as grid[0] frame groups of h x w
                self._grids[f.identifier] = (grid[0], h, w) if video else (h, w)
            mask = f.mm_position.is_embed
            images.append((f.mm_position.offset, f.mm_position.length,
                           None if mask is None else mask.numpy(), f.identifier, rows, self._grids[f.identifier]))
        return vision.plan(ids, images)

    # ---- worker thread ----

    def _run(self) -> None:
        try:
            while True:
                try:  # block for work when nothing is running
                    cmd = self._cmds.get() if self._pool.active == 0 else self._cmds.get_nowait()
                except queue.Empty:
                    cmd = None
                while cmd is not None:
                    if cmd[0] == "stop":
                        return
                    if cmd[0] == "open":
                        self._open(*cmd[1:])
                    elif cmd[0] == "release":
                        sid = cmd[1].get("sid")
                        if sid is not None:
                            self._pool.close_stream(sid)
                            self._streams.pop(sid, None)
                    try:
                        cmd = self._cmds.get_nowait()
                    except queue.Empty:
                        cmd = None
                if self._pool.active:
                    for ev in self._pool.step():
                        ent = self._streams.get(ev.stream)
                        if ent is not None:
                            ent[0].call_soon_threadsafe(ent[1].put_nowait, ev)
        except BaseException as exc:  # the engine itself failed: nothing after this is trustworthy
            self._dead = exc
            logger.exception("q38 worker died")
            for loop, q in self._streams.values():
                loop.call_soon_threadsafe(q.put_nowait, exc)
            while True:  # requests still waiting to open must not hang
                try:
                    cmd = self._cmds.get_nowait()
                except queue.Empty:
                    break
                if cmd[0] == "open":
                    loop, fut = cmd[1], cmd[3]
                    loop.call_soon_threadsafe(lambda f=fut: None if f.done() else f.set_exception(EngineDeadError()))

    def _open(self, loop, q, fut, state, ids, features, kwargs) -> None:
        def settle(fn, value) -> None:
            loop.call_soon_threadsafe(lambda: None if fut.done() else fn(value))

        try:
            if features:
                tokens, embeds, pos3 = self._plan_images(ids, features)
                sid = self._pool.open(tokens.tolist(), embeds=embeds, pos3=pos3, **kwargs)
            else:
                sid = self._pool.open(ids, **kwargs)
            if sid == Pool.BUSY:
                raise ValueError("no free q38 stream")  # admission is counted in generate(); not expected
        except ValueError as exc:  # this request is at fault, the engine is fine
            settle(fut.set_exception, exc)
            return
        state["sid"] = sid
        self._streams[sid] = (loop, q)
        settle(fut.set_result, sid)

    # ---- EngineClient: the part the front end uses ----

    @property
    def is_running(self) -> bool:
        return not self._stopped and self._dead is None

    @property
    def is_stopped(self) -> bool:
        return self._stopped

    @property
    def errored(self) -> bool:
        return self._dead is not None

    @property
    def dead_error(self) -> BaseException:
        return EngineDeadError() if self._dead is None else self._dead

    async def get_supported_tasks(self):
        return _SUPPORTED_TASKS

    async def generate(
        self,
        prompt,
        sampling_params: SamplingParams,
        request_id: str,
        *,
        prompt_text: str | None = None,
        lora_request=None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
        reasoning_ended: bool | None = None,
        reasoning_parser_kwargs: dict[str, Any] | None = None,
    ) -> AsyncGenerator[RequestOutput, None]:
        if self._dead is not None:
            raise EngineDeadError() from self._dead
        if isinstance(prompt, EngineCoreRequest):
            request = prompt
        else:
            request = self.input_processor.process_inputs(
                request_id, prompt, sampling_params, supported_tasks=_SUPPORTED_TASKS,
                arrival_time=time.time(), lora_request=lora_request,
                tokenization_kwargs=tokenization_kwargs, trace_headers=trace_headers, priority=priority)
        self.input_processor.assign_request_id(request)
        params = request.sampling_params
        assert params is not None
        self._check(params, request)
        ids = list(request.prompt_token_ids or [])
        if not ids:
            raise ValueError("empty prompt")

        rid = request.request_id
        out_q = RequestOutputCollector(params.output_kind, rid)
        self.output_processor.add_request(request, prompt_text, None, 0, out_q)
        loop = asyncio.get_running_loop()
        eos = {t for t in (params.eos_token_id, *(params.stop_token_ids or [])) if t is not None}
        # the engine only needs to know the ids that end the model's own turn
        engine_eos = [t for t in (248046, 248044) if not params.ignore_eos]
        seed = params.seed if params.seed is not None else random.getrandbits(63)
        top_k = params.top_k if params.top_k and params.top_k > 0 else 64
        max_tokens = params.max_tokens if params.max_tokens is not None else (
            self.model_config.max_model_len - len(ids))
        if self._free is None:
            self._free = asyncio.Semaphore(self._pool.max_streams)
        state: dict[str, int] = {}
        opened = False
        try:
            async with self._free:
                t0 = time.perf_counter()
                events: asyncio.Queue = asyncio.Queue()
                fut = loop.create_future()
                opened = True
                self._cmds.put(("open", loop, events, fut, state, ids, request.mm_features, dict(
                    temperature=params.temperature, top_k=top_k, top_p=params.top_p, seed=seed,
                    eos=engine_eos, max_tokens=max_tokens, snapshot_at=self._snapshot_marks(ids))))
                await fut
                self._stats["requests"] += 1
                self._stats["prompt_tokens"] += len(ids)
                prefill: PrefillStats | None = None
                t1 = None
                produced = 0
                finished = False
                while not finished:
                    ev = await events.get()
                    if isinstance(ev, BaseException):
                        raise EngineDeadError() from ev
                    if t1 is None and (ev.prompt_done or ev.tokens):
                        t1 = time.perf_counter()
                        self._stats["cached_tokens"] += ev.reused
                        self._stats["prefill_s"] += t1 - t0
                        prefill = PrefillStats(
                            num_prompt_tokens=len(ids), num_computed_tokens=len(ids) - ev.reused,
                            num_cached_tokens=ev.reused, num_local_cached_tokens=ev.reused)
                    toks = ev.tokens
                    reason: FinishReason | None = None
                    stop_reason: int | None = None
                    if ev.done == 1:
                        reason = FinishReason.STOP
                    elif ev.done == 2:
                        reason = FinishReason.LENGTH
                    for k, t in enumerate(toks):  # request-level stop ids end the turn too
                        if t in eos and not params.ignore_eos:
                            toks = toks[: k + 1]
                            reason = FinishReason.STOP
                            stop_reason = t if t != params.eos_token_id else None
                            break
                    if not toks and reason is None:
                        continue
                    produced += len(toks)
                    out = EngineCoreOutput(request_id=rid, new_token_ids=toks, finish_reason=reason,
                                           stop_reason=stop_reason, prefill_stats=prefill)
                    prefill = None
                    processed = self.output_processor.process_outputs([out])
                    finished = reason is not None or bool(processed.reqs_to_abort)  # stop string hit
                    while True:
                        ro = out_q.get_nowait()
                        if ro is None:
                            break
                        if isinstance(ro, Exception):
                            raise ro
                        finished = finished or ro.finished
                        yield ro
                self._stats["output_tokens"] += produced
                if t1 is not None:
                    self._stats["decode_s"] += time.perf_counter() - t1
        except (asyncio.CancelledError, GeneratorExit, ValueError, EngineDeadError):
            self.output_processor.abort_requests([rid], internal=True)
            raise
        except BaseException as exc:
            self.output_processor.abort_requests([rid], internal=True)
            raise EngineDeadError() from exc
        finally:
            if opened:  # runs after the open on the worker, so it sees the stream id
                self._cmds.put(("release", state))

    async def is_tracing_enabled(self) -> bool:
        return False

    async def do_log_stats(self) -> None:
        s = self._stats
        if s["requests"]:
            logger.info(
                "q38: %d requests, prompt %d tokens (%.1f%% reused), output %d tokens, "
                "%.1f tok/s per request while decoding", s["requests"], s["prompt_tokens"],
                100.0 * s["cached_tokens"] / max(1, s["prompt_tokens"]), s["output_tokens"],
                s["output_tokens"] / s["decode_s"] if s["decode_s"] else 0.0)

    async def check_health(self) -> None:
        if self._dead is not None:
            raise EngineDeadError() from self._dead

    def shutdown(self, timeout: float | None = None) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._cmds.put(("stop",))
        self._worker.join(timeout=timeout if timeout is not None else 30)
        if not self._worker.is_alive():  # never free the engine under a running pass
            self._pool.close()
            self._engine.close()

    # ---- EngineClient: not applicable to this engine ----

    async def abort(self, request_id: str | Iterable[str]) -> None:
        # requests stop when their generator is cancelled; nothing runs in the background
        return None

    def encode(self, *args, **kwargs):
        raise NotImplementedError("the q38 native engine is generate-only")

    async def notify_kv_transfer_request_rejected(self, request_id, kv_transfer_params, *,
                                                  data_parallel_rank=None) -> None:
        return None

    async def start_profile(self) -> None:
        raise NotImplementedError

    async def stop_profile(self) -> None:
        raise NotImplementedError

    async def reset_mm_cache(self) -> None:
        return None

    async def reset_encoder_cache(self) -> None:
        return None

    async def reset_prefix_cache(self, *args, **kwargs) -> None:
        raise NotImplementedError("q38 prefix snapshots are evicted by size, not reset on demand")

    async def sleep(self, level: int = 1, mode="abort") -> None:
        raise NotImplementedError

    async def wake_up(self, tags: list[str] | None = None) -> None:
        raise NotImplementedError

    async def is_sleeping(self) -> bool:
        return False

    async def add_lora(self, lora_request) -> bool:
        raise NotImplementedError("the q38 native engine has no LoRA support")

    async def pause_generation(self, *args, **kwargs) -> None:
        raise NotImplementedError

    async def resume_generation(self) -> None:
        raise NotImplementedError

    async def is_paused(self) -> bool:
        return False


def make_parser() -> FlexibleArgumentParser:
    parser = make_arg_parser(FlexibleArgumentParser(description="SlimServe q38 native engine server"))
    g = parser.add_argument_group("q38 native engine")
    g.add_argument("--q38-pack", required=True, help="weight pack (.q38w)")
    g.add_argument("--q38-mtp-pack", default=None, help="MTP draft head pack; enables speculative decoding")
    g.add_argument("--q38-dspine-pack", default=None,
                   help="DSpine block drafter pack (tools/pack_dspine.py); speculative decoding with it "
                        "instead of the MTP head")
    g.add_argument("--q38-build-dir", default=None, help="directory with libq38.dylib and the metallibs")
    g.add_argument("--q38-draft-depth", type=int, default=2, help="MTP drafts per decode cycle")
    g.add_argument("--q38-draft-adaptive", type=float, default=0.5,
                   help="stop drafting once a draft's own probability is below this")
    g.add_argument("--q38-snapshot-gb", type=float, default=16.0,
                   help="host memory for prefix snapshots and idle sequences")
    g.add_argument("--q38-big-prefill-every", type=int, default=None,
                   help="64-row prompt passes (twice the prompt tokens per second, nothing else rides in "
                        "them): -1 never, 0 only while no request is decoding, N also one after every N "
                        "ordinary passes while requests decode (default: the engine's, 2)")
    g.add_argument("--q38-max-streams", type=int, default=8, choices=range(1, 9),
                   help="requests decoded at the same time; more wait for a free stream")
    return parser


async def run(args) -> None:
    def _interrupt_init(*_) -> None:
        raise KeyboardInterrupt("terminated")

    signal.signal(signal.SIGTERM, _interrupt_init)
    signal.signal(signal.SIGINT, _interrupt_init)
    listen_address, sock = setup_server(args, reuse_port=False)
    # Front-end configuration only: no weights are loaded and no device is touched here.
    model_config = AsyncEngineArgs.from_cli_args(args).create_model_config()
    model_config.quantization = None
    # One process: there is no engine-side multimodal cache to rely on, so the
    # front end must always pass image data through (encoded rows are cached by
    # content hash in slimserve/native/vision.py).
    if model_config.multimodal_config is not None:
        model_config.multimodal_config.mm_processor_cache_gb = 0
    vllm_config = VllmConfig(model_config=model_config)
    client = Q38EngineClient(vllm_config, args)
    try:
        shutdown_task = await build_and_serve(client, listen_address, sock, args)
        await shutdown_task
    finally:
        client.shutdown()
        sock.close()


def main() -> None:
    args = make_parser().parse_args()
    validate_parsed_serve_args(args)
    uvloop.run(run(args))


if __name__ == "__main__":
    main()
