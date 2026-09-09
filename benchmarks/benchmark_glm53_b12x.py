#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fixed-start, stock-launcher B12X R28.1 competitive control on this TP4 box.

Different checkpoint, W4A4 activations and FP8 KV: not precision-matched to the
SlimServe recipe. No performance-based restart, exclusion or early stopping.
Only containers created by this invocation are stopped/removed. Their logs,
configuration, exit/OOM state, source lock and all workload receipts survive.
Run the controller in a memory-limited systemd scope; Docker is separately capped.
"""

import argparse
import hashlib
import json
import os
import secrets
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from benchmark_glm53_campaign import LOADED_BENCHMARK_SOURCES, require_benchmark_sources
from benchmark_glm53_server import run as run_workload
from benchmark_glm53_server import tokenizer_receipt

from slimserve.server import free_port

IMAGE = (
    "voipmonitor/vllm@sha256:"
    "52ef7badcc33918f276d778d29bd972a798297584ba776476c7c09b7bdb50e5f"
)
REVISION = "46aaae8a82032f77100f2f03e9cc11b391df3b4d"
LOCK_SHA256 = "4473b46dbf696a386da1fbd6f75e7ef9159c36d216d153c6beb5cfe68b7a7477"
MODEL_NAME = "GLM-5.3-Flash-NVFP4"
FA2_DESTINATION = "/opt/glm53-flash/vllm/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so"
ORIGINAL_FA2_SHA256 = "37e3a34edb6dbd3d86a7952b3ec9975cb38e3a56a2381cdab9e302eaefe4ff81"


def command(*argv, timeout=30):
    return subprocess.check_output(argv, text=True, timeout=timeout).strip()


def fa2_receipt(args):
    """A binary override must carry both exact-source qualification arms."""
    if args.fa2_library is None and args.fa2_qualification is None:
        return None
    if (
        not args.host_cuda_driver
        or args.fa2_library is None
        or args.fa2_qualification is None
    ):
        raise ValueError(
            "FA2 override requires host driver, library and qualification bundle"
        )
    library = args.fa2_library.resolve(strict=True)
    bundle = args.fa2_qualification.resolve(strict=True)
    original_path = bundle / "original-compat" / "summary.json"
    native_path = bundle / "native-host" / "summary.json"
    original = json.loads(original_path.read_text())
    native = json.loads(native_path.read_text())
    with library.open("rb") as handle:
        library_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    probe = Path(__file__).parent / "kernels" / "probe_b12x_fa2.py"
    if (
        original["status"] != "complete"
        or native["status"] != "complete"
        or original["binary"]["sha256"] != ORIGINAL_FA2_SHA256
        or native["binary"]["sha256"] != library_hash
        or original["schema"] != native["schema"]
        or native["reference"]["sha256"]
        != hashlib.sha256(original_path.read_bytes()).hexdigest()
        or original["probe_sha256"] != native["probe_sha256"]
        or native["probe_sha256"] != hashlib.sha256(probe.read_bytes()).hexdigest()
        or len(original["cases"]) != 30
        or len(native["cases"]) != 30
    ):
        raise ValueError("FA2 qualification identity or completeness mismatch")
    for before, after in zip(original["cases"], native["cases"]):
        if (
            before["status"] != "passed"
            or after["status"] != "passed"
            or not before["oracle"]["passed"]
            or not after["oracle"]["passed"]
            or not before["graph_exact"]
            or not after["graph_exact"]
            or not after["prior_binary"]["passed"]
            or before["input_sha256"] != after["input_sha256"]
        ):
            raise ValueError("FA2 numerical or changed-input graph gate failed")
    return {
        "path": str(library),
        "sha256": library_hash,
        "image_destination": FA2_DESTINATION,
        "qualification_directory": str(bundle),
        "original_qualification_sha256": hashlib.sha256(
            original_path.read_bytes()
        ).hexdigest(),
        "native_qualification_sha256": hashlib.sha256(
            native_path.read_bytes()
        ).hexdigest(),
        "label": "host-driver and exact-source native-SM120 FA2 adapted image",
    }


def docker_create_args(args, name, port):
    model_cache = args.model_cache.resolve(strict=True)
    if not (model_cache / "snapshots" / REVISION / "config.json").is_file():
        raise ValueError("the pinned B12X checkpoint is absent from model-cache")
    jit_cache = args.jit_cache.resolve(strict=True)
    if (
        jit_cache == model_cache
        or model_cache in jit_cache.parents
        or jit_cache in model_cache.parents
    ):
        raise ValueError("reference JIT and read-only model caches must be separate")
    environment = {
        "MODEL": f"/model/snapshots/{REVISION}",
        "MODEL_REVISION": REVISION,
        "SERVED_MODEL_NAME": MODEL_NAME,
        "HOST": "127.0.0.1",
        "PORT": str(port),
        "TP": "4",
        "DCP": "1",
        "SPECULATOR": "mtp",
        "MTP_DEPTH": "0",
        "CACHE_MODE": "vram",
        "KV_CACHE_QUANT": "fp8_ds_mla",
        "B12X_PCIE_ALLREDUCE": "1",
        "NCCL_P2P_DISABLE": "0",
        "NCCL_P2P_LEVEL": "SYS",
        "HF_HOME": "/cache/huggingface",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "MAX_JOBS": "2",
        "NVCC_THREADS": "2",
        "OMP_NUM_THREADS": "1",
    }
    if args.diagnostic_nccl:
        # Observation only: never disable transports or replace the library.
        environment["NCCL_DEBUG"] = "INFO"
    if args.host_cuda_driver:
        # The image's automatic shell hook prepends compat/lib.real, whose
        # CUDA IPC import fails on this host. Keep the injected host libcuda
        # instead. The launcher, CUDA runtime, NCCL and P2P remain unchanged.
        environment["BASH_ENV"] = "/dev/null"
    argv = [
        "docker",
        "create",
        "--name",
        name,
        "--init",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--gpus",
        "all",
        "--memory",
        "150g",
        "--memory-swap",
        "150g",
        "--shm-size",
        "16g",
        "--pids-limit",
        "2048",
        "--ulimit",
        "core=0",
        "--ulimit",
        "memlock=-1:-1",
        "--network",
        "host",
        "--label",
        "slimserve.campaign=b12x-r281-fixed-start",
        "--mount",
        f"type=bind,src={model_cache},dst=/model,readonly",
        "--mount",
        f"type=bind,src={jit_cache},dst=/cache",
    ]
    if adapted := fa2_receipt(args):
        argv += [
            "--mount",
            f"type=bind,src={adapted['path']},dst={FA2_DESTINATION},readonly",
        ]
    for key, value in environment.items():
        argv += ["-e", f"{key}={value}"]
    return argv + [
        IMAGE,
        "--enable-prompt-tokens-details",
        "--enable-per-request-metrics",
    ]


def stop_container(container_id):
    """Use the immutable ID returned by our successful docker create, not a name."""
    before = json.loads(command("docker", "inspect", container_id))[0]["State"]
    if before["Running"]:
        command("docker", "stop", "--time", "30", container_id, timeout=45)
    after = json.loads(command("docker", "inspect", container_id))[0]["State"]
    if after["Running"]:
        raise RuntimeError(f"owned container is still running: {container_id}")
    return {"before_stop": before, "after_stop": after}


def run(args):
    if min(args.boots, args.repeats, args.startup_timeout) < 1:
        raise ValueError("positive starts, repeats and startup timeout required")
    args.output.mkdir(parents=True, exist_ok=False)
    args.jit_cache.mkdir(parents=True, exist_ok=True)
    record = args.output / "summary.json"
    receipt = {
        "status": "preparing",
        "benchmark_implementation_sha256": dict(LOADED_BENCHMARK_SOURCES),
        "method": __doc__,
        "image": IMAGE,
        "model_revision": REVISION,
        "diagnostic_only": args.diagnostic_nccl or getattr(args, "canary_only", False),
        "canary_only": getattr(args, "canary_only", False),
        "cold_prefix": getattr(args, "cold_prefix", False),
        "host_cuda_driver": args.host_cuda_driver,
        "fa2_override": fa2_receipt(args),
        "command": sys.argv,
        "git_commit": command("git", "rev-parse", "HEAD"),
        "git_status": command("git", "status", "--short"),
        "runs": [],
    }

    def save():
        record.write_text(json.dumps(receipt, indent=2) + "\n")

    save()
    try:
        require_benchmark_sources()
        receipt["image_inspect"] = json.loads(
            command("docker", "image", "inspect", IMAGE)
        )[0]
        lock = subprocess.check_output(
            [
                "docker",
                "run",
                "--rm",
                "--memory",
                "256m",
                "--memory-swap",
                "256m",
                "--network",
                "none",
                "--entrypoint",
                "/bin/cat",
                IMAGE,
                "/opt/glm53-flash/source.lock",
            ],
            timeout=30,
        )
        if hashlib.sha256(lock).hexdigest() != LOCK_SHA256:
            raise ValueError("image source lock differs from published R28.1 identity")
        (args.output / "source.lock").write_bytes(lock)
        _, receipt["tokenizers"] = tokenizer_receipt(
            args.reference_tokenizer,
            args.model_cache / "snapshots" / REVISION,
            args.source.read_text(),
        )
        receipt["status"] = "running"
        save()
        for boot in range(1, args.boots + 1):
            require_benchmark_sources()
            folder = args.output / f"boot-{boot}"
            folder.mkdir()
            name = f"slimserve-b12x-r281-{secrets.token_hex(6)}"
            port = free_port()
            if fa2_receipt(args) != receipt["fa2_override"]:
                raise ValueError("FA2 binary or qualification changed during campaign")
            argv = docker_create_args(args, name, port)
            row = {"boot": boot, "status": "starting", "docker_create_argv": argv}
            receipt["runs"].append(row)
            save()
            container_id = None
            process = None
            print(f"B12X boot {boot}/{args.boots}: starting", flush=True)
            try:
                active = command(
                    "nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"
                )
                if active:
                    raise RuntimeError(f"GPUs already have compute processes: {active}")
                container_id = command(*argv)
                row["container_id"] = container_id
                row["container_inspect"] = json.loads(
                    command("docker", "inspect", container_id)
                )[0]
                save()
                with (folder / "server.log").open("wb") as log:
                    process = subprocess.Popen(
                        ["docker", "start", "--attach", container_id],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                    started = time.monotonic()
                    base = f"http://127.0.0.1:{port}"
                    while True:
                        if process.poll() is not None:
                            raise RuntimeError(
                                f"container exited {process.returncode}; "
                                f"see {folder / 'server.log'}"
                            )
                        try:
                            with urllib.request.urlopen(base + "/health", timeout=3):
                                break
                        except OSError as error:
                            if time.monotonic() - started > args.startup_timeout:
                                raise TimeoutError(
                                    "reference startup timed out"
                                ) from error
                            time.sleep(2)
                    row["startup_seconds"] = time.monotonic() - started
                    with urllib.request.urlopen(
                        base + "/v1/models", timeout=10
                    ) as response:
                        row["models"] = json.load(response)
                    if MODEL_NAME not in {
                        model["id"] for model in row["models"]["data"]
                    }:
                        raise ValueError("unexpected served model identity")
                    save()
                    workload = run_workload(
                        argparse.Namespace(
                            url=base,
                            model=MODEL_NAME,
                            source=args.source,
                            tokenizer=args.model_cache / "snapshots" / REVISION,
                            reference_tokenizer=args.reference_tokenizer,
                            output=folder / "workload",
                            repeats=args.repeats,
                            canary_only=getattr(args, "canary_only", False),
                            cold_prefix=getattr(args, "cold_prefix", False),
                            quality=True,
                            prefill=True,
                        )
                    )
                    if getattr(args, "canary_only", False):
                        row["canaries"] = workload["canaries"]
                    else:
                        row["aggregates"] = workload["aggregates"]
                        row["quality"] = workload["quality"]
                        row["prefill"] = workload["prefill"]
                    row["status"] = "complete"
            except BaseException as error:
                row["status"] = "failed"
                row["error"] = repr(error)
                print(f"B12X boot {boot}: {error}", flush=True)
                if not isinstance(error, Exception):
                    raise
            finally:
                if container_id is not None:
                    # Retain exit/OOM evidence before removing our stopped container.
                    row["container_exit_state"] = stop_container(container_id)
                    save()
                    if process is not None:
                        process.wait(timeout=10)
                    command("docker", "rm", container_id)
                    row["owned_container_removed"] = True
                save()
        receipt["status"] = (
            "failed"
            if any(row["status"] != "complete" for row in receipt["runs"])
            else "complete"
        )
    except BaseException as error:
        receipt["status"] = "failed"
        receipt["error"] = repr(error)
        raise
    finally:
        save()
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "output", "model-cache", "reference-tokenizer", "jit-cache"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--boots", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--startup-timeout", type=int, default=1200)
    parser.add_argument(
        "--cold-prefix",
        action="store_true",
        help="use isolated cache salts and require zero cached prompt tokens",
    )
    parser.add_argument(
        "--canary-only",
        action="store_true",
        help="diagnostic only: capture text/image replies without timing workloads",
    )
    parser.add_argument(
        "--diagnostic-nccl",
        action="store_true",
        help="enable NCCL initialization logs; results are diagnostic, not a baseline",
    )
    parser.add_argument(
        "--host-cuda-driver",
        action="store_true",
        help="skip the image shell compatibility-driver hook; preserve host libcuda",
    )
    parser.add_argument(
        "--fa2-library",
        type=Path,
        help="explicit native SM120 FA2 compatibility binary",
    )
    parser.add_argument(
        "--fa2-qualification",
        type=Path,
        help="bundle with original-compat and native-host probe receipts",
    )

    # A controller interruption must run the owned-container finally block.
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    return int(run(parser.parse_args())["status"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())
