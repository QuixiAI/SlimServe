#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fixed shell-default / host-driver / shell-default CUDA IPC/NCCL control.

Model-free diagnostic using the pinned B12X image and the existing runtime
probe. Change only BASH_ENV=/dev/null to suppress the image's shell startup
hook; preserve P2P, NCCL and CUDA runtime. All failed arms and logs survive.
Docker has its own 16-GiB/no-swap cap; run this controller under a memory cap.
"""

import argparse
import hashlib
import json
import os
import secrets
import signal
import subprocess
from pathlib import Path

from benchmark_glm53_b12x import IMAGE, command, stop_container


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.output.resolve()
    probe = Path(__file__).with_name("probe_cuda_ipc_nccl.py").resolve()
    cache = args.jit_cache.resolve(strict=True)
    receipt = {
        "status": "running",
        "diagnostic_only": True,
        "image": IMAGE,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "probe_sha256": hashlib.sha256(probe.read_bytes()).hexdigest(),
        "arms": [],
    }
    record = root / "summary.json"

    def save():
        record.write_text(json.dumps(receipt, indent=2) + "\n")

    save()
    try:
        for label, host_driver in (
            ("shell-default", False),
            ("shell-host", True),
            ("shell-default-return", False),
        ):
            active = command(
                "nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"
            )
            if active:
                raise RuntimeError(f"GPUs already have compute processes: {active}")
            argv = [
                "docker",
                "create",
                "--name",
                f"slimserve-compat-{secrets.token_hex(6)}",
                "--init",
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "--gpus",
                "all",
                "--memory",
                "16g",
                "--memory-swap",
                "16g",
                "--shm-size",
                "1g",
                "--pids-limit",
                "1024",
                "--network",
                "host",
                "--ulimit",
                "core=0",
                "--ulimit",
                "memlock=-1:-1",
                "--entrypoint",
                "/bin/bash",
                "--mount",
                f"type=bind,src={probe},dst=/probe.py,readonly",
                "--mount",
                f"type=bind,src={root},dst=/probe-out",
                "--mount",
                f"type=bind,src={cache},dst=/cache",
                "-e",
                "NCCL_DEBUG=INFO",
                "-e",
                "NCCL_P2P_DISABLE=0",
                "-e",
                "NCCL_P2P_LEVEL=SYS",
                "-e",
                "OMP_NUM_THREADS=1",
            ]
            if host_driver:
                argv += ["-e", "BASH_ENV=/dev/null"]
            argv += [
                IMAGE,
                "-c",
                (
                    "exec /opt/venv/bin/python -m torch.distributed.run --standalone "
                    "--nproc-per-node=4 /probe.py --timeout-seconds 45 "
                    f"--output /probe-out/{label}"
                ),
            ]
            row = {"arm": label, "status": "starting", "argv": argv}
            receipt["arms"].append(row)
            save()
            identity = None
            try:
                identity = command(*argv)
                row["container_id"] = identity
                row["container_inspect"] = json.loads(
                    command("docker", "inspect", identity)
                )[0]
                save()
                print(f"{label}: starting", flush=True)
                with (root / f"{label}.log").open("wb") as log:
                    process = subprocess.Popen(
                        ["docker", "start", "--attach", identity],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                    try:
                        row["exit_code"] = process.wait(timeout=150)
                    except subprocess.TimeoutExpired:
                        row["error"] = "150-second diagnostic bound exceeded"
                        stop_container(identity)
                        row["exit_code"] = process.wait(timeout=10)
                row["status"] = "complete" if row["exit_code"] == 0 else "failed"
                print(f"{label}: {row['status']}", flush=True)
            except BaseException as error:
                row["status"] = "failed"
                row["error"] = repr(error)
                raise
            finally:
                if identity is not None:
                    row["container_exit_state"] = stop_container(identity)
                    save()
                    command("docker", "rm", identity)
                    row["owned_container_removed"] = True
                save()
        receipt["status"] = "finished"
    except BaseException as error:
        receipt["status"] = "failed"
        receipt["error"] = repr(error)
        raise
    finally:
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jit-cache", type=Path, required=True)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    run(parser.parse_args())
