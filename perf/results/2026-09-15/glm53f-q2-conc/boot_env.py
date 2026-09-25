"""Boot glm53f-q2-1 --spec detached, with extra env overrides.

Same contract as perf/results/2026-09-14/boot_detached.py (detached session:
the harness kills background task trees once the model pins ~100 GiB), plus
KEY=VALUE arguments applied to the server's environment, and --flags
passed through to the slimserve CLI (--no-spec). Env overrides let a
diagnostic build (VLLM_QC_PHASE_PROF=1) or a re-sized record
(VLLM_QC_MOE_MM_MIN_TOKENS, max_num_seqs via SLIMSERVE_*) boot without
editing the record.

Usage: boot_env.py <outdir> [KEY=VALUE ...]
"""

import os
import subprocess
import sys

WT = "/Users/seangherardi/Code/slimserve/SlimServe-glm53f"
PY = "/Users/seangherardi/Code/slimserve/SlimServe/.venv/bin/python"

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
env = dict(os.environ)
env["PYTHONPATH"] = WT
env["SLIMSERVE_SERVE_IN_PROGRESS"] = "1"
cli_extra: list[str] = []
for arg in sys.argv[2:]:
    if arg.startswith("--"):
        # CLI passthrough, e.g. --no-spec for the non-speculative arm.
        cli_extra.append(arg)
        print("cli", arg)
        continue
    key, _, value = arg.partition("=")
    env[key] = value
    print("env", key, "=", value)

log = open(os.path.join(out, "boot.log"), "ab")
p = subprocess.Popen(
    [PY, "-m", "slimserve.cli", "glm53f-q2-1", "--serve",
     *([] if "--no-spec" in cli_extra else ["--spec"]),
     "--host", "127.0.0.1", "--port", "8000", "-y", *cli_extra],
    cwd=WT, env=env, stdin=subprocess.DEVNULL, stdout=log,
    stderr=subprocess.STDOUT, start_new_session=True,
)
print("launched pid", p.pid, "sid-detached; log", os.path.join(out, "boot.log"))
