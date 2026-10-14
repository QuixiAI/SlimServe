# SPDX-License-Identifier: Apache-2.0
"""Machine-specific locations for the parity harness, from the environment.

    LTX25_UPSTREAM     the `packages/` directory of a Lightricks/LTX-2 checkout
                       (default ~/.local/scratch/ltx25/upstream/packages)
    LTX25_SCRATCH      captured inputs, import stubs and reference kernels
                       (default ~/.local/scratch/ltx25)
    LTX25_MODEL_ROOT   the official LTX-2.5 weights, as SlimServe fetches them
                       (default $SLIMSERVE_CACHE/ltx-2.5/official, else
                       ~/models/ltx-2.5/official)

REPO is this checkout, found from this file.
"""

import os
from pathlib import Path


def _dir(var: str, default: str) -> str:
    return str(Path(os.environ.get(var) or default).expanduser())


SCRATCH = _dir("LTX25_SCRATCH", "~/.local/scratch/ltx25")
UP = _dir("LTX25_UPSTREAM", f"{SCRATCH}/upstream/packages")
MODELS = _dir(
    "LTX25_MODEL_ROOT",
    f"{os.environ.get('SLIMSERVE_CACHE') or '~/models'}/ltx-2.5/official",
)
REPO = str(Path(__file__).resolve().parents[2])
