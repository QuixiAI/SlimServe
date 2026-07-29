# SPDX-License-Identifier: Apache-2.0

import warnings
import glob
import itertools
import os
from collections.abc import Generator
from pathlib import Path

import gguf
import numpy as np
import torch
from huggingface_hub import snapshot_download
from vllm.logger import init_logger
from vllm.transformers_utils.gguf_utils import gguf_reader

logger = init_logger(__name__)


def download_gguf(
    repo_id: str,
    quant_type: str,
    cache_dir: str | None = None,
    revision: str | None = None,
    ignore_patterns: str | list[str] | None = None,
) -> str:
    prefix_list = ["*.", "*-"]
    suffix_list = ["-*", ""]
    allow_patterns = [
        f"{prefix}{qt}{suffix}.gguf"
        for qt in (quant_type.upper(), quant_type.lower())
        for prefix, suffix in itertools.product(prefix_list, suffix_list)
    ]

    folder = snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        allow_patterns=allow_patterns,
        revision=revision,
        ignore_patterns=ignore_patterns,
    )

    local_files: list[str] = []
    for pattern in allow_patterns:
        local_files.extend(glob.glob(os.path.join(folder, pattern)))

    if not local_files:
        raise ValueError(
            f"Downloaded GGUF files not found in {folder} for quant_type {quant_type}"
        )

    local_files.sort(key=lambda x: (x.count("-"), x))
    return local_files[0]




def get_gguf_extra_tensor_names(
    gguf_file: str | Path, gguf_to_hf_name_map: dict[str, str]
) -> list[str]:
    reader = gguf_reader(gguf_file)
    expected_gguf_keys = set(gguf_to_hf_name_map.keys())
    exact_gguf_keys = {tensor.name for tensor in reader.tensors}
    extra_keys = expected_gguf_keys - exact_gguf_keys
    return [gguf_to_hf_name_map[key] for key in extra_keys]


def get_gguf_weight_type_map(
    gguf_file: str | Path, gguf_to_hf_name_map: dict[str, str]
) -> dict[str, str]:
    reader = gguf_reader(gguf_file)
    return {
        gguf_to_hf_name_map[tensor.name]: tensor.tensor_type.name
        for tensor in reader.tensors
        if tensor.name in gguf_to_hf_name_map
    }


def gguf_quant_weights_iterator(
    gguf_file: str | Path, gguf_to_hf_name_map: dict[str, str] | None
) -> Generator[tuple[str, torch.Tensor], None, None]:
    yield from gguf_quant_weights_iterator_multi([gguf_file], gguf_to_hf_name_map)


def _as_tensor(arr: np.ndarray) -> torch.Tensor:
    """Wrap a GGUF mmap view without copying.

    ``torch.tensor(arr)`` copies; for a 262 GB checkpoint that is a
    single-threaded memcpy of every tensor before it is even moved to the
    device (measured: 271 s -> 102 s when removed). ``from_numpy`` aliases the
    mapping instead, so the H2D transfer reads straight out of page cache. The
    mapping outlives loading and the weights land on the GPU, so aliasing is
    safe here.

    Staging through a reused pinned buffer was tried and is *slower*
    (102 s -> 117 s): pinned H2D is ~9x faster than pageable on MI300X, but the
    extra host memcpy into the pinned buffer costs more than that saves.
    """
    with warnings.catch_warnings():
        # The mapping is read-only; we only ever read from it or copy it to the
        # device, so the non-writable warning is noise here.
        warnings.filterwarnings("ignore", message=".*non-writable.*")
        warnings.filterwarnings("ignore", message=".*not writable.*")
        return torch.from_numpy(arr)


def gguf_quant_weights_iterator_multi(
    gguf_files: list[str], gguf_to_hf_name_map: dict[str, str] | None = None
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Yield ``(name, tensor)`` for all tensors in *gguf_files*.

    When *gguf_to_hf_name_map* is ``None``, raw GGUF tensor names are used
    directly (useful when a caller will apply a :class:`WeightsMapper`
    afterwards).  When a mapping is provided, tensors not present in the map
    are skipped and names are translated accordingly.
    """
    _QUANT_TYPES = ("F32", "BF16", "F16")

    for gguf_file in gguf_files:
        reader = gguf_reader(gguf_file)
        for tensor in reader.tensors:
            if gguf_to_hf_name_map is not None:
                if tensor.name not in gguf_to_hf_name_map:
                    continue
                name = gguf_to_hf_name_map[tensor.name]
            else:
                name = tensor.name

            weight_type = tensor.tensor_type
            if weight_type.name not in _QUANT_TYPES:
                yield name.replace("weight", "qweight_type"), torch.tensor(weight_type)
                name = name.replace("weight", "qweight")

            weight = tensor.data
            if weight_type.name == "BF16" and weight.dtype == np.uint8:
                weight = weight.view(np.uint16)
                if reader.byte_order == "S":
                    weight = weight.byteswap()
                param = _as_tensor(weight).view(torch.bfloat16)
            else:
                param = _as_tensor(weight)
            yield name, param


    # for gguf_file in gguf_files:
    #     reader = gguf_reader(gguf_file)
    #     for tensor in reader.tensors:
    #         if tensor.tensor_type.name in unquant_types:
    #             yield tensor.name.rsplit(".", 1)[0]
