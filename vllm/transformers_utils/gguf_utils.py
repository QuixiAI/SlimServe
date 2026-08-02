# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GGUF utility functions."""

import hashlib
import json
import mmap
import os
import threading
import time
from collections import OrderedDict
from functools import cache
from os import PathLike
from pathlib import Path
from typing import Any

import gguf
import gguf.gguf_reader
import numpy as np
from gguf.constants import GGMLQuantizationType, Keys, VisionProjectorType
from gguf.quants import quant_shape_to_byte_shape
from transformers import Gemma3Config, PretrainedConfig, SiglipVisionConfig

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.utils.bootstamp import bootstamp

logger = init_logger(__name__)

_reader_lock = threading.Lock()
_GGUF_METADATA_CACHE_VERSION = 1


class _CachedField:
    def __init__(self, value: Any):
        self._value = value

    def contents(self, index_or_slice: int | slice = slice(None)) -> Any:  # noqa: B008
        if isinstance(index_or_slice, slice) and index_or_slice == slice(None):
            return self._value
        return self._value[index_or_slice]


class _CachedGGUFReader:
    def __init__(self, path: str, metadata: dict[str, Any]):
        self.data = _plain_mmap(path)
        self.byte_order = metadata["byte_order"]
        self.alignment = metadata["alignment"]
        self.data_offset = metadata["data_offset"]
        self.fields = OrderedDict(
            (name, _CachedField(value)) for name, value in metadata["fields"].items()
        )
        self.tensors = [
            self._build_tensor(tensor_metadata)
            for tensor_metadata in metadata["tensors"]
        ]

    def get_field(self, key: str):
        return self.fields.get(key)

    def get_tensor(self, idx: int):
        return self.tensors[idx]

    def _build_tensor(self, metadata: list[Any]):
        name, raw_type, raw_shape, n_elements, n_bytes, data_offset = metadata
        tensor_type = GGMLQuantizationType(raw_type)
        shape = np.asarray(raw_shape, dtype=np.uint64)
        np_shape = tuple(reversed(raw_shape))

        if tensor_type == GGMLQuantizationType.F16:
            item_count, item_type = n_elements, np.float16
        elif tensor_type == GGMLQuantizationType.F32:
            item_count, item_type = n_elements, np.float32
        elif tensor_type == GGMLQuantizationType.F64:
            item_count, item_type = n_elements, np.float64
        elif tensor_type == GGMLQuantizationType.I8:
            item_count, item_type = n_elements, np.int8
        elif tensor_type == GGMLQuantizationType.I16:
            item_count, item_type = n_elements, np.int16
        elif tensor_type == GGMLQuantizationType.I32:
            item_count, item_type = n_elements, np.int32
        elif tensor_type == GGMLQuantizationType.I64:
            item_count, item_type = n_elements, np.int64
        else:
            item_count, item_type = n_bytes, np.uint8
            np_shape = quant_shape_to_byte_shape(np_shape, tensor_type)

        item_size = np.dtype(item_type).itemsize
        data = self.data[data_offset : data_offset + item_count * item_size]
        data = data.view(item_type)[:item_count].reshape(np_shape)
        return gguf.gguf_reader.ReaderTensor(
            name=name,
            tensor_type=tensor_type,
            shape=shape,
            n_elements=n_elements,
            n_bytes=n_bytes,
            data_offset=data_offset,
            data=data,
            field=None,
        )


def _metadata_cache_path(path: str) -> Path:
    key = hashlib.sha256(path.encode()).hexdigest()
    return Path(envs.VLLM_CACHE_ROOT) / "gguf_metadata" / f"{key}.json"


def _load_cached_reader(path: str) -> _CachedGGUFReader | None:
    cache_path = _metadata_cache_path(path)
    try:
        with cache_path.open() as cache_file:
            metadata = json.load(cache_file)
        stat = os.stat(path)
        if (
            metadata["version"] != _GGUF_METADATA_CACHE_VERSION
            or metadata["size"] != stat.st_size
            or metadata["mtime_ns"] != stat.st_mtime_ns
        ):
            return None
        return _CachedGGUFReader(path, metadata)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _save_reader_cache(path: str, reader: gguf.GGUFReader) -> None:
    cache_path = _metadata_cache_path(path)
    temp_path = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
    stat = os.stat(path)
    metadata = {
        "version": _GGUF_METADATA_CACHE_VERSION,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "byte_order": reader.byte_order,
        "alignment": int(reader.alignment),
        "data_offset": int(reader.data_offset),
        "fields": {name: field.contents() for name, field in reader.fields.items()},
        "tensors": [
            [
                tensor.name,
                int(tensor.tensor_type),
                [int(dim) for dim in tensor.shape],
                int(tensor.n_elements),
                int(tensor.n_bytes),
                int(tensor.data_offset),
            ]
            for tensor in reader.tensors
        ],
    }
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with temp_path.open("w") as cache_file:
            json.dump(metadata, cache_file, separators=(",", ":"))
        os.replace(temp_path, cache_path)
    except (OSError, TypeError, ValueError):
        temp_path.unlink(missing_ok=True)


def _plain_mmap(path, mode="r"):
    """Map `path` as a plain ndarray rather than an `np.memmap`.

    `np.memmap` is an ndarray subclass, so every slice of it runs
    `__array_finalize__` and a `may_share_memory` check. The reader takes one
    slice per metadata string, and this model's shard 1 carries a 154,880-entry
    vocab plus 321,649 merges, so that is ~477k slices and ~4.4M finalize calls.
    A plain ndarray over the same mapping parses byte-identically in a third of
    the time.
    """
    f = open(path, "rb")  # noqa: SIM115 -- the mmap must outlive this call
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    return np.frombuffer(mm, dtype=np.uint8)


@cache
def gguf_reader(path: str | PathLike) -> gguf.GGUFReader | _CachedGGUFReader:
    """A `GGUFReader`, parsed the fast way and cached per process.

    Booting opens shard 1 dozens of times across the API server, the engine
    core and both TP workers; at 8.6 s a parse that dominated startup. Caching
    is safe because readers are opened read-only and never mutated.
    """
    path = str(Path(path).resolve())
    start = time.perf_counter()
    if cached_reader := _load_cached_reader(path):
        bootstamp(
            f"gguf_reader restored {Path(path).name} metadata "
            f"in {time.perf_counter() - start:.2f}s"
        )
        return cached_reader

    real = gguf.gguf_reader.np.memmap
    with _reader_lock:
        gguf.gguf_reader.np.memmap = _plain_mmap
        try:
            reader = gguf.GGUFReader(path)
        finally:
            gguf.gguf_reader.np.memmap = real
            bootstamp(
                f"gguf_reader parsed {Path(path).name} "
                f"in {time.perf_counter() - start:.2f}s"
            )
    if time.perf_counter() - start >= 0.5:
        _save_reader_cache(path, reader)
    return reader


@cache
def gguf_architecture(path: str | PathLike) -> str:
    """The `general.architecture` string, e.g. "glm-dsa" or "deepseek4".

    This is what the config parser, the tokenizer registry and the weights
    adapter all dispatch on, so it is read once here rather than three times
    with three spellings of the same lookup.
    """
    field = gguf_reader(str(path)).fields.get("general.architecture")
    return "" if field is None else str(field.contents())


@cache
def check_gguf_file(model: str | PathLike) -> bool:
    """Check if the file is a GGUF model."""
    model = Path(model)
    if not model.is_file():
        return False
    elif model.suffix == ".gguf":
        return True

    try:
        with model.open("rb") as f:
            header = f.read(4)

        return header == b"GGUF"
    except Exception as e:
        logger.debug("Error reading file %s: %s", model, e)
        return False


@cache
def is_gguf(model: str | Path) -> bool:
    """Check if the model is a GGUF model.

    Args:
        model: Model name, path, or Path object to check.

    Returns:
        True if the model is a GGUF model, False otherwise.
    """
    return check_gguf_file(str(model))


def detect_gguf_multimodal(model: str) -> Path | None:
    """Check if GGUF model has multimodal projector file.

    Args:
        model: Model path string

    Returns:
        Path to mmproj file if found, None otherwise
    """
    if not model.endswith(".gguf"):
        return None

    try:
        model_path = Path(model)
        if not model_path.is_file():
            return None

        model_dir = model_path.parent
        mmproj_patterns = ["mmproj.gguf", "mmproj-*.gguf", "*mmproj*.gguf"]
        for pattern in mmproj_patterns:
            mmproj_files = list(model_dir.glob(pattern))
            if mmproj_files:
                return mmproj_files[0]
        return None
    except Exception:
        return None


def extract_vision_config_from_gguf(mmproj_path: str) -> "SiglipVisionConfig | None":
    """Extract vision config parameters from mmproj.gguf metadata.

    Reads vision encoder configuration from GGUF metadata fields using
    standardized GGUF constants. Automatically detects the projector type
    (e.g., gemma3, llama4) and applies model-specific parameters accordingly.

    The function extracts standard CLIP vision parameters from GGUF metadata
    and applies projector-type-specific customizations. For unknown projector
    types, it uses safe defaults from SiglipVisionConfig.

    Args:
        mmproj_path: Path to mmproj.gguf file (str or Path)

    Returns:
        SiglipVisionConfig if extraction succeeds, None if any required
        field is missing from the GGUF metadata

    Raises:
        Exception: Exceptions from GGUF reading (file not found, corrupted
            file, etc.) propagate directly from gguf.GGUFReader
    """
    reader = gguf_reader(mmproj_path)

    # Detect projector type to apply model-specific parameters
    projector_type = None
    projector_type_field = reader.get_field(Keys.Clip.PROJECTOR_TYPE)
    if projector_type_field:
        try:
            projector_type = bytes(projector_type_field.parts[-1]).decode("utf-8")
        except (AttributeError, UnicodeDecodeError) as e:
            logger.warning("Failed to decode projector type from GGUF: %s", e)

    # Map GGUF field constants to SiglipVisionConfig parameters.
    # Uses official GGUF constants from gguf-py for standardization.
    # Format: {gguf_constant: (param_name, dtype)}
    VISION_CONFIG_FIELDS = {
        Keys.ClipVision.EMBEDDING_LENGTH: ("hidden_size", int),
        Keys.ClipVision.FEED_FORWARD_LENGTH: ("intermediate_size", int),
        Keys.ClipVision.BLOCK_COUNT: ("num_hidden_layers", int),
        Keys.ClipVision.Attention.HEAD_COUNT: ("num_attention_heads", int),
        Keys.ClipVision.IMAGE_SIZE: ("image_size", int),
        Keys.ClipVision.PATCH_SIZE: ("patch_size", int),
        Keys.ClipVision.Attention.LAYERNORM_EPS: ("layer_norm_eps", float),
    }

    # Extract and validate all required fields
    config_params = {}
    for gguf_key, (param_name, dtype) in VISION_CONFIG_FIELDS.items():
        field = reader.get_field(gguf_key)
        if field is None:
            logger.warning(
                "Missing required vision config field '%s' in mmproj.gguf",
                gguf_key,
            )
            return None
        # Extract scalar value from GGUF field and convert to target type
        config_params[param_name] = dtype(field.parts[-1])

    # Apply model-specific parameters based on projector type
    if projector_type == VisionProjectorType.GEMMA3:
        # Gemma3 doesn't use the vision pooling head (multihead attention)
        # This is a vLLM-specific parameter used in SiglipVisionTransformer
        config_params["vision_use_head"] = False
        logger.info("Detected Gemma3 projector, disabling vision pooling head")
    # Add other projector-type-specific customizations here as needed
    # elif projector_type == VisionProjectorType.LLAMA4:
    #     config_params["vision_use_head"] = ...

    # Create config with extracted parameters
    # Note: num_channels and attention_dropout use SiglipVisionConfig defaults
    # (3 and 0.0 respectively) which are correct for all models
    config = SiglipVisionConfig(**config_params)

    if projector_type:
        logger.info(
            "Extracted vision config from mmproj.gguf (projector_type: %s)",
            projector_type,
        )
    else:
        logger.info("Extracted vision config from mmproj.gguf metadata")

    return config


def extract_vocab_size_from_gguf(model_path: str | Path) -> int | None:
    """Extract vocab size from a GGUF backbone file."""
    if not check_gguf_file(model_path):
        return None

    reader = gguf_reader(model_path)
    field = reader.get_field(Keys.Tokenizer.LIST)
    if field is None:
        logger.warning("Missing tokenizer token list in GGUF file: %s", model_path)
        return None
    return len(field.contents())


def extract_lm_head_from_gguf(model_path: str | Path) -> bool:
    """Check if GGUF file contains LM head weights based on tensor names."""
    if not check_gguf_file(model_path):
        return None

    reader = gguf_reader(model_path)
    return any(tensor.name == "output.weight" for tensor in reader.tensors)


def maybe_patch_hf_config_from_gguf(
    model: str,
    hf_config: PretrainedConfig,
) -> PretrainedConfig:
    """Patch HF config for GGUF models.

    Applies GGUF-specific patches to HuggingFace config:
    1. For multimodal models: patches architecture and vision config
    2. For all GGUF models: overrides vocab_size from embedding tensor

    This ensures compatibility with GGUF models that have extended
    vocabularies (e.g., Unsloth) where the GGUF file contains more
    tokens than the HuggingFace tokenizer config specifies.

    Args:
        model: Model path string
        hf_config: HuggingFace config to patch in-place

    Returns:
        Updated HuggingFace config
    """
    vocab_size = extract_vocab_size_from_gguf(model)
    if vocab_size is not None:
        if hasattr(hf_config, "vocab_size"):
            # Composite configs (Glm5vConfig) expose vocab_size as a read-only
            # property delegating to text_config, which is patched below
            # anyway, so a failure here is expected and harmless.
            try:  # noqa: SIM105 -- the comment above is the point
                hf_config.update({"vocab_size": vocab_size})
            except AttributeError:
                pass
        text_config = hf_config.get_text_config()
        if hasattr(text_config, "vocab_size"):
            text_config.update({"vocab_size": vocab_size})

    has_lm_head = extract_lm_head_from_gguf(model)
    if has_lm_head is not None:
        text_config = hf_config.get_text_config()
        text_config.update({"tie_word_embeddings": not has_lm_head})

    # Patch multimodal config if mmproj.gguf exists
    mmproj_path = detect_gguf_multimodal(model)
    if mmproj_path is not None:
        vision_config = extract_vision_config_from_gguf(str(mmproj_path))

        # Create HF config for Gemma3 multimodal
        text_config = hf_config.get_text_config()
        is_gemma3 = hf_config.model_type in ("gemma3", "gemma3_text")
        if vision_config is not None and is_gemma3:
            new_hf_config = Gemma3Config(
                text_config=text_config,
                vision_config=vision_config,
                architectures=["Gemma3ForConditionalGeneration"],
            )
            if vocab_size is not None:
                new_hf_config.text_config.update({"vocab_size": vocab_size})
            hf_config = new_hf_config

    return hf_config
