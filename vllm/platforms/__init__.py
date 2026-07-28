# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging
import os
import traceback
from itertools import chain
from typing import TYPE_CHECKING

from vllm import envs
from vllm.plugins import PLATFORM_PLUGINS_GROUP, load_plugins_by_group
from vllm.utils.import_utils import resolve_obj_by_qualname

from .interface import CpuArchEnum, Platform, PlatformEnum

logger = logging.getLogger(__name__)


def vllm_version_matches_substr(substr: str) -> bool:
    """
    Check to see if the vLLM version matches a substring.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        vllm_version = version("vllm")
    except PackageNotFoundError as e:
        logger.warning(
            "The vLLM package was not found, so its version could not be "
            "inspected. This may cause platform detection to fail."
        )
        raise e
    return substr in vllm_version




def rocm_platform_plugin() -> str | None:
    is_rocm = False
    logger.debug("Checking if ROCm platform is available.")
    try:
        import amdsmi

        amdsmi.amdsmi_init()
        try:
            if len(amdsmi.amdsmi_get_processor_handles()) > 0:
                is_rocm = True
                logger.debug("Confirmed ROCm platform is available.")
            else:
                logger.debug("ROCm platform is not available because no GPU is found.")
        finally:
            amdsmi.amdsmi_shut_down()
    except Exception as e:
        logger.debug("ROCm platform is not available because: %s", str(e))

    return "vllm.platforms.rocm.RocmPlatform" if is_rocm else None




builtin_platform_plugins = {
    "rocm": rocm_platform_plugin,
}


def resolve_current_platform_cls_qualname() -> str:
    platform_plugins = load_plugins_by_group(PLATFORM_PLUGINS_GROUP)

    activated_plugins = []

    for name, func in chain(builtin_platform_plugins.items(), platform_plugins.items()):
        try:
            assert callable(func)
            platform_cls_qualname = func()
            if platform_cls_qualname is not None:
                activated_plugins.append(name)
        except Exception:
            pass

    activated_builtin_plugins = list(
        set(activated_plugins) & set(builtin_platform_plugins.keys())
    )
    activated_oot_plugins = list(set(activated_plugins) & set(platform_plugins.keys()))

    if len(activated_oot_plugins) >= 2:
        raise RuntimeError(
            "Only one platform plugin can be activated, but got: "
            f"{activated_oot_plugins}"
        )
    elif len(activated_oot_plugins) == 1:
        platform_cls_qualname = platform_plugins[activated_oot_plugins[0]]()
        logger.info("Platform plugin %s is activated", activated_oot_plugins[0])
    elif len(activated_builtin_plugins) >= 2:
        raise RuntimeError(
            "Only one platform plugin can be activated, but got: "
            f"{activated_builtin_plugins}"
        )
    elif len(activated_builtin_plugins) == 1:
        platform_cls_qualname = builtin_platform_plugins[activated_builtin_plugins[0]]()
        logger.debug(
            "Automatically detected platform %s.", activated_builtin_plugins[0]
        )
    else:
        platform_cls_qualname = "vllm.platforms.interface.UnspecifiedPlatform"
        logger.debug("No platform detected, vLLM is running on UnspecifiedPlatform")
    return platform_cls_qualname


_current_platform = None
_init_trace: str = ""

if TYPE_CHECKING:
    current_platform: Platform


def __getattr__(name: str):
    if name == "current_platform":
        # lazy init current_platform.
        # 1. out-of-tree platform plugins need `from vllm.platforms import
        #    Platform` so that they can inherit `Platform` class. Therefore,
        #    we cannot resolve `current_platform` during the import of
        #    `vllm.platforms`.
        # 2. when users use out-of-tree platform plugins, they might run
        #    `import vllm`, some vllm internal code might access
        #    `current_platform` during the import, and we need to make sure
        #    `current_platform` is only resolved after the plugins are loaded
        #    (we have tests for this, if any developer violate this, they will
        #    see the test failures).
        global _current_platform
        if _current_platform is None:
            platform_cls_qualname = resolve_current_platform_cls_qualname()
            _current_platform = resolve_obj_by_qualname(platform_cls_qualname)()
            global _init_trace
            _init_trace = "".join(traceback.format_stack())
        return _current_platform
    elif name in globals():
        return globals()[name]
    else:
        raise AttributeError(f"No attribute named '{name}' exists in {__name__}.")


def __setattr__(name: str, value):
    if name == "current_platform":
        global _current_platform
        _current_platform = value
    elif name in globals():
        globals()[name] = value
    else:
        raise AttributeError(f"No attribute named '{name}' exists in {__name__}.")


__all__ = [
    "Platform",
    "PlatformEnum",
    "current_platform",
    "CpuArchEnum",
    "_init_trace",
]
