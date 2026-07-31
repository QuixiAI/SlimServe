#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#
# A command line tool for running pytorch's hipify preprocessor on CUDA
# source files.
#
# See https://github.com/ROCm/hipify_torch
# and <torch install dir>/utils/hipify/hipify_python.py
#

import argparse
import os
import shutil

from torch.utils.hipify.hipify_python import get_hip_file_path, hipify

HEADER_EXTENSIONS = (".h", ".hpp", ".cuh", ".hip.h", ".inl")


def _expected_hip_build_path(source_abs: str, output_directory: str) -> str:
    """Match torch.utils.hipify.hipify_python.preprocessor fout_path naming."""
    rel = os.path.relpath(source_abs, output_directory)
    return os.path.abspath(
        os.path.join(
            output_directory, get_hip_file_path(rel, is_pytorch_extension=True)
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Project directory where all the source + include files live.
    parser.add_argument(
        "-p",
        "--project_dir",
        help="The project directory.",
    )

    # Directory where hipified files are written.
    parser.add_argument(
        "-o",
        "--output_dir",
        help="The output directory.",
    )

    # Source files to convert.
    parser.add_argument(
        "sources", help="Source files to hipify.", nargs="*", default=[]
    )

    args = parser.parse_args()

    # Limit include scope to the project sources and their copies in the build
    # tree. The output dir must be listed too: hipify walks it (not project_dir)
    # to decide which headers are eligible, so with the project pattern alone
    # `all_files` holds no headers at all and every one is skipped as
    # "[ignored, not to be hipified]" -- leaving raw `cuda_runtime.h` includes
    # and `cudaDeviceProp` in headers the HIP compiler then chokes on.
    includes = [
        os.path.join(args.project_dir, "*"),
        os.path.join(args.output_dir, "*"),
    ]

    # Get absolute path for all source files.
    extra_files = [os.path.abspath(s) for s in args.sources]

    # Snapshot the previous conversion before the copytree below overwrites it.
    # hipify_all re-runs on every build; without this an unchanged rebuild would
    # rewrite every header, and a fresh mtime on a header half the tree includes
    # means recompiling half the tree.
    previous = {}
    for dirpath, _, names in os.walk(args.output_dir):
        for name in names:
            if not name.endswith(HEADER_EXTENSIONS):
                continue
            path = os.path.join(dirpath, name)
            with open(path, encoding="utf-8", errors="surrogateescape") as f:
                previous[path] = (f.read(), os.stat(path).st_mtime_ns)

    # Copy sources from project directory to output directory.
    # The directory might already exist to hold object files so we ignore that.
    shutil.copytree(args.project_dir, args.output_dir, dirs_exist_ok=True)

    # Hipify every header explicitly rather than relying on the sources to pull
    # them in. Sources whose `.hip` output is already current are skipped before
    # their includes are ever walked, but the copytree above has just replaced
    # the build tree's headers with the raw CUDA originals -- so on a rebuild
    # they would stay raw and the HIP compiler would fail on `cuda_runtime.h`
    # and `cudaDeviceProp`.
    # Enumerate from the project tree, not the output tree: hipify writes its
    # result to a renamed twin (foo.h -> foo_hip.h) that lands in the output dir,
    # and feeding those twins back in on the next run hipifies them again into
    # foo_hip_hip.h, leaving the real content stranded behind a chain of stubs.
    # Only files that exist in the source tree are ever inputs.
    headers = [
        os.path.join(
            args.output_dir,
            os.path.relpath(os.path.join(dirpath, name), args.project_dir),
        )
        for dirpath, _, names in os.walk(args.project_dir)
        for name in names
        if name.endswith(HEADER_EXTENSIONS)
    ]
    extra_files.extend(headers)

    hipify_result = hipify(
        project_directory=args.project_dir,
        output_directory=args.output_dir,
        # Hipify resolves quoted includes next to the including file first; vLLM
        # uses paths relative to csrc/ (e.g. "libtorch_stable/torch_utils.h"
        # from quantization/w8a8/fp8/*.cu). Without an include root here, those
        # headers are never found and are not hipified or rewritten in dependents.
        header_include_dirs=["."],
        includes=includes,
        extra_files=extra_files,
        show_detailed=True,
        is_pytorch_extension=True,
        hipify_extra_files_only=True,
    )

    # Land each converted header back under its original name. hipify renames
    # its output (torch_utils.h -> torch_utils_hip.h, cuda_compat.h ->
    # hip_compat.h) and rewrites `#include`s in dependents to match -- but that
    # rewrite only fires for include text that some hipified path ends with, so
    # vLLM's relative form ("../../torch_utils.h") never matches and dependents
    # keep including the original name. Overwriting in place is what makes those
    # includes resolve to HIP content; the compile only ever sees the build tree.
    def _write_if_changed(path: str, text: str) -> None:
        """Write `text`, keeping the old mtime when the conversion is unchanged.

        The mtime is what ninja compares, and it has to be restored rather than
        merely left alone: both the copytree and hipify itself have already
        rewritten these paths this run.
        """
        old_text, old_mtime_ns = previous.get(path, (None, None))
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        if old_text == text:
            os.utime(path, ns=(old_mtime_ns, old_mtime_ns))

    for header in headers:
        result = hipify_result.get(header)
        hipified = getattr(result, "hipified_path", None)
        if not hipified or os.path.normpath(hipified) == os.path.normpath(header):
            continue
        with open(hipified, encoding="utf-8") as f:
            converted = f.read()
        # Leave a forwarder at the renamed path. Some dependents *did* get their
        # include rewritten to it, and two real copies of a header full of
        # `inline` definitions is a redefinition error -- `#pragma once` is
        # per-file, so it cannot dedupe two paths holding the same text.
        rel = os.path.relpath(header, os.path.dirname(hipified))
        _write_if_changed(header, converted)
        _write_if_changed(hipified, f'#pragma once\n#include "{rel}"\n')

    hipified_sources = []
    for source in args.sources:
        s_abs = os.path.abspath(source)
        if s_abs in hipify_result and hipify_result[s_abs].hipified_path is not None:
            path = hipify_result[s_abs].hipified_path
            # PyTorch skips writing when is_pytorch_extension and text unchanged;
            # hipified_path then stays *.cu. CMake expects *.hip under output_dir.
            if s_abs.endswith(".cu") and path.endswith(".cu"):
                dest = _expected_hip_build_path(s_abs, args.output_dir)
                if os.path.normpath(path) != os.path.normpath(dest):
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    shutil.copy2(path, dest)
                hipified_s_abs = dest
            else:
                hipified_s_abs = path
        else:
            hipified_s_abs = s_abs
        hipified_sources.append(hipified_s_abs)

    assert len(hipified_sources) == len(args.sources)

    # Print hipified source files.
    print("\n".join(hipified_sources))
