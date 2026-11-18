#!/bin/bash
# Build the q38 engine library and lab tool. No dependencies beyond Xcode's
# toolchain and the QuixiCore launcher header in this repo.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
out="$here/build"
mkdir -p "$out"
flags=(-std=c++20 -O2 -fobjc-arc -Wall -Wno-unused-function -DACCELERATE_NEW_LAPACK
       -I "$here/include"
       -I "$repo/csrc/quixicore/metal/kernels/common"
       -I "$repo/csrc/quixicore/metal/include")
xcrun clang++ "${flags[@]}" -c "$here/src/engine.mm" -o "$out/engine.o"
xcrun clang++ "${flags[@]}" -c "$here/src/generate.cpp" -o "$out/generate.o"
xcrun clang++ "${flags[@]}" -c "$here/src/pool.cpp" -o "$out/pool.o"
xcrun clang++ "${flags[@]}" "$here/tools/q38_lab.mm" "$out/engine.o" "$out/generate.o" "$out/pool.o" \
  -framework Metal -framework Foundation -framework Accelerate -o "$out/q38_lab"
xcrun clang++ -dynamiclib "$out/engine.o" "$out/generate.o" "$out/pool.o" \
  -framework Metal -framework Foundation -framework Accelerate -install_name @rpath/libq38.dylib -o "$out/libq38.dylib"
xcrun metal -std=metal3.1 -O2 "$here"/kernels/*.metal -o "$out/q38_native.metallib"
echo "built $out/q38_lab"
