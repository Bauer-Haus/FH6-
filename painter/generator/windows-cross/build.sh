#!/usr/bin/env bash
# Cross-compile the generator for Windows from Linux (Debian/Ubuntu) or macOS.
# Output: ../../bin/forza-painter-geometrize-go-fixed.exe
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

# Needs: sudo apt install golang gcc-mingw-w64-x86-64 opencl-headers libvulkan-dev
#    or (macOS): brew install go mingw-w64 opencl-headers vulkan-headers
mkdir -p "$work/include"
header_dirs=(/usr/include)
if command -v brew >/dev/null 2>&1; then
  prefix="$(brew --prefix)"
  header_dirs+=("$prefix/opt/opencl-headers/include" "$prefix/opt/vulkan-headers/include" "$prefix/include")
fi
for name in CL vulkan vk_video; do
  for dir in "${header_dirs[@]}"; do
    if [ -d "$dir/$name" ]; then cp -r "$dir/$name" "$work/include/"; break; fi
  done
done
for name in CL vulkan; do
  [ -d "$work/include/$name" ] || { echo "missing $name headers (see Needs above)" >&2; exit 1; }
done
# Import libraries for the driver DLLs, from the list of functions the generator calls.
x86_64-w64-mingw32-dlltool -d "$here/OpenCL.def" -l "$work/libOpenCL.a"
x86_64-w64-mingw32-dlltool -d "$here/vulkan-1.def" -l "$work/libvulkan-1.a"

cd "$here/.."
CGO_ENABLED=1 GOOS=windows GOARCH=amd64 CC=x86_64-w64-mingw32-gcc \
CGO_CFLAGS="-I$work/include -DCL_TARGET_OPENCL_VERSION=120 -DCL_DEPTH_STENCIL=0x10BE -DCL_UNORM_INT24=0x10DF" \
CGO_LDFLAGS="-L$work -lOpenCL -lvulkan-1" \
go build -trimpath -o ../bin/forza-painter-geometrize-go-fixed.exe ./cmd/forza-painter-geometrize
echo "Built ../bin/forza-painter-geometrize-go-fixed.exe"
