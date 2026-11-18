#!/usr/bin/env bash
# Cross-compile the generator for Windows from Linux (Debian/Ubuntu).
# Output: ../../bin/forza-painter-geometrize-go-fixed.exe
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

# Needs: sudo apt install golang gcc-mingw-w64-x86-64 opencl-headers libvulkan-dev
mkdir -p "$work/include"
cp -r /usr/include/CL /usr/include/vulkan "$work/include/"
[ -d /usr/include/vk_video ] && cp -r /usr/include/vk_video "$work/include/"
# Import libraries for the driver DLLs, from the list of functions the generator calls.
x86_64-w64-mingw32-dlltool -d "$here/OpenCL.def" -l "$work/libOpenCL.a"
x86_64-w64-mingw32-dlltool -d "$here/vulkan-1.def" -l "$work/libvulkan-1.a"

cd "$here/.."
CGO_ENABLED=1 GOOS=windows GOARCH=amd64 CC=x86_64-w64-mingw32-gcc \
CGO_CFLAGS="-I$work/include -DCL_TARGET_OPENCL_VERSION=120 -DCL_DEPTH_STENCIL=0x10BE -DCL_UNORM_INT24=0x10DF" \
CGO_LDFLAGS="-L$work -lOpenCL -lvulkan-1" \
go build -trimpath -o ../bin/forza-painter-geometrize-go-fixed.exe ./cmd/forza-painter-geometrize
echo "Built ../bin/forza-painter-geometrize-go-fixed.exe"
