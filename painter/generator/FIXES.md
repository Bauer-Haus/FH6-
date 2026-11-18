# Changes in this copy of the generator

Source: [zjl88858/forza-painter-geometrize-gpu](https://github.com/zjl88858/forza-painter-geometrize-gpu)
at commit `894a267` (2026-06-23, "v1.2-Canary-20260613"), MIT license (`LICENSE`).
Its prebuilt `dist/*.exe` files were left out; build from source instead.

## About `painter/bin/forza-painter-geometrize-go.exe`

The app ships a prebuilt copy of this generator. `go version -m` on it shows:

- it is byte-identical (SHA-256 `fe9ca149…`) to the generator author's own
  published `dist/forza-painter-geometrize-go-canary.exe`, so it was not altered
  by the app's maintainer;
- it was built from commit `2b4c350` **with uncommitted changes**
  (`vcs.modified=true`). That commit is not in the public history (it predates the
  first public commit), so the exact source of that EXE cannot be checked.

## Bugs fixed

Every fix has a test that fails on the original code and passes here
(`go test ./...`, run against PoCL OpenCL and Mesa lavapipe Vulkan on CPU).

| # | Where | Bug | Effect |
|---|-------|-----|--------|
| 1 | `cmd/.../main.go` | `flag.Args()[1:]` with no arguments | Crash (`slice bounds out of range`) instead of usage text. |
| 2 | `engine.pruneOccludedShapes` | Occlusion check scanned only `cx±rx, cy±ry`, the box of an **unrotated** shape | Visible rotated shapes could be deleted as "hidden" during the every-500-layers and final recycling passes. |
| 3 | `kernels.go`, `opencl.go`, `vulkan.go`, `eval_v3/v4.comp` | Rectangles and triangles used the ellipse bounding box | Corners of rotated rectangles/triangles were never drawn or scored (only with `enableMultiPrimitiveShapes`). |
| 4 | `kernels.go` `evaluate_candidates_ssim` | Read 6 floats per candidate; host packs 7 | With `errorMetric = ssim`, every candidate after the first was scored with another candidate's data. |
| 5 | `shaders/eval_v3.comp` (Vulkan) | Same 6-vs-7 stride | With `-backend vulkan` and `useWorkGroupEval = false`, selection was effectively random: test image error 55.8 → 25.1 after the fix. |
| 6 | `opencl.go` `SubmitEval`; engine | SSIM branch unreachable with the default `useWorkGroupEval = true`; SSIM map not built before the first shape | `errorMetric = ssim` printed "SSIM" but scored with MSE; the first SSIM evaluation read an uninitialised buffer. Vulkan has no SSIM kernels; it now warns. |
| 7 | `imageutil.LoadAndPrepare` | Go's `RGBA()` is alpha-premultiplied but was used as straight colour | Dark fringes around transparent PNGs (edge brightness 90 vs true 104; 105 after the fix). |
| 8 | engine hill climb | Compared raw scores while the random pick used score + large-shape penalty | The climb could grow shapes past the penalty the random stage enforces. |

Normal output is unchanged: with default settings on an opaque image the result
is identical before and after (same RMSE with the same seed).

## Building

Windows: `build.ps1` as upstream (needs OpenCL-SDK and Vulkan SDK). Shaders: `build-shaders.ps1`.

Linux (used for testing here):

```sh
sudo apt install ocl-icd-opencl-dev opencl-headers libvulkan-dev glslc   # + pocl-opencl-icd / mesa-vulkan-drivers for CPU testing
export CGO_CFLAGS="-DCL_TARGET_OPENCL_VERSION=120 -DCL_DEPTH_STENCIL=0x10BE -DCL_UNORM_INT24=0x10DF"
export CGO_LDFLAGS="-lOpenCL -lvulkan"
for s in eval_v3 eval_v4 apply error_grid; do glslc -O --target-env=vulkan1.2 -fshader-stage=compute shaders/$s.comp -o shaders/$s.comp.spv; done
go test ./...
go build -o forza-painter-geometrize-go ./cmd/forza-painter-geometrize
```

## Known issues not changed

- The Vulkan backend has no SSIM support.
- After `-resume`, the reported normalised `score` values start from the blank canvas error (cosmetic).
