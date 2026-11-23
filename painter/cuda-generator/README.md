# forza-painter-geometrize-cuda

A replacement for `painter/bin/forza-painter-geometrize-go.exe` that runs the layer search on an NVIDIA GPU with CUDA.

It is a drop-in replacement. It takes the same command line and reads the same settings files. It writes the same JSON checkpoints, previews and log lines, so the painter app works with it unchanged.

**Status: not yet verified on a GPU.** This branch builds on GitHub, and its CPU tests pass. The GPU code has only been checked by the self-test, which has to run on a real card. Until the checks under *Trying it* pass, it stays off `main`.

## Why it is faster

The Go generator reads every pixel of every candidate shape on the GPU. Each layer also goes back and forth to the CPU several times. This generator differs in four ways:

- **Row running totals.** For every image row, it keeps running totals of what the scoring formula needs: pixel counts, target and canvas sums, canvas² and target·canvas. A shape's statistics then cost two lookups per row it covers, instead of one read per pixel. Each row's run of pixels comes from solving the ellipse equation, and its end pixels are checked with the Go kernel's own per-pixel test, so the pixels counted are exactly the same.
- **One warp (32 threads) per candidate.** The Go generator gives each candidate a block of 256 threads.
- **The whole layer runs on the GPU:**
  - making the random candidates;
  - picking the best;
  - the hill climb, rounding and painting;
  - updating the error map.

  It is recorded once as a CUDA graph and replayed for each layer. The CPU reads back about 100 bytes per layer.
- **A wider hill climb.** It climbs from the best 8 random shapes, not 1, over 8 rounds whose tweak sizes shrink as it goes. It spends the same number of tweaks per layer (`mutatedSamples`) as the Go generator.

Two smaller changes from the Go generator:

- The final shape's colour is rounded to 8 bits before it is scored and painted. The canvas then matches the exported JSON exactly.
- A layer is only added if the rounded shape still lowers the error.

## Which GPUs

| Series | Architecture | Support |
|--------|--------------|---------|
| RTX 20, GTX 16 | Turing (sm_75) | Built for it |
| RTX 30 | Ampere (sm_86) | Built for it |
| RTX 40 | Ada (sm_89) | Built for it |
| RTX 50 | Blackwell (sm_120) | Built for it |
| Newer | | Compiled by the driver from the portable code (PTX) on first run |
| GTX 10 and older | Pascal and earlier | Not supported |

The EXE is built with CUDA 12.9, so it needs a recent NVIDIA driver (575 or newer). With an older driver it reports "CUDA driver version is insufficient"; updating the driver fixes it. Super slow on a ~2,000 px image needs about 400 MB of GPU memory.

## Trying it (Windows, NVIDIA GPU)

**Easiest:** download `forza-painter-fh6-cuda.zip` from the [CUDA generator preview](https://github.com/Bauer-Haus/FH6-/releases/tag/cuda-preview) release. It is the whole app as one EXE with this generator inside, plus the self-test and benchmark tools, and a `START HERE.txt`. Nothing else to install besides the NVIDIA driver. Every push to the `cuda-generator` branch rebuilds it.

The steps below are for using the generator with the app run from source.

1. **Download.** Open the latest **cuda-generator** run under the repository's **Actions** tab and download the `forza-painter-geometrize-cuda` artifact. It contains the EXE, `bench.ps1` and this file. You need only an up-to-date NVIDIA driver; the CUDA runtime is built into the EXE.
2. **Self-test.** In a terminal, run:

   ```powershell
   .\forza-painter-geometrize-cuda.exe --selftest
   ```

   It compares the GPU with the CPU reference (scores, whole layers, the canvas and the tables), then times layers at the `c. balanced` sample counts. It should end with `SELFTEST PASSED`. If it doesn't, send the output.
3. **Benchmark against the Go generator.** Copy `bench.ps1` and the EXE into `painter\bin`, next to `forza-painter-geometrize-go.exe`, then run:

   ```powershell
   .\bench.ps1 -Images C:\path\car.png, C:\path\logo.png -StopAt 1000
   ```

   Both generators run with the same settings (by default `c. balanced`) and the same seed. The script prints seconds, ms per layer and RMSE for each. Lower RMSE is better.
4. **Use it in the app.**
   1. In `painter\bin`, rename `forza-painter-geometrize-go.exe` to keep it, for example to `forza-painter-geometrize-go.original.exe`.
   2. Copy `forza-painter-geometrize-cuda.exe` in under the name `forza-painter-geometrize-go.exe`.
   3. Generate as usual, import into FH6 and check the result in game.

   To go back, restore the original file.

## Command line

```
forza-painter-geometrize-cuda [-settings file.ini | -profile name] [-output base] [-preview file.png]
                              [-seed n] [-resume checkpoint.json] [--cpu] <image>
forza-painter-geometrize-cuda --selftest [image]
forza-painter-geometrize-cuda --score <geometry.json> <image>
```

- Options may come before or after the image.
- `-backend` is accepted and ignored.
- `--cpu` runs the same search on the CPU. It is slow, and meant for testing.
- `--score` prints the RMSE (0–255 scale) and PSNR of any geometry file against its image. It paints the shapes with their 8-bit colours, as the game does.
- With `FPG_PROFILE=1` set, the average GPU time per layer is printed every 10 layers.

## Settings

All the Go generator's keys are read, with the same defaults and checks. These keys have no effect here: `useWorkGroupEval`, `maxThreads`, progressive sampling and `randomSampleBudget`. The log says so when a file sets them. `errorMetric = ssim` falls back to MSE with a warning, as the Go Vulkan backend does. Multi-primitive shapes (rectangles and triangles) are not supported yet.

Two new optional keys tune the hill climb:

| Key | Default | Meaning |
|-----|---------|---------|
| `climbStarts` | 8 | How many of the best random shapes are climbed from (at most 32). |
| `climbRounds` | 8 | Rounds per climb. Tweak sizes shrink from the Go generator's steps to a fifth of them. |

## Building

**Windows:** CUDA Toolkit 12.x, Visual Studio 2022 and CMake 3.24 or newer.

```powershell
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

The GitHub workflow `.github/workflows/cuda-generator.yml` does this for every push that touches this folder.

**Any OS, without CUDA:** builds the CPU backend and the tests.

```
cmake -S . -B build -DFPG_CPU_ONLY=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

The tests check:

- the size penalty, radius snapping, rounding, mutation steps, score normalising and JSON number format, against values printed by the Go code in `tools/go-reference`;
- image loading and the Catmull-Rom resize, against Go's output for the same PNG;
- row runs against the per-pixel test, on 3,000 shapes;
- row-total scores against pixel-by-pixel scores;
- painting, and the row totals kept up to date while painting;
- that each layer's score equals the real change in error;
- a whole run, with checkpoints, previews and resume.

To regenerate the Go reference values:

```
cd tools/go-reference && go run . ../../tests
```

## Layout

| Path | What it is |
|------|------------|
| `src/core.hpp` | Maths shared by the CPU and GPU: geometry, row runs, the score formula, rounding, random numbers. |
| `src/gpu/backend_cuda.cu` | The CUDA kernels and the layer graph. |
| `src/backend_cpu.cpp` | The same search on the CPU, step for step. |
| `src/engine.cpp` | The run loop: logging, checkpoints, previews, pruning, resume. |
| `src/settings.cpp`, `src/image_io.cpp`, `src/geometry_json.cpp`, `src/platform.cpp` | Settings, images, JSON, files. |
| `src/tables.cpp` | Table building, the per-pixel reference, pruning. |
| `src/tools.cpp` | `--selftest` and `--score`. |
| `third_party/stb` | `stb_image` and `stb_image_write` (public domain). |
