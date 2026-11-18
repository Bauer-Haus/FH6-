# Generator EXEs

| File | What it is |
|------|------------|
| `forza-painter-geometrize-go.exe` | **Original**, as shipped by forza-painter-fh6 v1.9.7 (upstream author's build; its exact source is not public). The app uses this one. |
| `forza-painter-geometrize-go-fixed.exe` | Built from `painter/generator` at commit `b1b71f4`: the 9 bug fixes in `generator/FIXES.md` (including atomic preview/checkpoint writes) plus the speed work (faster scoring loop, all-core candidate generation, optional `randomSampleBudget`). Uses the Universal C Runtime built into Windows 10/11. SHA-256 `5c384a33fd3be27b4dbedacb250d66c4b9d5564fec0813218a206839c4959107`. |

The fixed EXE was cross-compiled on Linux (`generator/windows-cross/build.sh`) and
has **not been run on real Windows hardware yet**. It imports the same DLLs as the
original (`OpenCL.dll`, `vulkan-1.dll` from the graphics driver, plus system DLLs),
and it started and generated layers under Wine with a CPU OpenCL driver.

## Trying the fixed one

1. Close the app.
2. Rename `forza-painter-geometrize-go.exe` to `forza-painter-geometrize-go-original.exe` (keep it as a backup).
3. Rename `forza-painter-geometrize-go-fixed.exe` to `forza-painter-geometrize-go.exe`.
4. Optional, for extra speed: add the line `randomSampleBudget = 1024` to the preset you use in
   `config/settings/` (the original EXE ignores this line). In one test, 150 layers took 124 s
   instead of 240 s with near-identical quality; it has been tested on only one seed so far.
5. Start the app and generate as usual.

To go back, reverse the renames.
