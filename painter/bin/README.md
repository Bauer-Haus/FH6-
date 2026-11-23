# Generator EXE

| File | What it is |
|------|------------|
| `forza-painter-geometrize-go.exe` | **Original**, as shipped by forza-painter-fh6 v1.9.7 (upstream author's build; its exact source is not public). The app uses this one. |

## The fixed, faster build

A build of `painter/generator` with the 9 bug fixes in `generator/FIXES.md` and the speed work
(faster scoring loop, all-core candidate generation, optional `randomSampleBudget`) is kept on the
**`experimental`** branch as `painter/bin/forza-painter-geometrize-go-fixed.exe`, with instructions
for trying it. It is not on `main` because it has **not been run on real Windows hardware yet**:
it was cross-compiled on Linux and only tested under Wine with a CPU OpenCL driver.

To build it yourself, run `generator/windows-cross/build.sh`. That writes
`bin/forza-painter-geometrize-go-fixed.exe`.
