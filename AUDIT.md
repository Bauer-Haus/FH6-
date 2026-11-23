# Audit of `main`

Done on 23 November 2026, at commit `baf5a54`, on macOS. Every tracked file outside the vinyl shape data was
covered, folder by folder. The 2,800 shape files in `painter/src/data/fh6_vinyl_resources/` were checked as a set:
all 35 category folders match the type codes in `fh6_vinyl_resources.py`, with none missing and none extra.

**How files were checked**
- Python: `pyflakes` on every `.py` file, plus an import graph to find modules nothing imports.
- Docs: every relative link and image in the Markdown files.
- Scripts: file paths named in the `.bat`, `.ps1` and `.sh` scripts.
- Data: settings presets parsed as INI, duplicate-file hashes, byte-order marks.
- Test suites: the sticker suite and the painter suite (198 passed, 2 skipped, after fixing 2 failures). In the Go generator, `go test` passes for
  `imageutil` and `render`; the other packages need the Vulkan headers, which aren't installed on this machine.

## Fixes — all done

All fixes below are made. The painter suite now has 200 passed and 2 skipped; the sticker suite has 8 passed. `pyflakes`
reports nothing on `fh6_sticker.py`, `painter/src` or `painter/tests`, and every edited module imports.

| File | Was | Done |
|------|-----|------|
| `painter/tests/test_text_stroke_trace.py` | 2 tests failed on macOS. They traced whatever font the app picked, which on a Mac was the thin *STHeiti Light*, and it misses the pixel-coverage targets. Every regular-weight font checked (Arial, Helvetica, STHeiti Medium, Impact) passes easily. | The font tests use a fixed regular-weight font per platform (Arial or Microsoft YaHei on Windows, Arial or Helvetica on macOS, DejaVu or Liberation on Linux), and skip if none is installed. Also removed an unused import and an unused variable. |
| `painter/src/region_painter/image_processor.py` | Applied the selection mask with a Python loop over every pixel (about 0.6 s on a 2000×2000 image), after computing and discarding an `Image.composite`. The file started with a byte-order mark, and two docstring characters were garbled (`鈫?` for →, `脳` for ×). | Uses `ImageChops.multiply` (about 0.12 s on the same image; results match or differ by 1 alpha level, from rounding). Mark removed, characters fixed. New tests in `painter/tests/test_region_image_processor.py`. |
| `painter/src/text/geometry.py` | `estimate_typed_text_layers` ran a full `count_trace_layers_from_mask` and threw the result away (`total`). | Removed the call. Results are identical on 48 text, mode, cell-size and budget combinations, and the estimate is about 1.9× faster (7.8 s → 4.2 s for those 48). |
| `.gitignore` | Building the fixed generator would leave an untracked `painter/bin/forza-painter-geometrize-go-fixed.exe` that could be committed to `main`. | Ignored. |
| `painter/src/app.py` | `_preview_bounds` was defined twice (the first copy was dead). A commented-out trim step left an unused `trim_backup`. `HORIZONTAL` and `Spinbox` were unused, and the `"Image.Image"` type hints named an unimported `Image`. | Dead copy removed. The trim block is replaced by a two-line note saying it is off on purpose. Unused imports removed; `Image` is imported for type checking only. |
| `painter/src/main.py` | `import *` from `native` and `internal_classes` hid where 10 names came from; `importlib` was unused; `ctypes` and `sys` were imported twice. | Explicit imports, checked against both modules' definitions. (It can't be imported on macOS because `native.py` needs Windows, so it was checked by lint and compile only.) |
| `painter/src/text/cli.py` | The usage examples named a missing `text_to_json.py`. | The examples now use `python -m text.cli`, run from `painter/src`. |
| `painter/src/text/fonts.py` | The only macOS fallbacks were a missing `PingFang.ttc` and the thin *STHeiti Light*. | Added *STHeiti Medium* and *Hiragino Sans GB*. On this Mac the default is now *Hiragino Sans GB*, which meets the stroke-test coverage targets easily. Unused imports removed. |
| `painter/src/region_painter/preview_renderer.py`, `workflow.py` | `"Image.Image"` hint without an import; two unused imports. | Fixed. |
| `curved_trace.py`, `layer_masks.py`, `mandarin_chars.py`, `stroke_trace.py`, `text/ui/workspace.py`, `text/ui/color_values_editor.py`, `ui_preferences.py`, `test_fh6_layer_masks.py`, `test_text_font_complexity.py` | Unused imports; two unused `panel` lookups in `workspace.py`. | Removed. `_panel` is a plain dictionary lookup, so removing the unused calls changes nothing. |

**Still open:** re-run `go test ./...` in `painter/generator` on a machine with the OpenCL and Vulkan headers. This Mac
can only build and test `imageutil` and `render`, which pass.

## Deletes — all done

| File | Why |
|------|-----|
| `painter/assets/imgs/ayylmao.png` | Nothing in the repo referenced `painter/assets/`, and the file was byte-for-byte identical to `painter/generator/demo/ayylmao.png`, which the generator docs do reference. `painter/assets/` is now gone. |
| `painter/generator/demo/Purple.png`, `painter/generator/demo/Purple.preview.2500.png` | The upstream demo pair for "Purple". Nothing referenced them, unlike the other two demo images, which are named in the generator's benchmark tables. |

Everything else on `main` is used by the app, the sticker maker, a build or release script, a test, or the docs.
The deleted files are still in the Git history (so the repository is no smaller), and can be restored from it.

## Still open

Re-run `go test ./...` in `painter/generator` on a machine with the OpenCL and Vulkan headers.
