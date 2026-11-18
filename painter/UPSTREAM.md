# About this folder

`painter/` is a copy of [bvzrays/forza-painter-fh6](https://github.com/bvzrays/forza-painter-fh6)
at commit `75cb365` (v1.9.7, 2026-11-16), used under its MIT licenses (`LICENSE`,
`LICENSE.custom-importer`). All original copyright and attribution notices are kept.

## Changes made here

| File | Problem | Fix |
|------|---------|-----|
| `src/native.py` | `except ctypes.WinError` – `WinError` is a function, not an exception class, so a failed `OpenProcess`/`WriteProcessMemory` raised a confusing `TypeError` instead of `ProcessNotFoundError`/`MemoryWriteError` (e.g. when not run as administrator). | Catch `OSError`. |
| `src/main.py` | The importer always exited with code 0, so the desktop app reported **done** even when the import failed (wrong menu, grouped template, too few layers, unreadable table). | `load_geometry` returns success; `main` returns 0/1; the script calls `sys.exit`. |
| `src/app.py` | The one-file EXE's `--helper` mode dropped the helper's return value, with the same effect as above. | Exit with the helper's integer result. |
| `src/main.py` | A layer slot that could not be read (pointer 0) was written at address 0 + offset; a bare `except:` also swallowed Ctrl+C. | Check the pointer first with a clear message; catch `Exception`. Fixed "1also" typo. |
| `src/geometry_json.py` | `NaN`/`Infinity` in a JSON file (accepted by Python's `json`) crashed the whole import in `round()`. | Skip just the bad shape. Test added. |
| `src/text/fonts.py` | `find_cjk_font()` and `find_font_for_text("")` called each other forever (`RecursionError`), e.g. with a blank font box in the Text tab. | Fall back to the best discovered font, then known system fonts, else a clear `FileNotFoundError`. |
| `tests/` | Font tests crashed on machines without the Windows fonts; `os.startfile` tests ran on non-Windows. | Skip cleanly instead. |

None of the memory offsets, signatures or the layer-writing order were changed.
