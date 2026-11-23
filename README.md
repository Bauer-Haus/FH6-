# FH6 Sticker Maker

Turn any photo from the internet into a vinyl group you can build in Forza Horizon 6 and put on your car.

Forza has no way to import images. Every decal is built from stacked vinyl shapes. This tool works out
which shapes (circles/ellipses and squares/rectangles) and colours best rebuild your picture, then gives
you a step-by-step layer sheet to follow in the game.

## Setup (Windows, Mac or Linux)

1. Install Python 3.10+ from https://www.python.org (on Windows, tick "Add Python to PATH").
2. In this folder run:
   ```
   pip install -r requirements.txt
   ```

## Use

Right-click an image in your browser, choose **Copy image address**, then:

```
python fh6_sticker.py "https://example.com/picture.png" --shapes 60
```

This creates an `out/` folder with:

| File | What it is |
|------|------------|
| `preview.png` | What the finished sticker will look like |
| `progress.png` | The picture after 10% / 25% / 50% / 100% of the layers |
| `layers.html` | Layer sheet: shape, colour (hex), position, size and rotation for every layer, with tick boxes |
| `layers.json` | The same data in machine-readable form |

Options:

- `--shapes N`: more layers look closer to the photo but take longer to build in-game. 30–60 gives a rough look; detailed logos with text need around 1,000 (takes a few minutes to compute).
- `--size N`: working resolution in pixels. The default is 160, or 320 above 200 shapes. Higher catches finer detail but runs slower.
- `--kinds circle` or `--kinds square`: use only one shape type.
- `--quality N`: try more candidate shapes per layer (slower, slightly better).
- `--out folder`: choose the output folder.
- `--seed N`: get the same result every time. (Seeds from versions before the November 2026 speed-up give a different, similar-quality result.)

Simple, high-contrast images like logos, emblems and cartoon art work far better than busy photos.
Cropping and removing the background before you run the tool helps a lot.

## Building it in Forza Horizon 6

1. Creative Hub → **Paints, Liveries & Designs** → **Create Vinyl Group**.
2. Open `layers.html` on your phone or a second screen.
3. Add the layers **in order, layer 1 first**. Layer 1 is the background rectangle, so delete it at the end if you want a cut-out sticker.
4. For each layer, pick the shape (Circle or Square), set the colour from the hex code, then set the position, size and rotation.
   The sheet's numbers are **percentages of the sticker's width**, with x to the right, y up and rotation clockwise.
   Pick one scale factor (for example, 1% = 1 in-game unit) and use it for every layer.
5. Save the vinyl group, then apply it to your car in the livery editor.

## Automatic import with forza-painter (Windows PC)

The `painter/` folder contains [forza-painter FH6](https://github.com/bvzrays/forza-painter-fh6)
(MIT), with several bug fixes listed in `painter/UPSTREAM.md`. It writes the layers straight into
the game's Vinyl Group Editor, so you don't place them by hand. See `painter/README.md` for the full guide.

- **Best quality:** use its own GPU generator (`painter/start_app.bat`, or the release EXE).
- **No compatible GPU:** generate with this tool instead, then import the file in the app's **Import** page:
  ```
  python fh6_sticker.py "https://example.com/logo.png" --shapes 1000 --painter-json
  ```
  This writes `out/painter.json` (ellipses only, because the importer has no rotated rectangles).

Writing into the game's memory is not an official feature. Use it at your own risk.

## Tests

```
python -m pytest tests          # this tool
cd painter && python -m pytest  # forza-painter
```
