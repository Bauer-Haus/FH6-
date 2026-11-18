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

- `--shapes N`: more layers look closer to the photo but take longer to build in-game. 30–60 works for logos, 100+ for photos.
- `--kinds circle` or `--kinds square`: use only one shape type.
- `--quality N`: try more candidate shapes per layer (slower, slightly better).
- `--out folder`: choose the output folder.
- `--seed N`: get the same result every time.

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
