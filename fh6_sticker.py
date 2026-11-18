#!/usr/bin/env python3
"""
fh6_sticker.py - turn a photo (URL or local file) into a Forza Horizon vinyl group.

Forza Horizon has no image import: every decal is a stack of vinyl shapes.
This tool approximates your picture with a small number of opaque shapes
(ellipses and rotated rectangles, which match the game's Circle / Square
primitives) and writes:

  out/preview.png      what the finished vinyl group should look like
  out/progress.png     the build-up after 10%, 25%, 50%, 100% of the layers
  out/layers.json      machine-readable layer list
  out/layers.html      printable layer sheet: one row per layer, in placement order

Then, in-game: Creative Hub -> Paints, Liveries & Designs -> Create Vinyl Group,
add the layers from the sheet bottom-to-top (layer 1 first), save the group and
apply it to your car in the livery editor.

Usage:
  python fh6_sticker.py https://example.com/logo.png
  python fh6_sticker.py photo.jpg --shapes 80 --out mylogo
"""

import argparse
import html
import io
import json
import math
import os
import random
import sys
import urllib.request

import numpy as np
from PIL import Image, ImageDraw

WORK_SIZE = 160  # default longest side used for fitting (speed vs. detail); see --size


# --------------------------------------------------------------------------- input

def load_image(src: str) -> Image.Image:
    if src.startswith(("http://", "https://")):
        req = urllib.request.Request(src, headers={"User-Agent": "fh6-sticker/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
        except Exception as e:
            sys.exit(f"Could not download {src}: {e}\n"
                     "Tip: right-click the image -> 'Copy image address' so the URL points "
                     "at the picture itself, or save it and pass the file path instead.")
        img = Image.open(io.BytesIO(data))
    else:
        img = Image.open(src)
    img.load()
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    return img.convert("RGB")


# --------------------------------------------------------------------------- shapes

class Shape:
    """kind: 'circle' (ellipse) or 'square' (rectangle). All units in work pixels."""

    def __init__(self, kind, cx, cy, w, h, rot):
        self.kind, self.cx, self.cy, self.w, self.h, self.rot = kind, cx, cy, w, h, rot

    def copy(self):
        return Shape(self.kind, self.cx, self.cy, self.w, self.h, self.rot)

    def mask(self, W, H):
        """(region, mask): region is a (row-slice, col-slice) of the canvas, mask a
        boolean array of the shape inside that region. Only the bounding box is
        rasterised, which keeps big layer counts fast."""
        pts = self.polygon()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x0, x1 = max(int(min(xs)), 0), min(int(math.ceil(max(xs))) + 1, W)
        y0, y1 = max(int(min(ys)), 0), min(int(math.ceil(max(ys))) + 1, H)
        if x1 - x0 < 1 or y1 - y0 < 1:
            return None, None
        m = Image.new("L", (x1 - x0, y1 - y0), 0)
        ImageDraw.Draw(m).polygon([(x - x0, y - y0) for x, y in pts], fill=255)
        return (slice(y0, y1), slice(x0, x1)), np.asarray(m) > 127

    def polygon(self, scale=1):
        a = math.radians(self.rot)
        ca, sa = math.cos(a), math.sin(a)
        hw, hh = self.w / 2, self.h / 2
        if self.kind == "circle":
            local = [(hw * math.cos(t), hh * math.sin(t))
                     for t in np.linspace(0, 2 * math.pi, 48, endpoint=False)]
        else:
            local = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        return [((self.cx + x * ca - y * sa) * scale, (self.cy + x * sa + y * ca) * scale)
                for x, y in local]

    def mutate(self, W, H):
        s = self.copy()
        r = random.randint(0, 3)
        if r == 0:
            s.cx = min(max(s.cx + random.gauss(0, W * 0.05), 0), W)
            s.cy = min(max(s.cy + random.gauss(0, H * 0.05), 0), H)
        elif r == 1:
            s.w = min(max(s.w + random.gauss(0, W * 0.05), 1.5), W * 1.5)
        elif r == 2:
            s.h = min(max(s.h + random.gauss(0, H * 0.05), 1.5), H * 1.5)
        else:
            s.rot = (s.rot + random.gauss(0, 20)) % 180
        return s


def random_shape(W, H, kinds, max_frac=0.4, err_map=None):
    """New random shape. With err_map, the centre is picked where the picture is
    still most wrong, so later layers go to the details."""
    m = max(W, H)
    if err_map is not None:
        flat = err_map.ravel()
        tot = flat.sum()
        idx = np.random.choice(flat.size, p=flat / tot) if tot > 0 else random.randrange(flat.size)
        cy, cx = divmod(int(idx), W)
        cx, cy = cx + random.random(), cy + random.random()
    else:
        cx, cy = random.uniform(0, W), random.uniform(0, H)
    hi = max(m * max_frac, 3)
    return Shape(random.choice(kinds), cx, cy, random.uniform(1.5, hi), random.uniform(1.5, hi),
                 random.uniform(0, 180))


# --------------------------------------------------------------------------- fitting

def evaluate(shape, target, current, err_map):
    """Return (new_total_error_delta, color, mask). Opaque fill with best (mean) color."""
    H, W, _ = target.shape
    region, m = shape.mask(W, H)
    if m is None or m.sum() < 2:
        return 0.0, None, None
    t = target[region][m]
    color = t.mean(axis=0)
    new_err = ((t - color) ** 2).sum()
    old_err = err_map[region][m].sum()
    return new_err - old_err, color, (region, m)


def fit(target, n_shapes, kinds, candidates=150, climbs=120, log=print):
    H, W, _ = target.shape
    bg = np.median(target.reshape(-1, 3), axis=0)
    current = np.tile(bg, (H, W, 1))
    err_map = ((target - current) ** 2).sum(axis=2)
    layers = []
    for i in range(n_shapes):
        best = None
        for _ in range(candidates):
            # shapes start large and shrink as the picture fills in
            frac = 0.4 * (1 - i / n_shapes) ** 2 + 0.02
            s = random_shape(W, H, kinds, frac, err_map if i else None)
            d, c, m = evaluate(s, target, current, err_map)
            if c is not None and (best is None or d < best[0]):
                best = (d, s, c, m)
        d, s, c, m = best
        for _ in range(climbs):
            s2 = s.mutate(W, H)
            d2, c2, m2 = evaluate(s2, target, current, err_map)
            if c2 is not None and d2 < d:
                d, s, c, m = d2, s2, c2, m2
        if d >= 0:
            log(f"  layer {i + 1}: no improvement found, stopping early")
            break
        region, mk = m
        current[region][mk] = c
        err_map[region][mk] = ((target[region][mk] - c) ** 2).sum(axis=1)
        layers.append((s, tuple(int(round(v)) for v in c)))
        if (i + 1) % (10 if n_shapes <= 200 else 50) == 0 or i + 1 == n_shapes:
            rmse = math.sqrt(err_map.mean() / 3)
            log(f"  {i + 1}/{n_shapes} layers, error {rmse:.1f}")
    return tuple(int(round(v)) for v in bg), layers


# --------------------------------------------------------------------------- output

def render(bg, layers, W, H, scale, upto=None):
    img = Image.new("RGB", (W * scale, H * scale), bg)
    d = ImageDraw.Draw(img)
    for s, c in layers[:upto]:
        d.polygon(s.polygon(scale), fill=c)
    return img


def to_game_units(s, W, H):
    """Express a shape relative to the sticker canvas.

    Position: centre offset from the canvas centre, as % of the canvas' longest side
              (+x right, +y up). Size: % of the longest side. Rotation: degrees clockwise.
    Use these as *relative* values: put layer 1 (the background) at a size you like,
    then scale every other layer by the same factor.
    """
    m = max(W, H)
    return {
        "x": round((s.cx - W / 2) / m * 100, 2),
        "y": round((H / 2 - s.cy) / m * 100, 2),
        "width": round(s.w / m * 100, 2),
        "height": round(s.h / m * 100, 2),
        "rotation": round(s.rot % 180, 1),
    }


def hexc(c):
    return "#{:02X}{:02X}{:02X}".format(*c)


def write_outputs(out, src, bg, layers, W, H):
    os.makedirs(out, exist_ok=True)
    scale = max(1, round(640 / max(W, H)))
    render(bg, layers, W, H, scale).save(os.path.join(out, "preview.png"))

    steps = [max(1, int(len(layers) * f)) for f in (0.1, 0.25, 0.5, 1.0)]
    tiles = [render(bg, layers, W, H, 2, k) for k in steps]
    strip = Image.new("RGB", (sum(t.width for t in tiles) + 10 * (len(tiles) - 1), tiles[0].height), "white")
    x = 0
    for t in tiles:
        strip.paste(t, (x, 0))
        x += t.width + 10
    strip.save(os.path.join(out, "progress.png"))

    rows = [{"layer": 1, "shape": "square", "color": hexc(bg), "x": 0, "y": 0,
             "width": round(W / max(W, H) * 100, 2), "height": round(H / max(W, H) * 100, 2),
             "rotation": 0, "note": "background"}]
    for i, (s, c) in enumerate(layers, start=2):
        rows.append({"layer": i, "shape": s.kind, "color": hexc(c), **to_game_units(s, W, H)})

    with open(os.path.join(out, "layers.json"), "w") as f:
        json.dump({"source": src, "units": "percent of sticker's longest side; +y is up; "
                   "rotation in degrees clockwise", "layers": rows}, f, indent=2)

    trs = "\n".join(
        f"<tr><td>{r['layer']}</td><td>{r['shape'].title()}</td>"
        f"<td><span class=sw style='background:{r['color']}'></span>{r['color']}</td>"
        f"<td>{r['x']}</td><td>{r['y']}</td><td>{r['width']}</td><td>{r['height']}</td>"
        f"<td>{r['rotation']}°</td><td><input type=checkbox></td></tr>" for r in rows)
    page = f"""<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>FH6 Layer Sheet</title><style>
body{{font:14px system-ui,sans-serif;margin:16px;background:#fff;color:#111}}
img{{max-width:100%;border:1px solid #ccc}} table{{border-collapse:collapse;width:100%}}
th,td{{border-bottom:1px solid #ddd;padding:4px 8px;text-align:right}} th{{position:sticky;top:0;background:#f4f4f4}}
td:nth-child(2),td:nth-child(3){{text-align:left}} .sw{{display:inline-block;width:14px;height:14px;
border:1px solid #888;vertical-align:middle;margin-right:6px}} tr:has(input:checked){{opacity:.35}}
</style></head><body>
<h1>Vinyl group layer sheet</h1>
<p>Source: {html.escape(src)}<br>{len(rows)} layers. Place them in order, layer 1 at the bottom.
Units: % of the sticker's longest side; x right, y up, rotation clockwise.
Tick each row as you go.</p>
<p><img src="preview.png" alt="Target result"></p>
<table><tr><th>#</th><th>Shape</th><th>Colour</th><th>X</th><th>Y</th><th>Width</th><th>Height</th><th>Rotation</th><th>Done</th></tr>
{trs}</table></body></html>"""
    with open(os.path.join(out, "layers.html"), "w") as f:
        f.write(page)


# --------------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description="Convert a photo into Forza Horizon vinyl layers.")
    p.add_argument("source", help="image URL or local file path")
    p.add_argument("--shapes", type=int, default=60,
                   help="number of layers to fit (default 60; more = closer match, more work in-game)")
    p.add_argument("--kinds", default="circle,square",
                   help="comma-separated shapes to use: circle, square (default both)")
    p.add_argument("--out", default="out", help="output folder (default ./out)")
    p.add_argument("--quality", type=int, default=150,
                   help="random candidates tried per layer (default 150; higher = slower, better)")
    p.add_argument("--size", type=int, default=None,
                   help="working resolution in pixels (default 160, or 320 above 200 shapes)")
    p.add_argument("--seed", type=int, default=None, help="random seed for repeatable results")
    a = p.parse_args()

    if a.seed is not None:
        random.seed(a.seed)
        np.random.seed(a.seed)
    kinds = [k.strip() for k in a.kinds.split(",") if k.strip() in ("circle", "square")]
    if not kinds:
        sys.exit("--kinds must include circle and/or square")

    print(f"Loading {a.source} ...")
    img = load_image(a.source)
    size = a.size or (WORK_SIZE if a.shapes <= 200 else 320)
    img.thumbnail((size, size), Image.LANCZOS)
    W, H = img.size
    target = np.asarray(img, dtype=np.float64)
    print(f"Fitting {a.shapes} shapes on a {W}x{H} working image ...")
    bg, layers = fit(target, a.shapes, kinds, candidates=a.quality)
    write_outputs(a.out, a.source, bg, layers, W, H)
    print(f"Done. Open {os.path.join(a.out, 'layers.html')} and build the group in-game.")


if __name__ == "__main__":
    main()
