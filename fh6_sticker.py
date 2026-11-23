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
  out/painter.json     (with --painter-json) geometry JSON for the bundled
                       forza-painter FH6 importer in ./painter, so the layers are
                       written into the game instead of placed by hand

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

# Outline points of a unit circle / square, computed once (polygon() scales them).
_T = np.linspace(0, 2 * math.pi, 48, endpoint=False)
_UNIT_CIRCLE = (np.cos(_T), np.sin(_T))
_UNIT_SQUARE = (np.array([-1.0, 1.0, 1.0, -1.0]), np.array([-1.0, -1.0, 1.0, 1.0]))


class Shape:
    """kind: 'circle' (ellipse) or 'square' (rectangle). All units in work pixels."""

    def __init__(self, kind, cx, cy, w, h, rot):
        self.kind, self.cx, self.cy, self.w, self.h, self.rot = kind, cx, cy, w, h, rot

    def polygon(self, scale=1):
        a = math.radians(self.rot)
        ca, sa = math.cos(a), math.sin(a)
        hw, hh = self.w / 2, self.h / 2
        ux, uy = _UNIT_CIRCLE if self.kind == "circle" else _UNIT_SQUARE
        x, y = ux * hw, uy * hh
        px = (self.cx + x * ca - y * sa) * scale
        py = (self.cy + x * sa + y * ca) * scale
        return list(zip(px.tolist(), py.tolist()))


# --------------------------------------------------------------------------- fitting
#
# Shapes are fitted as rows of a (B, 6) float array: cx, cy, w, h, rot (degrees,
# rotated the same way as Shape.polygon), is_circle. Both shape kinds are convex,
# so each one covers a single run of pixels [x0, x1) on every image row. A pixel
# belongs to a shape when its centre is inside. With per-row running totals of the
# target and error images, a shape's best colour and error change come from two
# lookups per row instead of touching every pixel it covers, and a whole batch of
# shapes is scored in one numpy call.

CX, CY, W_, H_, ROT, CIRC = range(6)
SCORE_CHUNK = 32  # candidates scored together; see _Scorer.score_many


def _line_range(k, c, r):
    """For each element, the x range where -r <= k*x + c <= r (may be empty or unbounded)."""
    safe = np.abs(k) > 1e-9
    ks = np.where(safe, k, 1.0)
    a, b = (-r - c) / ks, (r - c) / ks
    inside = np.abs(c) <= r
    lo = np.where(safe, np.minimum(a, b), np.where(inside, -np.inf, np.inf))
    hi = np.where(safe, np.maximum(a, b), np.where(inside, np.inf, -np.inf))
    return lo, hi


def shape_spans(P, W, H, rows=None):
    """Per-row pixel runs of shapes P (B, 6) on the given rows (default all):
    integer arrays x0, x1 of shape (B, len(rows))."""
    cx, cy, w, h, rot, circ = (P[:, j:j + 1] for j in range(6))
    Y = (np.arange(H) if rows is None else rows) + 0.5 - cy   # row centres relative to the shape
    a = np.radians(rot)
    ca, sa = np.cos(a), np.sin(a)
    hw, hh = w / 2, h / 2
    # Ellipse: local u = X*ca + Y*sa, v = -X*sa + Y*ca, inside when (u/hw)^2 + (v/hh)^2 <= 1.
    # For a fixed row that is a quadratic A*X^2 + B*X + C <= 0 in X.
    ia, ib = 1 / (hw * hw), 1 / (hh * hh)
    A = ca * ca * ia + sa * sa * ib
    B = 2 * ca * sa * Y * (ia - ib)
    C = Y * Y * (sa * sa * ia + ca * ca * ib) - 1
    disc = B * B - 4 * A * C
    root = np.sqrt(np.maximum(disc, 0))
    el_lo = np.where(disc >= 0, (-B - root) / (2 * A), np.inf)
    el_hi = np.where(disc >= 0, (-B + root) / (2 * A), -np.inf)
    # Rectangle: |u| <= hw and |v| <= hh, each linear in X.
    lo1, hi1 = _line_range(ca, sa * Y, hw)
    lo2, hi2 = _line_range(-sa, ca * Y, hh)
    is_c = circ > 0.5
    lo = np.where(is_c, el_lo, np.maximum(lo1, lo2)) + cx
    hi = np.where(is_c, el_hi, np.minimum(hi1, hi2)) + cx
    x0 = np.clip(np.ceil(lo - 0.5), 0, W).astype(np.intp)
    x1 = np.clip(np.floor(hi - 0.5) + 1, 0, W).astype(np.intp)
    return x0, np.maximum(x1, x0)


def shape_mask(P, W, H):
    """Boolean (H, W) mask of one shape P (6,), using the same pixel rule as the fitter."""
    x0, x1 = shape_spans(P[None], W, H)
    cols = np.arange(W)
    return (cols >= x0[0][:, None]) & (cols < x1[0][:, None])


class _Scorer:
    """Per-row running totals of the target (colour and squared colour) and of the
    current error map, so any shape is scored from its row runs alone."""

    def __init__(self, target, err_map):
        H, W, _ = target.shape
        self.rows = np.arange(H)
        self.ct = np.zeros((H, W + 1, 3))
        self.ct[:, 1:] = np.cumsum(target, axis=1)
        self.cq = np.zeros((H, W + 1))
        self.cq[:, 1:] = np.cumsum((target ** 2).sum(axis=2), axis=1)
        self.ce = np.zeros((H, W + 1))
        self.ce[:, 1:] = np.cumsum(err_map, axis=1)

    def refresh_rows(self, err_map, rows):
        self.ce[rows, 1:] = np.cumsum(err_map[rows], axis=1)

    def score(self, P, W, H):
        """(delta, colour) for each shape: the change in total squared error from
        painting it opaque in its mean colour. delta is inf for shapes under 2 px.
        Only rows some shape in P can reach are looked at, so keep batches local
        (see score_many)."""
        reach = np.hypot(P[:, W_], P[:, H_]) / 2      # covers the shape at any rotation
        y0 = max(int(np.floor((P[:, CY] - reach).min())), 0)
        y1 = min(int(np.ceil((P[:, CY] + reach).max())) + 1, H)
        r = self.rows[y0:max(y1, y0)]
        x0, x1 = shape_spans(P, W, H, r)
        n = (x1 - x0).sum(axis=1)
        s = (self.ct[r, x1] - self.ct[r, x0]).sum(axis=1)
        q = (self.cq[r, x1] - self.cq[r, x0]).sum(axis=1)
        e = (self.ce[r, x1] - self.ce[r, x0]).sum(axis=1)
        nn = np.maximum(n, 1)
        delta = q - (s * s).sum(axis=1) / nn - e      # sum((t - mean)^2) - old error
        return np.where(n >= 2, delta, np.inf), s / nn[:, None]


    def score_many(self, P, W, H, chunk=SCORE_CHUNK):
        """score() for a large, spread-out batch: shapes are grouped by height on the
        image so each group only reads the rows it needs."""
        if len(P) <= chunk:
            return self.score(P, W, H)
        order = np.argsort(P[:, CY])
        d, c = np.empty(len(P)), np.empty((len(P), 3))
        for k in range(0, len(P), chunk):
            idx = order[k:k + chunk]
            d[idx], c[idx] = self.score(P[idx], W, H)
        return d, c


def error_cdf(err_map):
    """Running total of the error map, for picking pixels in proportion to their
    error. Build it once per layer (the map only changes when a layer is added)."""
    return np.cumsum(err_map.ravel())


def random_shapes(n, W, H, kinds, max_frac=0.4, cdf=None):
    """n new random shapes (n, 6). With cdf (from error_cdf), centres are picked where
    the picture is still most wrong, so later layers go to the details."""
    P = np.empty((n, 6))
    if cdf is not None:
        tot = cdf[-1]
        if tot > 0:
            idx = np.minimum(np.searchsorted(cdf, np.random.random(n) * tot, side="right"), cdf.size - 1)
        else:
            idx = np.random.randint(0, cdf.size, n)
        cy, cx = np.divmod(idx, W)
        P[:, CX] = cx + np.random.random(n)
        P[:, CY] = cy + np.random.random(n)
    else:
        P[:, CX] = np.random.uniform(0, W, n)
        P[:, CY] = np.random.uniform(0, H, n)
    hi = max(max(W, H) * max_frac, 3)
    P[:, W_] = np.random.uniform(1.5, hi, n)
    P[:, H_] = np.random.uniform(1.5, hi, n)
    P[:, ROT] = np.random.uniform(0, 180, n)
    P[:, CIRC] = np.array([k == "circle" for k in kinds], float)[np.random.randint(0, len(kinds), n)]
    return P


def mutate(P, W, H):
    """Copy of shapes P (n, 6) with one random tweak each: move, resize or rotate."""
    P = P.copy()
    n = len(P)
    kind = np.random.randint(0, 4, n)
    g1, g2 = np.random.standard_normal(n), np.random.standard_normal(n)
    m = kind == 0
    P[m, CX] = np.clip(P[m, CX] + g1[m] * W * 0.05, 0, W)
    P[m, CY] = np.clip(P[m, CY] + g2[m] * H * 0.05, 0, H)
    m = kind == 1
    P[m, W_] = np.clip(P[m, W_] + g1[m] * W * 0.05, 1.5, W * 1.5)
    m = kind == 2
    P[m, H_] = np.clip(P[m, H_] + g1[m] * H * 0.05, 1.5, H * 1.5)
    m = kind == 3
    P[m, ROT] = (P[m, ROT] + g1[m] * 20) % 180
    return P


def to_shape(p):
    return Shape("circle" if p[CIRC] > 0.5 else "square", *(float(v) for v in p[:5]))


CLIMB_BATCH = 4  # mutations tried at once per hill-climb step


def fit(target, n_shapes, kinds, candidates=150, climbs=120, log=print):
    H, W, _ = target.shape
    target = np.asarray(target, dtype=np.float64)
    bg = np.median(target.reshape(-1, 3), axis=0)
    current = np.tile(bg, (H, W, 1))
    err_map = ((target - current) ** 2).sum(axis=2)
    scorer = _Scorer(target, err_map)
    layers = []
    for i in range(n_shapes):
        # shapes start large and shrink as the picture fills in
        frac = 0.4 * (1 - i / n_shapes) ** 2 + 0.02
        P = random_shapes(candidates, W, H, kinds, frac, error_cdf(err_map) if i else None)
        d, c = scorer.score_many(P, W, H)
        k = int(np.argmin(d))
        p, d, c = P[k], d[k], c[k]
        for _ in range(max(1, climbs // CLIMB_BATCH)):
            M = mutate(np.repeat(p[None], CLIMB_BATCH, axis=0), W, H)
            d2, c2 = scorer.score(M, W, H)
            k = int(np.argmin(d2))
            if d2[k] < d:
                p, d, c = M[k], d2[k], c2[k]
        if not d < 0:
            log(f"  layer {i + 1}: no improvement found, stopping early")
            break
        mk = shape_mask(p, W, H)
        current[mk] = c
        err_map[mk] = ((target[mk] - c) ** 2).sum(axis=1)
        scorer.refresh_rows(err_map, np.flatnonzero(mk.any(axis=1)))
        layers.append((to_shape(p), tuple(int(round(v)) for v in c)))
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


def write_painter_json(path, bg, layers, W, H, out_size):
    """Write forza-painter geometry JSON.

    Format (see painter/src/geometry_json.py): shape 0 is the background
    rectangle [0, 0, width, height]; ellipses are type 16 with data
    [cx, cy, radius_x, radius_y, rotation_deg] in image pixels, rotation
    clockwise. Coordinates are scaled from the working image up to out_size
    (longest side) to match what the GPU generator produces.
    """
    k = out_size / max(W, H)
    shapes = [{"type": 1, "data": [0, 0, round(W * k), round(H * k)], "color": [*bg, 255], "score": 0}]
    for s, c in layers:
        if s.kind != "circle":
            raise ValueError("forza-painter import only supports ellipses; fit with --kinds circle")
        shapes.append({
            "type": 16,
            "data": [round(s.cx * k), round(s.cy * k), round(s.w / 2 * k, 2), round(s.h / 2 * k, 2),
                     round(s.rot) % 360],
            "color": [*c, 255],
            "score": 0,
        })
    with open(path, "w") as f:
        json.dump({"shapes": shapes}, f)


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
    p.add_argument("--painter-json", action="store_true",
                   help="also write painter.json for the forza-painter importer in ./painter "
                        "(uses ellipses only, because that importer has no rotated rectangles)")
    p.add_argument("--seed", type=int, default=None, help="random seed for repeatable results")
    a = p.parse_args()

    if a.seed is not None:
        np.random.seed(a.seed)
    kinds = [k.strip() for k in a.kinds.split(",") if k.strip() in ("circle", "square")]
    if not kinds:
        sys.exit("--kinds must include circle and/or square")
    if a.painter_json and kinds != ["circle"]:
        print("--painter-json: fitting with ellipses only (the importer has no rotated rectangles).")
        kinds = ["circle"]

    print(f"Loading {a.source} ...")
    img = load_image(a.source)
    orig_longest = max(img.size)
    size = a.size or (WORK_SIZE if a.shapes <= 200 else 320)
    img.thumbnail((size, size), Image.LANCZOS)
    W, H = img.size
    target = np.asarray(img, dtype=np.float64)
    print(f"Fitting {a.shapes} shapes on a {W}x{H} working image ...")
    bg, layers = fit(target, a.shapes, kinds, candidates=a.quality)
    write_outputs(a.out, a.source, bg, layers, W, H)
    if a.painter_json:
        pj = os.path.join(a.out, "painter.json")
        write_painter_json(pj, bg, layers, W, H, min(orig_longest, 1400))
        print(f"Wrote {pj}. Import it with the forza-painter app (painter/README.md, 'Import JSON').")
    print(f"Done. Open {os.path.join(a.out, 'layers.html')} and build the group in-game.")


if __name__ == "__main__":
    main()
