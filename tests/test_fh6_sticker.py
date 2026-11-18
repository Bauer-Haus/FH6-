"""Tests for fh6_sticker.py, including its forza-painter JSON export."""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "painter" / "src"))

import fh6_sticker  # noqa: E402
from geometry_json import ROTATED_ELLIPSE, normalize_geometry_payload  # noqa: E402


def _target():
    img = Image.new("RGB", (80, 60), "white")
    d = ImageDraw.Draw(img)
    d.ellipse((10, 10, 50, 50), fill="red")
    d.rectangle((45, 20, 75, 40), fill="navy")
    return np.asarray(img, dtype=np.float64)


def test_fit_reduces_error_and_returns_layers():
    fh6_sticker.random.seed(1)
    np.random.seed(1)
    target = _target()
    bg, layers = fh6_sticker.fit(target, 20, ["circle"], candidates=60, climbs=40, log=lambda *_: None)
    assert len(layers) > 5
    img = np.asarray(fh6_sticker.render(bg, layers, 80, 60, 1), dtype=np.float64)
    flat = np.tile(np.array(bg, dtype=np.float64), (60, 80, 1))
    assert ((img - target) ** 2).mean() < ((flat - target) ** 2).mean() / 2


def test_painter_json_is_readable_by_forza_painter(tmp_path):
    fh6_sticker.random.seed(2)
    np.random.seed(2)
    bg, layers = fh6_sticker.fit(_target(), 10, ["circle"], candidates=40, climbs=20, log=lambda *_: None)
    out = tmp_path / "painter.json"
    fh6_sticker.write_painter_json(out, bg, layers, 80, 60, 800)

    shapes = normalize_geometry_payload(json.loads(out.read_text()))["shapes"]
    assert shapes[0]["data"][:2] == [0, 0] and shapes[0]["data"][2:] == [800, 600]
    assert len(shapes) == len(layers) + 1
    s, c = layers[0]
    x, y, rx, ry, rot = shapes[1]["data"]
    assert shapes[1]["type"] == ROTATED_ELLIPSE
    assert abs(x - s.cx * 10) <= 1 and abs(y - s.cy * 10) <= 1
    assert abs(rx - s.w / 2 * 10) < 0.01 and abs(ry - s.h / 2 * 10) < 0.01
    assert shapes[1]["color"] == [*c, 255]


def test_painter_json_rejects_rectangles(tmp_path):
    sq = fh6_sticker.Shape("square", 10, 10, 5, 5, 0)
    try:
        fh6_sticker.write_painter_json(tmp_path / "p.json", (0, 0, 0), [(sq, (1, 2, 3))], 20, 20, 100)
    except ValueError:
        return
    raise AssertionError("expected ValueError for rectangle layers")
