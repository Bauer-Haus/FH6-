"""Tests for region_painter.image_processor.apply_selection_mask."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from region_painter.image_processor import apply_selection_mask  # noqa: E402


def _random_image(rng, size, mode):
    w, h = size
    channels = 4 if mode == "RGBA" else 1
    data = rng.integers(0, 256, (h, w, channels) if channels > 1 else (h, w), dtype=np.uint8)
    return Image.fromarray(data, mode)


def test_alpha_is_multiplied_by_mask_and_colour_kept(tmp_path):
    rng = np.random.default_rng(0)
    target = _random_image(rng, (40, 30), "RGBA")
    mask = _random_image(rng, (40, 30), "L")
    src, out = tmp_path / "in.png", tmp_path / "out" / "masked.png"
    target.save(src)

    apply_selection_mask(src, mask, out)

    got = np.asarray(Image.open(out).convert("RGBA")).astype(int)
    t = np.asarray(target).astype(int)
    m = np.asarray(mask).astype(int)
    assert np.array_equal(got[..., :3], t[..., :3])
    # alpha * mask / 255, allowing 1 level of rounding difference
    assert np.abs(got[..., 3] - t[..., 3] * m / 255).max() <= 1
    # fully transparent / fully opaque mask values are exact
    assert np.all(got[..., 3][m == 0] == 0)
    assert np.array_equal(got[..., 3][m == 255], t[..., 3][m == 255])


def test_mask_is_resized_to_target(tmp_path):
    target = Image.new("RGBA", (20, 10), (10, 20, 30, 200))
    mask = Image.new("L", (10, 5), 0)
    mask.paste(255, (0, 0, 5, 5))  # left half selected
    src, out = tmp_path / "in.png", tmp_path / "out.png"
    target.save(src)

    apply_selection_mask(src, mask, out)

    alpha = np.asarray(Image.open(out))[..., 3]
    assert np.all(alpha[:, :10] == 200) and np.all(alpha[:, 10:] == 0)

