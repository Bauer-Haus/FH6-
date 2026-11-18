from __future__ import annotations

from pathlib import Path

import generator_backend
from generator_backend import SettingProfile, parse_settings, write_custom_settings, write_user_settings_preset


def _base(tmp_path: Path) -> SettingProfile:
    path = tmp_path / "base.ini"
    path.write_text(
        "description = base\n"
        "stopAt = 2000\n"
        "errorMetric = ssim\n"
        "ssimWeight = 0.4\n"
        "randomSampleBudget = 1024\n"
        "someFutureKey = 7\n",
        encoding="utf-8",
    )
    return SettingProfile(index=1, source="bundled", path=path, label="1. base")


def test_custom_settings_keep_every_generator_key(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(generator_backend, "CUSTOM_SETTINGS_DIR", tmp_path / "custom")
    profile = write_custom_settings(_base(tmp_path), {"stopAt": "500"})
    values = parse_settings(profile.path)
    assert values["stopAt"] == "500"
    assert values["errorMetric"] == "ssim"
    assert values["ssimWeight"] == "0.4"
    assert values["randomSampleBudget"] == "1024"
    assert values["someFutureKey"] == "7"
    assert values["description"] == "Custom UI settings"


def test_user_preset_keeps_every_generator_key(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(generator_backend, "USER_SETTINGS_DIR", tmp_path / "user")
    path = write_user_settings_preset(_base(tmp_path), {}, tmp_path / "user" / "mine")
    values = parse_settings(path)
    assert path.suffix == ".ini"
    assert values["errorMetric"] == "ssim"
    assert values["randomSampleBudget"] == "1024"
