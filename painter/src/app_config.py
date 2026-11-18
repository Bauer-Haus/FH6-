from __future__ import annotations

from app_paths import ROOT, SOURCE_DIR


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
APP_DIR = SOURCE_DIR
PROBE_DIR = ROOT / "webui-data" / "probes"
SESSION_PATH = PROBE_DIR / "current-fh6-session.json"
FULL_SHAPE_ROOT = ROOT / "runtime" / "full-shape-io"
# Generate-page outputs: image/<ext>/ archives every source image that was
# sent to the generator; out/<image stem>/<ext>/ collects the JSON files of
# one image and never holds images.
GENERATED_IMAGE_DIR = ROOT / "image"
GENERATED_OUTPUT_DIR = ROOT / "out"


# ---------------------------------------------------------------------------
# Limits & thresholds
# ---------------------------------------------------------------------------
MEMORY_SNAPSHOT_LIMIT_MB = 2048
PREVIEW_MAX = 520
DETAILED_LOG_OUTPUT_LIMIT = 50000
DETAILED_LOG_MEMORY_LIMIT = 120000
FH6_AUTO_LOCATE_MAX_SECONDS = 300
FH6_AUTO_LOCATE_TIMEOUT_SECONDS = 360
UPDATE_CHECK_TIMEOUT_SECONDS = 8


# ---------------------------------------------------------------------------
# Update URLs
# ---------------------------------------------------------------------------
UPDATE_VERSION_URL = (
    "https://raw.githubusercontent.com/bvzrays/forza-painter-fh6/main/src/version.py"
)
UPDATE_CHANGELOG_URL = (
    "https://raw.githubusercontent.com/bvzrays/forza-painter-fh6/main/CHANGELOG.md"
)
UPDATE_RELEASE_URL = "https://github.com/bvzrays/forza-painter-fh6/releases/latest"
MARKET_URL = "https://painter6.com"


# ---------------------------------------------------------------------------
# Theme colours (dark)
# ---------------------------------------------------------------------------
class Theme:
    BG = "#0e1014"
    PANEL = "#15181e"
    PANEL_ALT = "#1d2129"
    INPUT = "#0a0c10"
    TEXT = "#eef0f4"
    MUTED = "#8c95a3"
    # Accent: text/highlights use ACCENT; filled surfaces with white text use
    # ACCENT_DARK (contrast >= 4.5:1), brightening to ACCENT_HOVER under the mouse.
    ACCENT = "#ff5c93"
    ACCENT_DARK = "#c2185b"
    ACCENT_HOVER = "#d81b60"
    WARN = "#ffc857"
    SUCCESS = "#3ddc97"
    BORDER = "#2a2f39"
    BUTTON = "#242933"
    BUTTON_HOVER = "#2e3440"
    BUTTON_ACTIVE = "#363d4b"
    FONT = "Segoe UI"
    MONO = "Consolas"
