"""scripts/generate-icon.py — generates the source app icon for
apps/desktop/src-tauri/icons/.

A simple, deliberate first mark (not a professional final logo — see
BUILD-JOURNAL.md): a solid off-white circle, representing *sarva*
(Sanskrit: सर्व, "all / whole"), centered on a solid indigo rounded
square. Kept intentionally simple because an icon has to stay legible at
16x16 — one shape, one contrast, no gradients or fine detail to lose.

This script only produces the 1024x1024 source PNG
(apps/desktop/src-tauri/icons/app-icon.png). The full icon set (all
platform sizes + .icns/.ico) is generated FROM that source by Tauri's own
`tauri icon` CLI command, not by this script — see README.md.

Pure Pillow shape-drawing, no font/system dependency — reproducible on
any platform with the project's own dependencies installed.

Run: uv run python scripts/generate-icon.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

_SIZE = 1024
_BACKGROUND = (67, 56, 202)  # indigo
_FOREGROUND = (253, 250, 240)  # warm off-white
_CORNER_RADIUS = 220
_CIRCLE_DIAMETER_RATIO = 0.58

_OUTPUT = Path(__file__).parent.parent / "apps/desktop/src-tauri/icons/app-icon.png"


def main() -> None:
    img = Image.new("RGBA", (_SIZE, _SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    draw.rounded_rectangle([(0, 0), (_SIZE, _SIZE)], radius=_CORNER_RADIUS, fill=_BACKGROUND)

    d = int(_SIZE * _CIRCLE_DIAMETER_RATIO)
    offset = (_SIZE - d) // 2
    draw.ellipse([(offset, offset), (offset + d, offset + d)], fill=_FOREGROUND)

    _OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(_OUTPUT)
    print(f"Wrote {_OUTPUT} ({_SIZE}x{_SIZE})")


if __name__ == "__main__":
    main()
