"""Regenerate the README button images in buttons/.

The labels are baked into the SVGs as outlines rather than text, and the icons are
inlined from icons/, so the buttons look the same whatever fonts the reader has and
fetch nothing at render time. Rerun this after changing LABELS.

icons/ holds one file per button, matching the glyphs the docs pills use. The Font
Awesome Free ones are CC BY 4.0, the Hugging Face and Weights & Biases marks come from
simple-icons and are CC0.
"""

import re
from pathlib import Path

from matplotlib.font_manager import FontProperties
from matplotlib.path import Path as MPath
from matplotlib.textpath import TextPath

# IBL Core palette, mirroring the --bwb-* variables in custom.css. One blue throughout,
# so the filled Leaderboard and the outlines read as the same colour. Rerun this file
# after changing either of them.
CYAN, MAGENTA = "#009FD7", "#CE2C97"

FONT = FontProperties(family="DejaVu Sans", weight="bold")
# a little roomier than the .bwb-pill rule in custom.css, which sits at 36 by 18
FONT_SIZE, HEIGHT, PAD = 14.4, 40, 22
# baked into each side so the row breathes like the pills, which use a 0.6rem gap
GAP = 3
# the icon box and the space after it, matching the pills' 1em icon and 0.5rem gap
ICON, ICON_GAP = 15.0, 8.0

LABELS = [
    "Homepage",
    "Leaderboard",
    "Paper",
    "Documentation",
    "Checkpoints",
    "Runs",
    "Cite",
]

# the two calls to action, filled in the logo colours; everything else is outlined
FILLED = {"Homepage": MAGENTA, "Leaderboard": CYAN}


def icon(slug):
    """Return one icon as (path data, viewBox width, viewBox height)."""
    svg = (Path(__file__).parent / "icons" / f"{slug}.svg").read_text()
    _, _, vb_w, vb_h = re.search(r'viewBox="([^"]+)"', svg).group(1).split()
    d = " ".join(re.findall(r'<path[^>]+\bd="([^"]+)"', svg))
    return d, float(vb_w), float(vb_h)


def outline(label):
    """Return the label as SVG path data, plus its width and vertical extent."""
    path = TextPath((0, 0), label, size=FONT_SIZE, prop=FONT)
    parts = []
    for verts, code in path.iter_segments():
        p = [f"{v:.2f}" for v in verts]
        if code == MPath.MOVETO:
            parts.append(f"M{p[0]} {-float(p[1]):.2f}")
        elif code == MPath.LINETO:
            parts.append(f"L{p[0]} {-float(p[1]):.2f}")
        elif code == MPath.CURVE3:
            parts.append(f"Q{p[0]} {-float(p[1]):.2f} {p[2]} {-float(p[3]):.2f}")
        elif code == MPath.CURVE4:
            parts.append(
                f"C{p[0]} {-float(p[1]):.2f} {p[2]} {-float(p[3]):.2f} {p[4]} {-float(p[5]):.2f}"
            )
        elif code == MPath.CLOSEPOLY:
            parts.append("Z")
    box = path.get_extents()
    return " ".join(parts), box.width, (box.y0 + box.y1) / 2


def pill(label, slug, fill):
    """Render one button, filled in `fill` or outlined when `fill` is None."""
    d, width, mid = outline(label)
    icon_d, vb_w, vb_h = icon(slug)
    scale = ICON / vb_h
    icon_w = vb_w * scale

    pill_w = icon_w + ICON_GAP + width + 2 * PAD
    w = round(pill_w + 2 * GAP)
    ink = "#FFFFFF" if fill else CYAN
    edge = "" if fill else f' stroke="{CYAN}" stroke-width="1.5"'

    icon_x = GAP + PAD
    text_x = icon_x + icon_w + ICON_GAP
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{HEIGHT}" '
        f'viewBox="0 0 {w} {HEIGHT}" role="img" aria-label="{label}">'
        f'<rect x="{GAP + 0.75}" y="0.75" width="{pill_w - 1.5:.2f}" height="{HEIGHT - 1.5}" '
        f'rx="{(HEIGHT - 1.5) / 2}" fill="{fill or "none"}"{edge}/>'
        f'<g fill="{ink}">'
        f'<g transform="translate({icon_x:.2f} {(HEIGHT - ICON) / 2:.2f}) scale({scale:.5f})">'
        f'<path d="{icon_d}"/></g>'
        f'<g transform="translate({text_x:.2f} {HEIGHT / 2 + mid:.2f})"><path d="{d}"/></g>'
        f"</g></svg>"
    )


def main():
    out = Path(__file__).parent / "buttons"
    out.mkdir(exist_ok=True)
    for label in LABELS:
        slug = label.lower().replace(" ", "-")
        (out / f"{slug}.svg").write_text(pill(label, slug, FILLED.get(label)))
    print(f"wrote {len(LABELS)} buttons to {out}")


if __name__ == "__main__":
    main()
