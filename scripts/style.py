"""One visual style for the Ling-3.0-tiny card figures. Tokens from docs/design-style.md.

    from style import theme, apply, gradient_rect
    T = theme("light")   # or "dark": hex colours by role
    apply(T)             # matplotlib rcParams (fonts, opaque background, axes)

Colour roles: ours = main blue, bloomer = slate (#7f93c9, hollow circle), bart = muted (hollow square),
kda = pale sky (always drawn with the router-border outline), mla = inclusionAI gradient, dense = slate,
warm = PASS/FAIL and the "codes refined" highlight only. Shape, fill and direct labels repeat every colour.

Fonts: Red Hat Display 700 (titles) and Red Hat Text 400/500 (body) from Google Fonts in build/fonts/
(git-ignored; fetch with the curl lines in the README). Without them the module falls back to DejaVu Sans
and prints a note. SVGs keep text as paths, so the files do not need the fonts to display.
"""
from pathlib import Path

import numpy as np

FONT_DIR = Path(__file__).resolve().parent.parent / "build" / "fonts"

BAND = ["#081a52", "#16257a", "#33237e"]          # 0%, 55%, 100% at 135 degrees
BAND_POS = [0.0, 0.55, 1.0]
ACCENT = ["#2ec5ff", "#4a63ff", "#8b5cf6"]        # inclusionAI gradient, same stops
WARM = ["#f6d77a", "#ef9a6a"]

COMMON = {
    "band_meta": "#c9d3ef", "band_link": "#8fe0ff", "band_ink": "#ffffff",
    "kda": "#cdeeff", "dense": "#7f93c9", "border": "#8fa3d6",
    "mla": ACCENT[1], "bloomer": "#7f93c9", "warm0": WARM[0], "warm1": WARM[1],
    "deep": "#3346e0", "oxford": "#0b1d5c",
}
THEMES = {
    "light": {**COMMON, "bg": "#ffffff", "ink": "#0b1020", "mute": "#4a5470", "rule": "#d9deea",
              "panel": "#f1f4ff", "ours": "#4a63ff", "bart": "#4a5470", "warm": "#c4521f"},
    "dark": {**COMMON, "bg": "#05132e", "ink": "#e8ecf7", "mute": "#a5afc9", "rule": "#22305a",
             "panel": "#0e1d55", "ours": "#9db4ff", "bart": "#a5afc9", "warm": "#f6a56e"},
}
TITLE = "Red Hat Display"
BODY = "Red Hat Text"


def theme(name):
    t = dict(THEMES[name])
    t["name"] = name
    return t


def _fonts():
    from matplotlib import font_manager as fm
    found = sorted(FONT_DIR.glob("*.ttf")) if FONT_DIR.is_dir() else []
    for f in found:
        fm.fontManager.addfont(str(f))
    names = {e.name for e in fm.fontManager.ttflist}
    if TITLE in names and BODY in names:
        return TITLE, BODY
    print(f"style: Red Hat fonts not in {FONT_DIR}; falling back to DejaVu Sans")
    return "DejaVu Sans", "DejaVu Sans"


def apply(T):
    import matplotlib as mpl
    title, body = _fonts()
    T["title_font"], T["body_font"] = title, body
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [body, "DejaVu Sans"], "font.size": 10,
        "axes.edgecolor": T["mute"], "axes.labelcolor": T["ink"], "axes.titlecolor": T["ink"],
        "axes.titlesize": 11, "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": T["rule"], "grid.linewidth": 0.6, "axes.axisbelow": True,
        "xtick.color": T["mute"], "ytick.color": T["mute"], "xtick.labelcolor": T["mute"],
        "ytick.labelcolor": T["mute"], "text.color": T["ink"],
        "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 200,
        "figure.facecolor": T["bg"], "axes.facecolor": T["bg"], "savefig.facecolor": T["bg"],
        "savefig.transparent": False, "svg.fonttype": "path",
    })


def title_kw(T, size=12):
    return {"fontfamily": T["title_font"], "fontweight": "bold", "fontsize": size, "color": T["ink"]}


def ramp(stops, pos, n=256):
    """n x 3 RGB array interpolated through hex stops at positions pos (0..1)."""
    from matplotlib.colors import to_rgb
    rgb = np.array([to_rgb(s) for s in stops])
    x = np.linspace(0, 1, n)
    return np.stack([np.interp(x, pos, rgb[:, k]) for k in range(3)], 1)


def gradient_image(stops, pos, w=256, h=256, angle=135):
    """h x w x 3 image of a CSS-style linear gradient (135 deg = top left to bottom right)."""
    a = np.deg2rad(angle)
    yy, xx = np.mgrid[0:1:h * 1j, 0:1:w * 1j]
    t = xx * np.sin(a) - yy * np.cos(a)   # CSS angles: 0 = to top, 90 = to right; rows run downwards
    t = (t - t.min()) / (t.max() - t.min())
    r = ramp(stops, pos)
    return r[(t * 255).astype(int)]


def gradient_rect(ax, x, y, w, h, stops=ACCENT, pos=BAND_POS, angle=90, edge=None, lw=0.8, z=2):
    """Rectangle filled with a gradient (data coordinates), optional outline."""
    from matplotlib.patches import Rectangle
    img = gradient_image(stops, pos, 64, 64, angle)
    im = ax.imshow(img, extent=(x, x + w, y, y + h), origin="upper", aspect="auto", zorder=z, interpolation="bilinear")
    clip = Rectangle((x, y), w, h, transform=ax.transData, facecolor="none", edgecolor=edge or "none", lw=lw, zorder=z + 0.1)
    ax.add_patch(clip)
    im.set_clip_path(clip)
    return im
