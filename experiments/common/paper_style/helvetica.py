"""Render matplotlib text in Helvetica using renamed copies of URW Nimbus Sans; raises if they are missing.

Needs fonts-urw-base35 and fontTools.
"""
from __future__ import annotations

import contextlib
import os

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.path.join(HERE, "fonts")          # created on first use, not tracked
STYLE = os.path.join(HERE, "paper.mplstyle")

SOURCES = {
    "Regular":    ("NimbusSans-Regular.otf",    "Helvetica",             "Helvetica"),
    "Bold":       ("NimbusSans-Bold.otf",       "Helvetica Bold",        "Helvetica-Bold"),
    "Italic":     ("NimbusSans-Italic.otf",     "Helvetica Oblique",     "Helvetica-Oblique"),
    "BoldItalic": ("NimbusSans-BoldItalic.otf", "Helvetica BoldOblique", "Helvetica-BoldOblique"),
}
SEARCH = ("/usr/share/fonts/opentype/urw-base35", "/usr/share/fonts/type1/urw-base35")


def _find(basename: str) -> str | None:
    for d in SEARCH:
        p = os.path.join(d, basename)
        if os.path.exists(p):
            return p
    hits = [f for f in fm.findSystemFonts() if os.path.basename(f) == basename]
    return hits[0] if hits else None


def build(dest: str = FONT_DIR, force: bool = False) -> list[str]:
    """Write Helvetica-named copies of the Nimbus Sans faces into `dest`. Returns the paths."""
    from fontTools.ttLib import TTFont

    os.makedirs(dest, exist_ok=True)
    out: list[str] = []
    for face, (basename, full, ps) in SOURCES.items():
        target = os.path.join(dest, f"Helvetica-{face}.otf")
        if os.path.exists(target) and not force:
            out.append(target)
            continue
        src = _find(basename)
        if src is None:                        # only the regular face is required
            if face == "Regular":
                raise RuntimeError(
                    f"{basename} not found. Install the Helvetica-metric clone with:\n"
                    "    apt-get install -y fonts-urw-base35")
            continue
        f = TTFont(src)
        sub = "Helvetica" if face == "Regular" else full[len("Helvetica "):]
        for rec in f["name"].names:
            nid = rec.nameID
            if nid == 1:                       # family
                rec.string = "Helvetica"
            elif nid == 2:                     # subfamily
                rec.string = "Regular" if face == "Regular" else sub
            elif nid == 4:                     # full name
                rec.string = full
            elif nid == 6:                     # postscript name
                rec.string = ps
            elif nid in (16, 17):              # typographic family / subfamily
                rec.string = "Helvetica" if nid == 16 else ("Regular" if face == "Regular" else sub)
        f.save(target)
        f.close()
        out.append(target)
    return out


def register(dest: str = FONT_DIR) -> str:
    """Build the fonts if needed, register every face, and return the Helvetica file path."""
    for path in build(dest):
        fm.fontManager.addfont(path)
    # also register the original Nimbus and Liberation fonts as fallbacks
    for f in fm.findSystemFonts():
        if "Nimbus" in f or "Liberation" in f:
            try:
                fm.fontManager.addfont(f)
            except Exception:
                pass
    resolved = fm.findfont(fm.FontProperties(family=["Helvetica"]), fallback_to_default=False)
    if "Helvetica" not in os.path.basename(resolved):
        raise RuntimeError(f"Helvetica did not resolve; got {resolved}")
    return resolved


def use(style: str | None = STYLE, dest: str = FONT_DIR) -> str:
    """Register Helvetica, apply the paper style and set the font rcParams. Returns the font path."""
    path = register(dest)
    if style:
        plt.style.use(style)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Nimbus Sans", "Liberation Sans", "DejaVu Sans"]
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "Helvetica"
    plt.rcParams["mathtext.it"] = "Helvetica:italic"
    plt.rcParams["mathtext.bf"] = "Helvetica:bold"
    plt.rcParams["mathtext.sf"] = "Helvetica"
    plt.rcParams["mathtext.cal"] = "Helvetica"   # avoids matplotlib looking for a cursive font
    plt.rcParams["mathtext.default"] = "regular"
    return path


PINS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Helvetica",
    "mathtext.it": "Helvetica:italic",
    "mathtext.bf": "Helvetica:bold",
    "mathtext.sf": "Helvetica",
    "mathtext.cal": "Helvetica",
    "mathtext.default": "regular",
}


@contextlib.contextmanager
def context(style: str | None = STYLE, dest: str = FONT_DIR):
    """Context-manager version of `use()`; the settings apply only inside the block."""
    register(dest)
    with plt.style.context(style if style else {}):
        with plt.rc_context(PINS):
            yield


if __name__ == "__main__":
    p = register()
    print("Helvetica resolves to:", p)
    print("family name:", fm.FontProperties(fname=p).get_name())
    print("built:", "\n  ".join([""] + build()))
