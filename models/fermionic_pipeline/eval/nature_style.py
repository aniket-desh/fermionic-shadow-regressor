"""Shared Matplotlib style for publication figures (Nature-optimised, LaTeX).

Importing a plotting module applies this style, so every figure renders text
through LaTeX with sizes tuned for a Nature paper. The defaults follow Nature's
figure guide:

  - Sans-serif type (Helvetica/Arial) rendered via LaTeX (``helvet`` + ``sansmath``
    so math is sans-serif too). Set ``USE_SANS = False`` for serif Computer Modern.
  - Text 5-7 pt at final print size: base 7 pt, axis labels 7, tick labels 6,
    legend 6, panel titles 8, figure titles 9.
  - Column widths: single 89 mm, 1.5-column 120 mm, double 183 mm.
  - Thin rules/lines, fonts embedded as editable TrueType (``pdf.fonttype = 42``),
    600 dpi raster fallback.

Usage in a plotting module::

    from fermionic_pipeline.eval.nature_style import (
        apply_nature_style, tex_escape, DOUBLE_COL, grid_figsize,
    )
    apply_nature_style()

If LaTeX is unavailable in the environment, call ``apply_nature_style(usetex=False)``
to keep the Nature sizing/spacing but fall back to Matplotlib's mathtext renderer.
"""

from __future__ import annotations

# ── Column widths in inches (Nature: 89 / 120 / 183 mm) ──────────────────────
_MM = 1.0 / 25.4
SINGLE_COL = 89 * _MM       # 3.50 in
ONEHALF_COL = 120 * _MM     # 4.72 in
DOUBLE_COL = 183 * _MM      # 7.20 in

# Flip to False for serif Computer Modern math/text instead of Nature sans-serif.
USE_SANS = True

# Colour-vision-deficiency (colourblind) safe mode. When True, the categorical
# line palette becomes the Okabe-Ito set and colour maps avoid red-green
# (e.g. the quality heatmap's RdYlGn -> cividis). Toggle per call with
# ``apply_nature_style(colorblind=True)`` or globally by flipping this default.
COLORBLIND = False

# Okabe-Ito qualitative palette — eight colours distinguishable under the common
# CVD types and in greyscale. Ordered for good contrast between the first few.
OKABE_ITO = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]

# Default (non-CVD) categorical palette: Matplotlib's tab10.
_DEFAULT_CYCLE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

# Colour maps. Normal maps are the default; the colourblind-safe variants are
# returned only when a caller explicitly asks (``colorblind=True``).
_SEQUENTIAL = "viridis"
_DIVERGING = "RdBu_r"        # ColorBrewer CVD-safe — same in both modes
_QUALITY = "RdYlGn"          # red-green: NOT CVD-safe
_QUALITY_CB = "cividis"      # CVD-safe replacement for 0->1 quality fields
_SEQUENTIAL_CB = "cividis"

# Font sizes (pt) at final print size.
FONT_BASE = 7
FONT_LABEL = 7
FONT_TICK = 6
FONT_LEGEND = 6
FONT_TITLE = 8
FONT_SUPTITLE = 9

_SANS_PREAMBLE = (
    r"\usepackage[T1]{fontenc}"
    r"\usepackage{helvet}"
    r"\renewcommand{\familydefault}{\sfdefault}"
    r"\usepackage{sansmath}"
    r"\sansmath"
    r"\usepackage{amsmath}"
    r"\usepackage{siunitx}"
)
_SERIF_PREAMBLE = (
    r"\usepackage[T1]{fontenc}"
    r"\usepackage{amsmath}"
    r"\usepackage{siunitx}"
)


def apply_nature_style(usetex: bool = True, sans: bool | None = None,
                       colorblind: bool | None = None) -> None:
    """Set Matplotlib rcParams for Nature-quality, LaTeX-rendered figures.

    Args:
        usetex: render text with LaTeX. Requires ``latex``, ``dvipng`` and
            ``ghostscript`` on PATH. Set False to fall back to mathtext.
        sans: sans-serif (Nature default) when True, serif Computer Modern when
            False. Defaults to the module-level ``USE_SANS``.
        colorblind: use the colour-vision-deficiency-safe palette (Okabe-Ito)
            and avoid red-green colour maps. Defaults to ``COLORBLIND``. The
            choice is recorded so ``sequential_cmap()`` / ``diverging_cmap()`` /
            ``quality_cmap()`` resolve to safe maps too.
    """
    import matplotlib as mpl
    from cycler import cycler

    sans = USE_SANS if sans is None else sans
    colorblind = COLORBLIND if colorblind is None else colorblind

    rc = {
        # --- LaTeX text rendering ---
        "text.usetex": usetex,
        "font.family": "sans-serif" if sans else "serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.serif": ["Computer Modern Roman", "Times", "DejaVu Serif"],
        # --- font sizes (pt) ---
        "font.size": FONT_BASE,
        "axes.titlesize": FONT_TITLE,
        "axes.labelsize": FONT_LABEL,
        "xtick.labelsize": FONT_TICK,
        "ytick.labelsize": FONT_TICK,
        "legend.fontsize": FONT_LEGEND,
        "figure.titlesize": FONT_SUPTITLE,
        # --- thin rules and lines ---
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.0,
        "patch.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        # --- legend: frameless, tight (Nature house style) ---
        "legend.frameon": False,
        "legend.handlelength": 1.4,
        "legend.columnspacing": 1.0,
        "legend.labelspacing": 0.3,
        # --- spines: drop top/right ---
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlepad": 4.0,
        "axes.labelpad": 2.0,
        # --- output: embed editable fonts, high-res raster fallback ---
        "figure.dpi": 200,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # --- categorical colour cycle ---
        "axes.prop_cycle": cycler(color=OKABE_ITO if colorblind else _DEFAULT_CYCLE),
    }
    if usetex:
        rc["text.latex.preamble"] = _SANS_PREAMBLE if sans else _SERIF_PREAMBLE

    mpl.rcParams.update(rc)


# ── Colour-scheme helpers ────────────────────────────────────────────────────
# Normal maps/palette are the default; pass ``colorblind=True`` to a helper to
# specifically request the colour-vision-deficiency-safe variant.

def line_colors(colorblind: bool = False) -> list:
    """Categorical line/marker palette. Okabe-Ito when ``colorblind=True``."""
    return list(OKABE_ITO if colorblind else _DEFAULT_CYCLE)


def sequential_cmap(colorblind: bool = False) -> str:
    """Sequential colour map: ``viridis`` by default, ``cividis`` if colourblind."""
    return _SEQUENTIAL_CB if colorblind else _SEQUENTIAL


def diverging_cmap(colorblind: bool = False) -> str:
    """Diverging colour map. ``RdBu_r`` is ColorBrewer CVD-safe, so used in both."""
    return _DIVERGING


def quality_cmap(colorblind: bool = False) -> str:
    """Map for 0->1 "quality" fields: ``RdYlGn`` by default (red-green, NOT
    CVD-safe); ``cividis`` when ``colorblind=True``."""
    return _QUALITY_CB if colorblind else _QUALITY


def grid_figsize(ncols: int, nrows: int, aspect: float = 0.78,
                 width: float = DOUBLE_COL) -> tuple[float, float]:
    """Figure size (inches) for an ``ncols x nrows`` panel grid at ``width``.

    Each panel is ``width / ncols`` wide and ``aspect`` times as tall, so the
    full grid fits the target column width with consistent panel proportions.
    """
    panel_w = width / max(ncols, 1)
    return (width, panel_w * aspect * max(nrows, 1))


_TEX_SPECIAL = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}


def tex_escape(s) -> str:
    """Escape LaTeX-special characters in a dynamic string (e.g. data labels).

    Use for any text interpolated into a label from data — observable keys,
    molecule names, file tags — so a stray ``_`` or ``%`` cannot break a
    ``usetex`` render.
    """
    s = str(s)
    return "".join(_TEX_SPECIAL.get(ch, ch) for ch in s)
