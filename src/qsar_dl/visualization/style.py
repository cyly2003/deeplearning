r"""Publication style helpers for QSAR evaluation figures.

This module provides reusable matplotlib style helpers for publication-ready
QSAR / ecotoxicology figures.

Main fixes compared with the previous version
---------------------------------------------
1. English fonts are placed before Chinese fonts to avoid strange English glyphs.
2. Math text uses a custom Arial-like font setting to reduce strange superscripts.
3. Tick labels, legends, annotations and colorbar text can be forced to bold.
4. The original public interfaces are preserved as much as possible.
5. SVG export can choose editable text or stable path rendering.

Typical usage
-------------
    import matplotlib.pyplot as plt
    from style_fixed import set_publication_style, save_figure

    set_publication_style(language="zh", font_weight="bold")

    fig, ax = plt.subplots()
    ax.set_xlabel(r"$\mathrm{EC_{50}}$ (mg/L)")
    ax.set_ylabel(r"$\mathrm{R^2}$")
    ax.plot([1, 2, 3], [1, 4, 9], label="Model A")
    ax.legend()

    save_figure(fig, "outputs/model_performance")
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

DEFAULT_DPI = 300
DEFAULT_FORMATS = ("png", "tiff", "pdf", "svg")

_PALETTES: dict[str, tuple[str, ...]] = {
    "journal": (
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#CC79A7",
        "#E69F00",
        "#56B4E9",
        "#F0E442",
        "#000000",
    ),
    "colorblind": (
        "#0072B2",
        "#E69F00",
        "#009E73",
        "#D55E00",
        "#CC79A7",
        "#56B4E9",
        "#F0E442",
        "#000000",
    ),
    "endpoint": (
        "#1B9E77",
        "#D95F02",
        "#7570B3",
        "#E7298A",
        "#66A61E",
        "#E6AB02",
    ),
    "chemical_category": (
        "#1F77B4",
        "#FF7F0E",
        "#2CA02C",
        "#D62728",
        "#9467BD",
        "#8C564B",
        "#E377C2",
        "#7F7F7F",
        "#BCBD22",
        "#17BECF",
        "#4C78A8",
        "#F58518",
        "#54A24B",
    ),
    "diverging": (
        "#2166AC",
        "#4393C3",
        "#92C5DE",
        "#F7F7F7",
        "#F4A582",
        "#D6604D",
        "#B2182B",
    ),
}

_ALIASES = {
    "default": "journal",
    "est": "journal",
    "es&t": "journal",
    "jhm": "journal",
    "category": "chemical_category",
    "categories": "chemical_category",
}


def _require_matplotlib() -> Any:
    """Import matplotlib with a clear error message."""

    try:
        import matplotlib as mpl
    except ImportError as exc:
        message = (
            "matplotlib is required for QSAR visualization plotting helpers. "
            "Install it with: pip install matplotlib"
        )
        raise ImportError(message) from exc
    return mpl


def _require_pyplot() -> Any:
    """Import matplotlib.pyplot with a clear error message."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        message = (
            "matplotlib.pyplot is required to close figures in save_figure. "
            "Install it with: pip install matplotlib"
        )
        raise ImportError(message) from exc
    return plt


def _normalize_palette_name(name: str) -> str:
    """Normalize palette aliases."""

    key = name.strip().lower()
    return _ALIASES.get(key, key)


def _cycle_to_length(colors: Sequence[str], n_colors: int) -> list[str]:
    """Cycle a color palette to a required length."""

    if n_colors < 1:
        raise ValueError("n_colors must be at least 1.")
    return [colors[index % len(colors)] for index in range(n_colors)]


def get_palette(
    name: str | Sequence[str] = "journal",
    n_colors: int | None = None,
    *,
    as_cmap: bool = False,
    cmap_name: str | None = None,
) -> list[str] | Any:
    """Return a publication-friendly color palette.

    Parameters
    ----------
    name:
        Built-in palette name or an explicit sequence of color strings.
    n_colors:
        Optional number of colors to return. Palettes are cycled when more
        colors are requested than the palette contains.
    as_cmap:
        If true, return a matplotlib ``ListedColormap``.
    cmap_name:
        Optional name for the returned colormap.

    Returns
    -------
    list[str] or Any
        Selected colors or a matplotlib ListedColormap.
    """

    if isinstance(name, str):
        key = _normalize_palette_name(name)
        if key not in _PALETTES:
            available = ", ".join(sorted(_PALETTES))
            raise ValueError(f"Unknown palette {name!r}. Available palettes: {available}.")
        colors: Sequence[str] = _PALETTES[key]
    else:
        colors = list(name)
        if not colors:
            raise ValueError("Custom palettes must contain at least one color.")
        key = "custom"

    selected = list(colors) if n_colors is None else _cycle_to_length(colors, n_colors)

    if not as_cmap:
        return selected

    mpl = _require_matplotlib()
    return mpl.colors.ListedColormap(selected, name=cmap_name or key)


def get_font_stack(language: str = "auto") -> list[str]:
    """Return a robust sans-serif font fallback stack.

    Important
    ---------
    English fonts are placed before Chinese fonts. This avoids English letters,
    numbers, superscripts and subscripts being rendered by Chinese fonts such as
    SimHei or SimSun.

    Parameters
    ----------
    language:
        One of "auto", "zh", "cn", or "en".

    Returns
    -------
    list[str]
        Ordered font fallback list.
    """

    language_key = language.lower()
    if language_key not in {"auto", "zh", "cn", "en"}:
        raise ValueError("language must be one of: auto, zh, cn, en.")

    if language_key == "en":
        return [
            "Arial",
            "Helvetica",
            "Liberation Sans",
            "DejaVu Sans",
        ]

    if language_key in {"zh", "cn"}:
        return [
            # Chinese-first fonts are required when figures contain Chinese
            # titles or legends; otherwise matplotlib may bind the whole text
            # artist to Arial and emit missing-glyph warnings.
            "Microsoft YaHei",
            "SimHei",
            "SimSun",

            # English fallbacks.
            "Arial",
            "Helvetica",
            "Liberation Sans",

            # Open-source Chinese fallbacks.
            "Noto Sans CJK SC",
            "Source Han Sans SC",
            "WenQuanYi Micro Hei",

            # Last-resort fallback.
            "DejaVu Sans",
        ]

    return [
        # English-first fonts.
        "Arial",
        "Helvetica",
        "Liberation Sans",

        # Common Windows Chinese fonts.
        "Microsoft YaHei",
        "Microsoft YaHei UI",
        "SimHei",

        # Open-source Chinese fallbacks.
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Micro Hei",

        # Last-resort fallback.
        "DejaVu Sans",
    ]


def set_publication_style(
    *,
    language: str = "auto",
    title_size: int = 20,
    label_size: int = 18,
    tick_size: int = 18,
    text_size: int = 18,
    legend_size: int = 16,
    font_weight: str = "bold",
    axis_linewidth: float = 1.5,
    dpi: int = DEFAULT_DPI,
    palette: str | Sequence[str] = "journal",
    svg_text_as_path: bool = False,
    rc: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the project publication style to matplotlib rcParams.

    Compatibility note
    ------------------
    The original arguments are preserved. The new argument ``svg_text_as_path``
    is optional and does not affect old calls.

    Parameters
    ----------
    language:
        "auto", "zh", "cn", or "en".
    title_size:
        Title font size.
    label_size:
        X/Y axis label font size.
    tick_size:
        Tick label font size.
    text_size:
        General text font size.
    legend_size:
        Legend font size.
    font_weight:
        Font weight, usually "bold" or "normal".
    axis_linewidth:
        Axis spine and tick width.
    dpi:
        Figure and saved image DPI. Must be at least 300.
    palette:
        Built-in palette name or custom color sequence.
    svg_text_as_path:
        False keeps SVG text editable.
        True converts SVG text to paths, making appearance more stable.
    rc:
        Optional additional rcParams to override defaults.

    Returns
    -------
    dict[str, Any]
        The rcParams dictionary applied to matplotlib.
    """

    if dpi < DEFAULT_DPI:
        raise ValueError(f"dpi must be at least {DEFAULT_DPI}.")
    if axis_linewidth <= 0:
        raise ValueError("axis_linewidth must be positive.")

    mpl = _require_matplotlib()

    sans_fonts = get_font_stack(language)
    colors = get_palette(palette)

    style_rc: dict[str, Any] = {
        # Main font settings.
        "font.family": "sans-serif",
        "font.sans-serif": sans_fonts,
        "font.size": text_size,
        "font.weight": font_weight,

        # Axis title and label settings.
        "axes.titlesize": title_size,
        "axes.titleweight": font_weight,
        "axes.labelsize": label_size,
        "axes.labelweight": font_weight,
        "axes.linewidth": axis_linewidth,
        "axes.unicode_minus": False,
        "axes.prop_cycle": mpl.cycler(color=colors),

        # Tick settings.
        "xtick.labelsize": tick_size,
        "ytick.labelsize": tick_size,
        "xtick.major.width": axis_linewidth,
        "ytick.major.width": axis_linewidth,
        "xtick.minor.width": max(axis_linewidth * 0.8, 0.8),
        "ytick.minor.width": max(axis_linewidth * 0.8, 0.8),

        # Legend settings.
        "legend.fontsize": legend_size,
        "legend.frameon": False,

        # Math text settings.
        # This fixes many cases where R², EC50, LC50, p-values, units and
        # superscripts/subscripts look inconsistent with normal English text.
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "mathtext.default": "regular",

        # Figure export settings.
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",

        # Keep TrueType fonts in vector output.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        # SVG behavior:
        # "none" keeps text editable but depends on local fonts.
        # "path" keeps appearance stable but makes text non-editable.
        "svg.fonttype": "path" if svg_text_as_path else "none",
    }

    if rc:
        style_rc.update(dict(rc))

    mpl.rcParams.update(style_rc)
    return style_rc


def apply_figure_text_weight(fig: Any, font_weight: str | None = None) -> Any:
    """Apply the active font weight to every visible text artist in a figure.

    Parameters
    ----------
    fig:
        Matplotlib figure.
    font_weight:
        Weight to apply. If omitted, uses ``matplotlib.rcParams["font.weight"]``.

    Returns
    -------
    Any
        The same figure, modified in place.
    """

    mpl = _require_matplotlib()
    weight = font_weight or str(mpl.rcParams.get("font.weight", "normal"))

    if not weight or weight == "normal":
        return fig

    # General text artists: titles, labels, annotations, colorbar labels, etc.
    for text in fig.findobj(match=mpl.text.Text):
        text.set_fontweight(weight)

    # Explicitly cover tick labels, axis labels and legends.
    for ax in fig.axes:
        ax.xaxis.label.set_fontweight(weight)
        ax.yaxis.label.set_fontweight(weight)
        ax.title.set_fontweight(weight)

        for label in ax.get_xticklabels(which="both"):
            label.set_fontweight(weight)

        for label in ax.get_yticklabels(which="both"):
            label.set_fontweight(weight)

        legend = ax.get_legend()
        if legend is not None:
            for legend_text in legend.get_texts():
                legend_text.set_fontweight(weight)
            legend_title = legend.get_title()
            if legend_title is not None:
                legend_title.set_fontweight(weight)

    return fig


def apply_figure_font_family(fig: Any, font_family: str | None = None) -> Any:
    """Apply one font family to every visible text artist in a figure.

    This function is optional. It is useful when some text artists were created
    before ``set_publication_style`` was called.

    Parameters
    ----------
    fig:
        Matplotlib figure.
    font_family:
        Font family to apply. If omitted, uses the first font in
        ``matplotlib.rcParams["font.sans-serif"]``.

    Returns
    -------
    Any
        The same figure, modified in place.
    """

    mpl = _require_matplotlib()

    if font_family is None:
        sans_fonts = mpl.rcParams.get("font.sans-serif", ["Arial"])
        font_family = sans_fonts[0] if sans_fonts else "Arial"

    for text in fig.findobj(match=mpl.text.Text):
        text.set_fontfamily(font_family)

    return fig


def standardize_axes_text(
    ax: Any,
    *,
    font_weight: str | None = None,
    font_family: str | None = None,
) -> Any:
    """Standardize text properties for a single axis.

    Parameters
    ----------
    ax:
        Matplotlib axes.
    font_weight:
        Optional font weight.
    font_family:
        Optional font family.

    Returns
    -------
    Any
        The same axis, modified in place.
    """

    texts = [
        ax.title,
        ax.xaxis.label,
        ax.yaxis.label,
        *ax.get_xticklabels(which="both"),
        *ax.get_yticklabels(which="both"),
    ]

    legend = ax.get_legend()
    if legend is not None:
        texts.extend(legend.get_texts())
        texts.append(legend.get_title())

    for text in texts:
        if text is None:
            continue
        if font_weight:
            text.set_fontweight(font_weight)
        if font_family:
            text.set_fontfamily(font_family)

    return ax


def _normalize_formats(
    output_path: Path,
    formats: Iterable[str] | None,
) -> tuple[Path, tuple[str, ...]]:
    """Normalize output stem and export formats."""

    if formats is None:
        if not output_path.suffix:
            raise ValueError("formats=None requires output_path to include a file suffix.")
        return output_path.with_suffix(""), (output_path.suffix.lstrip(".").lower(),)

    normalized = tuple(fmt.lower().lstrip(".") for fmt in formats)
    if not normalized:
        raise ValueError("formats must contain at least one file format.")

    return output_path.with_suffix("") if output_path.suffix else output_path, normalized


def save_figure(
    fig: Any,
    output_path: str | Path,
    *,
    formats: Iterable[str] | None = DEFAULT_FORMATS,
    dpi: int = DEFAULT_DPI,
    close: bool = False,
    force_bold: bool = True,
    font_weight: str | None = None,
    **savefig_kwargs: Any,
) -> list[Path]:
    """Save a matplotlib figure to one or more publication formats.

    Compatibility note
    ------------------
    The original arguments are preserved:
    ``fig``, ``output_path``, ``formats``, ``dpi``, ``close`` and
    ``**savefig_kwargs`` all continue to work.

    New optional arguments
    ----------------------
    force_bold:
        If True, applies ``apply_figure_text_weight`` before saving.
    font_weight:
        Optional font weight used when ``force_bold=True``.

    Parameters
    ----------
    fig:
        Matplotlib figure.
    output_path:
        Output path. If ``formats`` is not None, this is treated as the output
        stem. For example, ``"figures/model"`` creates ``model.png``,
        ``model.tiff``, ``model.pdf`` and ``model.svg``.
    formats:
        Iterable of formats. Use ``formats=None`` to save only the suffix
        already included in ``output_path``.
    dpi:
        Export DPI. Must be at least 300.
    close:
        If True, close the figure after saving.
    force_bold:
        If True, force all visible text artists to use the selected weight.
    font_weight:
        Font weight used when ``force_bold=True``.
    **savefig_kwargs:
        Extra keyword arguments passed to ``fig.savefig``.

    Returns
    -------
    list[pathlib.Path]
        Paths of saved figures.
    """

    if dpi < DEFAULT_DPI:
        raise ValueError(f"dpi must be at least {DEFAULT_DPI}.")

    path = Path(output_path)
    stem, normalized_formats = _normalize_formats(path, formats)
    stem.parent.mkdir(parents=True, exist_ok=True)

    default_kwargs = {
        "dpi": dpi,
        "bbox_inches": "tight",
        "facecolor": "white",
    }
    default_kwargs.update(savefig_kwargs)

    if force_bold:
        apply_figure_text_weight(fig, font_weight=font_weight)

    saved_paths: list[Path] = []
    for fmt in normalized_formats:
        figure_path = stem.with_suffix(f".{fmt}")
        fig.savefig(figure_path, format=fmt, **default_kwargs)
        saved_paths.append(figure_path)

    if close:
        plt = _require_pyplot()
        plt.close(fig)

    return saved_paths


def recommended_math_label(text: str) -> str:
    """Return a simple helper for roman math text labels.

    This is a small convenience helper for common labels such as "R^2",
    "EC_{50}", or "LC_{50}". It wraps the expression with ``\\mathrm{}``.

    Examples
    --------
    >>> recommended_math_label("R^2")
    '$\\\\mathrm{R^2}$'
    >>> recommended_math_label("EC_{50}")
    '$\\\\mathrm{EC_{50}}$'

    Notes
    -----
    This helper is optional. You can always write labels manually, for example:
    ``ax.set_ylabel(r"$\\mathrm{R^2}$")``.
    """

    clean = text.strip()
    if not clean:
        raise ValueError("text must not be empty.")
    return rf"$\mathrm{{{clean}}}$"


__all__ = [
    "DEFAULT_DPI",
    "DEFAULT_FORMATS",
    "get_palette",
    "get_font_stack",
    "set_publication_style",
    "apply_figure_text_weight",
    "apply_figure_font_family",
    "standardize_axes_text",
    "save_figure",
    "recommended_math_label",
]
