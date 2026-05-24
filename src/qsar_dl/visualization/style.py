"""Publication style helpers for QSAR evaluation figures."""

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
    try:
        import matplotlib as mpl
    except ImportError as exc:
        message = (
            "matplotlib is required for qsar_dl.visualization plotting helpers. "
            "Install the visualization extra, for example: pip install -e .[viz]"
        )
        raise ImportError(message) from exc
    return mpl


def _require_pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        message = (
            "matplotlib.pyplot is required to close figures in save_figure. "
            "Install the visualization extra, for example: pip install -e .[viz]"
        )
        raise ImportError(message) from exc
    return plt


def _normalize_palette_name(name: str) -> str:
    key = name.strip().lower()
    return _ALIASES.get(key, key)


def _cycle_to_length(colors: Sequence[str], n_colors: int) -> list[str]:
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
    rc: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the project publication style to matplotlib rcParams.

    The default font stack prioritizes HeiTi-compatible Chinese fonts and Arial
    for English text while keeping broadly available fallbacks.
    """

    if dpi < DEFAULT_DPI:
        raise ValueError(f"dpi must be at least {DEFAULT_DPI}.")

    mpl = _require_matplotlib()

    language_key = language.lower()
    if language_key not in {"auto", "zh", "cn", "en"}:
        raise ValueError("language must be one of: auto, zh, cn, en.")

    if language_key == "en":
        sans_fonts = ["Arial", "DejaVu Sans", "Liberation Sans"]
    else:
        sans_fonts = [
            "SimHei",
            "Microsoft YaHei",
            "Arial",
            "Noto Sans CJK SC",
            "Source Han Sans SC",
            "DejaVu Sans",
        ]

    colors = get_palette(palette)
    style_rc: dict[str, Any] = {
        "font.family": "sans-serif",
        "font.sans-serif": sans_fonts,
        "font.size": text_size,
        "font.weight": font_weight,
        "axes.titlesize": title_size,
        "axes.titleweight": font_weight,
        "axes.labelsize": label_size,
        "axes.labelweight": font_weight,
        "axes.linewidth": axis_linewidth,
        "axes.unicode_minus": False,
        "axes.prop_cycle": mpl.cycler(color=colors),
        "xtick.labelsize": tick_size,
        "ytick.labelsize": tick_size,
        "xtick.major.width": axis_linewidth,
        "ytick.major.width": axis_linewidth,
        "xtick.minor.width": max(axis_linewidth * 0.8, 0.8),
        "ytick.minor.width": max(axis_linewidth * 0.8, 0.8),
        "legend.fontsize": legend_size,
        "legend.frameon": False,
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
    if rc:
        style_rc.update(dict(rc))

    mpl.rcParams.update(style_rc)
    return style_rc


def _normalize_formats(output_path: Path, formats: Iterable[str] | None) -> tuple[Path, tuple[str, ...]]:
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
    **savefig_kwargs: Any,
) -> list[Path]:
    """Save a matplotlib figure to one or more publication formats.

    By default the figure is exported as PNG, TIFF, PDF, and SVG at 300 DPI.
    Passing ``formats=None`` saves only the suffix included in ``output_path``.
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

    saved_paths: list[Path] = []
    for fmt in normalized_formats:
        figure_path = stem.with_suffix(f".{fmt}")
        fig.savefig(figure_path, format=fmt, **default_kwargs)
        saved_paths.append(figure_path)

    if close:
        plt = _require_pyplot()
        plt.close(fig)

    return saved_paths
