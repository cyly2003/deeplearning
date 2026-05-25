from __future__ import annotations

import pytest

from qsar_dl.visualization import get_palette


def test_get_palette_returns_requested_number_of_colors() -> None:
    palette = get_palette("journal", n_colors=10)

    assert len(palette) == 10
    assert palette[0] == "#0072B2"
    assert palette[8] == "#0072B2"


def test_get_palette_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown palette"):
        get_palette("not-a-palette")


def test_set_publication_style_updates_matplotlib_rcparams() -> None:
    matplotlib = pytest.importorskip("matplotlib")

    from qsar_dl.visualization import set_publication_style

    style = set_publication_style()

    assert style["axes.titlesize"] == 20
    assert style["xtick.labelsize"] == 18
    assert style["ytick.labelsize"] == 18
    assert style["font.weight"] == "bold"
    assert style["axes.linewidth"] == 1.5
    assert style["legend.frameon"] is False
    assert matplotlib.rcParams["savefig.dpi"] >= 300


def test_save_figure_applies_bold_weight_to_text(tmp_path) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    pyplot = pytest.importorskip("matplotlib.pyplot")

    from qsar_dl.visualization import save_figure, set_publication_style

    set_publication_style(font_weight="bold")
    fig, ax = pyplot.subplots()
    ax.set_title("Diagnostic")
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    text = ax.text(0.5, 0.5, "R2=0.90")

    save_figure(fig, tmp_path / "bold_text", formats=["png"], close=True)

    assert text.get_fontweight() == "bold"


def test_save_figure_exports_publication_formats(tmp_path) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    pyplot = pytest.importorskip("matplotlib.pyplot")

    from qsar_dl.visualization import save_figure, set_publication_style

    set_publication_style()
    fig, ax = pyplot.subplots()
    ax.plot([0, 1], [0, 1], label="R2")
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.legend()

    saved_paths = save_figure(fig, tmp_path / "performance_plot", close=True)

    assert {path.suffix for path in saved_paths} == {".png", ".tiff", ".pdf", ".svg"}
    assert all(path.exists() for path in saved_paths)
    assert all(path.stat().st_size > 0 for path in saved_paths)


def test_save_figure_rejects_low_dpi(tmp_path) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    pyplot = pytest.importorskip("matplotlib.pyplot")

    from qsar_dl.visualization import save_figure

    fig, _ = pyplot.subplots()
    with pytest.raises(ValueError, match="dpi must be at least 300"):
        save_figure(fig, tmp_path / "low_dpi", dpi=200)
    pyplot.close(fig)
