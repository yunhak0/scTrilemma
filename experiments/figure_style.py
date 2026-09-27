"""Shared figure typography for the paper figures.

The figures modulate emphasis rather than bolding everything:

* **bold** -- panel titles, axis labels, group headers, and data-value annotations
* normal -- numeric tick labels, reference readouts, and parenthetical sublabels
* ``PRIMARY_TEXT`` for foreground text, ``SECONDARY_TEXT`` for subordinate notes

This module carries only the text half of that recipe -- weight, colour and
family. Font *sizes* stay with each figure, because the appendix figures are laid
out at very different densities, and no graphical property (line widths, colours,
markers, layout) is touched.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

PRIMARY_TEXT = "#202020"
SECONDARY_TEXT = "#777777"

# Weight/colour/family only -- sizes stay with the calling figure.
RC_PARAMS: dict[str, Any] = {
    "font.family": "DejaVu Sans",
    "font.weight": "bold",
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "text.color": PRIMARY_TEXT,
    "axes.labelcolor": PRIMARY_TEXT,
    "xtick.labelcolor": PRIMARY_TEXT,
    "ytick.labelcolor": PRIMARY_TEXT,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply_style(overrides: dict[str, Any] | None = None) -> None:
    """Install the shared hierarchy, letting the caller keep its own font sizes."""
    params = dict(RC_PARAMS)
    if overrides:
        params.update(overrides)
    plt.rcParams.update(params)


def demote_ticks(
    axes: Axes | Iterable[Axes],
    which: Literal["x", "y", "both"] = "both",
) -> None:
    """Return numeric tick labels to normal weight.

    ``font.weight`` is bold globally so that titles and annotations lead; numeric
    ticks are lightened so they recede. Call this only for numeric
    axes -- categorical ticks that name a condition should stay bold.
    """
    axis_list = [axes] if isinstance(axes, Axes) else list(axes)
    for axis in axis_list:
        labels = []
        if which in {"x", "both"}:
            labels.extend(axis.get_xticklabels())
        if which in {"y", "both"}:
            labels.extend(axis.get_yticklabels())
        for label in labels:
            label.set_fontweight("normal")


def demote_colorbar(colorbar: Any) -> None:
    """Lighten a colorbar's numeric ticks while keeping its label bold."""
    for label in colorbar.ax.get_yticklabels() + colorbar.ax.get_xticklabels():
        label.set_fontweight("normal")


def demote_legend(legend: Any) -> None:
    """Keep legend entries at normal weight so titles and values lead."""
    if legend is None:
        return
    for text in legend.get_texts():
        text.set_fontweight("normal")


def secondary_text(*text_objects: Any) -> None:
    """Mark annotations as subordinate: normal weight in the secondary colour."""
    for text_object in text_objects:
        text_object.set_fontweight("normal")
        text_object.set_color(SECONDARY_TEXT)
