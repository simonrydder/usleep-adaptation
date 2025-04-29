import os
from typing import Literal

from matplotlib.axis import Axis
from matplotlib.figure import Figure


def save_figure(fig: Figure, path: str, dpi: int = 300) -> None:
    folder = os.path.dirname(path)
    if not os.path.exists(folder):
        os.makedirs(folder)

    fig.savefig(path, dpi=dpi, bbox_inches="tight")


def adjust_axis_font(
    ax: Axis,
    size: int | None = None,
    weight: Literal["normal", "bold"] | None = None,
    horizontal_alignment: Literal["left", "center", "right"] | None = None,
    vertical_alighnment: Literal["top", "center", "bottom"] | None = None,
    rotation: int | None = None,
    color: str | tuple | None = None,
) -> None:
    for label in ax.get_ticklabels():
        if size is not None:
            label.set_fontsize(size)

        if weight is not None:
            label.set_fontweight(weight)

        if horizontal_alignment is not None:
            label.set_horizontalalignment(horizontal_alignment)

        if vertical_alighnment is not None:
            label.set_verticalalignment(vertical_alighnment)

        if rotation is not None:
            label.set_rotation(rotation)

        if color is not None:
            label.set_color(color)
