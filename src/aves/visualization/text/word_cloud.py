from random import Random

import matplotlib.font_manager
import numpy as np
from cytoolz import keyfilter, valfilter
from emoji.unicode_codes import EMOJI_DATA
from wordcloud import WordCloud


class colormap_size_func(object):
    """Color func created from matplotlib colormap.
    Parameters
    ----------
    colormap : string or matplotlib colormap
        Colormap to sample from
    Example
    -------
    >>> WordCloud(color_func=colormap_color_func("magma"))
    """

    def __init__(self, colormap, max_font_size):
        import matplotlib.pyplot as plt

        self.colormap = plt.cm.get_cmap(colormap)
        self.max_font_size = max_font_size

    def __call__(
        self, word, font_size, position, orientation, random_state=None, **kwargs
    ):
        if random_state is None:
            random_state = Random()
        r, g, b, _ = 255 * np.array(self.colormap(font_size / self.max_font_size))
        return "rgb({:.0f}, {:.0f}, {:.0f})".format(r, g, b)


def draw_wordcloud(
    ax,
    vocabulary,
    relative_scaling=0.5,
    size_scaling=3,
    img_scale=1,
    max_words=250,
    prefer_horizontal=1.0,
    min_font_size=8,
    max_font_size=None,
    emoji_scaling=1,
    cmap="cividis",
    emoji_path=None,
    background_color="white",
    mode="RGB",
    fontname="Fira Sans Extra Condensed",
    **kwargs
):
    fig_width = int(ax.get_window_extent().width * size_scaling)
    fig_height = int(ax.get_window_extent().height * size_scaling)

    if max_font_size is None:
        max_font_size = int(ax.get_window_extent().height * size_scaling * 0.66)

    wc = WordCloud(
        font_path=matplotlib.font_manager.findfont(fontname),
        prefer_horizontal=prefer_horizontal,
        max_font_size=max_font_size,
        min_font_size=min_font_size,
        background_color=background_color,
        color_func=colormap_size_func(cmap, max_font_size),
        width=fig_width,
        height=fig_height,
        scale=img_scale,
        max_words=max_words,
        relative_scaling=relative_scaling,
        random_state=42,
        mode=mode,
    )

    wc.generate_from_frequencies(
        keyfilter(
            lambda x: not x in EMOJI_DATA, valfilter(lambda v: v > 0, vocabulary)
        ),
        max_font_size=max_font_size,
    )
    # wc_array = wc.to_image()

    ax.imshow(
        wc,
        interpolation="hanning",
        extent=(0, wc.width, wc.height, 0),
        aspect="auto",
        **kwargs
    )

    return ax
