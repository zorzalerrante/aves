from operator import itemgetter
from random import Random

import matplotlib.font_manager
import matplotlib.pyplot as plt
import numpy as np
from cytoolz import keymap
from emoji.unicode_codes import UNICODE_EMOJI
from PIL import Image, ImageDraw, ImageFont
from wordcloud import WordCloud
from wordcloud.wordcloud import IntegralOccupancyMap

from aves.visualization.text.emoji import draw_emoji, load_emoji, remove_prefix
from matplotlib.pyplot import imread as read_png


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
    load_func=None,
    fontname="Fira Sans Extra Condensed",
):

    fig_width = int(ax.get_window_extent().width * size_scaling)
    fig_height = int(ax.get_window_extent().height * size_scaling)

    if max_font_size is None:
        max_font_size = int(ax.get_window_extent().height * size_scaling * 0.66)

    if load_func is None:
        load_func = load_emoji

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
    )
    emoji_box_size = {}

    def generate_from_frequencies(self, frequencies, max_font_size=None):
        # make sure frequencies are sorted and normalized
        frequencies = sorted(frequencies.items(), key=itemgetter(1), reverse=True)
        frequencies = frequencies[: self.max_words]

        # largest entry will be 1
        max_frequency = float(frequencies[0][1])

        frequencies = [(word, freq / max_frequency) for word, freq in frequencies]

        if self.random_state is not None:
            random_state = self.random_state
        else:
            random_state = Random()

        boolean_mask = None
        height, width = self.height, self.width
        occupancy = IntegralOccupancyMap(height, width, boolean_mask)

        # create image
        img_grey = Image.new("L", (width, height))
        draw = ImageDraw.Draw(img_grey)
        img_array = np.asarray(img_grey)
        font_sizes, positions, orientations, colors = [], [], [], []

        last_freq = 1.0

        if max_font_size is None:
            # if not provided use default font_size
            max_font_size = self.max_font_size

        if max_font_size is None:
            # figure out a good font size by trying to draw with
            # just the first two words
            if len(frequencies) == 1:
                # we only have one word. We make it big!
                font_size = self.height
            else:
                self.generate_from_frequencies(
                    dict(frequencies[:2]), max_font_size=self.height
                )
                # find font sizes
                sizes = [x[1] for x in self.layout_]
                try:
                    font_size = int(2 * sizes[0] * sizes[1] / (sizes[0] + sizes[1]))
                # quick fix for if self.layout_ contains less than 2 values
                # on very small images it can be empty
                except IndexError:
                    try:
                        font_size = sizes[0]
                    except IndexError:
                        raise ValueError(
                            "Couldn't find space to draw. Either the Canvas size"
                            " is too small or too much of the image is masked "
                            "out."
                        )
        else:
            font_size = max_font_size

        # we set self.words_ here because we called generate_from_frequencies
        # above... hurray for good design?
        self.words_ = dict(frequencies)

        if self.repeat and len(frequencies) < self.max_words:
            # pad frequencies with repeating words.
            times_extend = int(np.ceil(self.max_words / len(frequencies))) - 1
            # get smallest frequency
            frequencies_org = list(frequencies)
            downweight = frequencies[-1][1]
            for i in range(times_extend):
                frequencies.extend(
                    [
                        (word, freq * downweight ** (i + 1))
                        for word, freq in frequencies_org
                    ]
                )

        # start drawing grey image
        for word, freq in frequencies:
            if freq == 0:
                continue
            # select the font size
            rs = self.relative_scaling
            if rs != 0:
                font_size = int(
                    round((rs * (freq / float(last_freq)) + (1 - rs)) * font_size)
                )

            while True:
                # try to find a position

                # get size of resulting text
                if word not in UNICODE_EMOJI["en"]:
                    font = ImageFont.truetype(self.font_path, font_size)
                    # transpose font optionally
                    transposed_font = ImageFont.TransposedFont(font, orientation=None)
                    box_size = draw.textsize(word, font=transposed_font)
                else:
                    # print('rendering emoji', word)
                    font = ImageFont.truetype(
                        matplotlib.font_manager.findfont("DejaVu Sans"),
                        int(font_size * emoji_scaling * 1.05),
                    )
                    transposed_font = ImageFont.TransposedFont(font, orientation=None)
                    # transposed_font = None
                    emoji = load_func(word, path=emoji_path)
                    if emoji is None:
                        print(word, "not found", remove_prefix(word))
                        break
                    box_size = draw.textsize("O", font=transposed_font)
                    emoji_box_size[word] = [box_size[0], box_size[0]]
                # find possible places using integral image:
                result = occupancy.sample_position(
                    box_size[0] + self.margin, box_size[0] + self.margin, random_state
                )
                if result is not None or font_size < self.min_font_size:
                    # either we found a place or font-size went too small
                    break
                # if we didn't find a place, make font smaller
                # but first try to rotate!
                font_size -= self.font_step

            if font_size < self.min_font_size:
                # we were unable to draw any more
                break

            x, y = np.array(result) + self.margin // 2
            # actually draw the text
            if word not in UNICODE_EMOJI["en"]:
                draw.text((y, x), word, fill="white", font=transposed_font)
            else:
                if not word in emoji_box_size:
                    continue
                emoji = load_func(word, path=emoji_path)
                pil_emoji = Image.fromarray(
                    (emoji * 255).astype(np.uint8), mode="RGBA"
                ).resize(emoji_box_size[word])
                img_grey.paste(pil_emoji, (y, x))
            positions.append((x, y))
            orientations.append(None)
            font_sizes.append(font_size)
            colors.append(
                self.color_func(
                    word,
                    font_size=font_size,
                    position=(x, y),
                    orientation=None,
                    random_state=random_state,
                    font_path=self.font_path,
                )
            )
            # recompute integral image
            if self.mask is None:
                img_array = np.asarray(img_grey)
            else:
                img_array = np.asarray(img_grey) + boolean_mask
            # recompute bottom right
            # the order of the cumsum's is important for speed ?!
            occupancy.update(img_array, x, y)
            last_freq = freq

        self.layout_ = list(
            zip(frequencies, font_sizes, positions, orientations, colors)
        )
        return self

    def to_image(wc):
        wc._check_generated()
        height, width = wc.height, wc.width

        img = Image.new(
            "RGBA",
            (int(width * wc.scale), int(height * wc.scale)),
            wc.background_color,
        )
        draw = ImageDraw.Draw(img)
        for (word, count), font_size, position, orientation, color in wc.layout_:
            if word not in UNICODE_EMOJI["en"]:
                font = ImageFont.truetype(wc.font_path, int(font_size * wc.scale))
                transposed_font = ImageFont.TransposedFont(font, orientation=None)
                pos = (int(position[1] * wc.scale), int(position[0] * wc.scale))
                draw.text(pos, word, fill=color, font=transposed_font)
            else:
                print("rendering emoji", word)
                pos = (int(position[1] * wc.scale), int(position[0] * wc.scale))
                emoji = load_func(word, path=emoji_path)
                if emoji is None:
                    print(word, "unavailable")
                    continue
                pil_emoji = Image.fromarray(
                    (emoji * 255).astype(np.uint8), mode="RGBA"
                ).resize(
                    [
                        emoji_box_size[word][1] * wc.scale,
                        emoji_box_size[word][0] * wc.scale,
                    ],
                    resample=Image.HAMMING,
                )
                img.alpha_composite(pil_emoji, pos)

        return np.array(wc._draw_contour(img=img))

    generate_from_frequencies(wc, vocabulary, max_font_size=max_font_size)
    wc_array = to_image(wc)

    ax.imshow(
        wc_array,
        interpolation="hanning",
        extent=(0, wc.width, wc.height, 0),
        aspect="auto",
    )

    return ax
