from matplotlib.pyplot import imread as read_png
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.preprocessing import minmax_scale, robust_scale
import numpy as np
import os
import re
from cytoolz import memoize


def draw_emoji(
    ax,
    key,
    position,
    zoom=0.2,
    code=None,
    frameon=False,
    xycoords="data",
    xybox=None,
    path=None,
):
    img = load_emoji(key, code=code, path=path)
    if img is None:
        return

    if xybox is None:
        xybox = (0, 0)

    imagebox = OffsetImage(img, zoom=zoom, interpolation="hanning")

    ab = AnnotationBbox(
        imagebox,
        position,
        xycoords=xycoords,
        xybox=xybox,
        boxcoords="offset points",
        frameon=frameon,
    )

    ax.add_artist(ab)


MOCHI_IMAGES = "../twemoji/assets/72x72"
PREFIX = re.compile("^u|^U0+")


def remove_prefix(key):
    return PREFIX.sub("", key.encode("unicode-escape")[1:].decode("ascii"))


@memoize
def load_emoji(key, code=None, path=None):
    if path is None:
        path = MOCHI_IMAGES
    if code is None:
        img_code = None
        for i in range(len(key)):
            if img_code is None:
                img_code = remove_prefix(key[0])
            else:
                img_code = img_code + "-" + remove_prefix(key[1])
            if i > 5:
                break
        try:
            img = read_png("{}/{}.png".format(path, img_code))
        except FileNotFoundError:
            print(
                key,
                key.encode("unicode-escape")[1:].decode("ascii"),
                img_code,
                "not found",
            )
            raise Exception(key)

    else:
        img = read_png("{}/{}.png".format(path, code))
    return img
