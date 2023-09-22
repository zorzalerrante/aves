import matplotlib.colors as colors
import numpy as np
import seaborn as sns

from . import MidpointNormalize


def build_palette(
    bins,
    palette=None,
    center_value=None,
    default_divergent="RdBu_r",
    default_negative="Blues_r",
    default_positive="Reds",
    palette_type="light",
):
    cmap_name = None
    norm = None
    midpoint = 0.0
    using_divergent = False
    built_palette = None
    k = len(bins) - 1

    min_value, max_value = np.min(bins), np.max(bins)

    if center_value is not None:
        midpoint = center_value

    if palette is None:
        if min_value < 0 and max_value > 0:
            # divergent
            cmap_name = default_divergent
            norm = MidpointNormalize(vmin=min_value, vmax=max_value, midpoint=midpoint)
            using_divergent = True
        elif min_value < 0:
            # sequential
            cmap_name = default_negative
        else:
            # sequential
            cmap_name = default_positive
    else:
        if not isinstance(palette, colors.Colormap):
            if colors.is_color_like(palette):
                if palette_type == "light":
                    built_palette = sns.light_palette(palette, n_colors=k)
                else:
                    built_palette = sns.dark_palette(palette, n_colors=k)
            else:
                built_palette = sns.color_palette(palette, n_colors=k)

    if norm is None:
        if center_value is None:
            # norm = colors.Normalize(vmin=min_value, vmax=max_value)
            norm = colors.BoundaryNorm(bins, k)
        else:
            norm = MidpointNormalize(vmin=min_value, vmax=max_value, midpoint=midpoint)
            using_divergent = True

    if built_palette is None:
        if not using_divergent:
            built_palette = sns.color_palette(cmap_name, n_colors=k)
        else:
            middle_idx = np.where((bins[:-1] * bins[1:]) <= 0)[0][0]
            left = middle_idx
            right = k - middle_idx - 1
            if left == right:
                built_palette = sns.color_palette(cmap_name, n_colors=k)
            else:
                delta = np.abs(left - right)
                expanded_k = k + delta
                start_idx = max(middle_idx - left, right - middle_idx)
                built_palette = sns.color_palette(cmap_name, n_colors=expanded_k)[
                    start_idx : start_idx + k
                ]

    return built_palette
