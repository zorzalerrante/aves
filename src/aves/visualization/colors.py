import numpy as np
import matplotlib.colors as colors
import seaborn as sns
import spectra
from matplotlib import colorbar
from matplotlib.patches import Patch

# from http://chris35wills.github.io/matplotlib_diverging_colorbar/

# set the colormap and centre the colorbar
class MidpointNormalize(colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))
    
    
def colormap_from_palette(palette_name, n_colors=10):
    return colors.ListedColormap(sns.color_palette(palette_name, n_colors=n_colors))


def color_legend(ax, color_list, bins, sizes=None, orientation='horizontal'):
    if bins is None or colors is None:
        raise Exception('bins and colors are required if size is not None (histogram)')
            
    if sizes is not None:
        bar_width = (bins[1:] - bins[0:-1])
        if orientation == 'horizontal':
            ax.bar(bins[:-1], sizes, width=bar_width, align='edge', color=color_list, edgecolor=color_list)
            ax.set_xticks(bins)
        else:
            ax.barh(bins[:-1], sizes, height=bar_width, align='edge', color=color_list, edgecolor=color_list)
            ax.set_yticks(bins)
        sns.despine(ax=ax, top=True, bottom=True, left=True, right=True)
    else:
        cbar_norm = colors.BoundaryNorm(bins, len(bins) - 1)
        cmap = colors.ListedColormap(color_list)
        colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=cbar_norm,
                                ticks=bins,
                                spacing='proportional',
                                orientation=orientation)
        sns.despine(ax=ax, top=True, bottom=True, left=True, right=True)   
    
    
def bivariate_matrix_from_palette(palette_name='PiYG', n_colors=3):
    #full_palette = sns.diverging_palette(150, 275, s=80, l=40, n=(cmap_n_colors - 1) * 2 + 1)
    full_palette = sns.color_palette(palette_name, n_colors=(n_colors - 1) * 2 + 1)

    cmap_x = full_palette[n_colors - 1:]
    cmap_y = list(reversed(full_palette))[n_colors - 1:]

    cmap_xy = []

    for j in range(n_colors):
        for i in range(n_colors):
            x = spectra.rgb(*cmap_x[i][0:3])
            y = spectra.rgb(*cmap_y[j][0:3])

            if i == j and i == 0:
                cmap_xy.append(x.darken(1.5).rgb)
            elif i == 0:
                cmap_xy.append(y.rgb)
            elif j == 0:
                cmap_xy.append(x.rgb)
            else: 
                blended = x.blend(y, ratio=0.5)

                if i == j:
                    blended = blended.saturate(7.5 * (i + 1))
                else:
                    blended = blended.saturate(4.5 * (i + 1))

                cmap_xy.append(blended.rgb)

    cmap_xy = np.array(cmap_xy).reshape(n_colors, n_colors, 3)
    return cmap_xy

def categorical_color_legend(ax, color_list, labels, loc='upper left', n_columns=None):
    legend_elements = []

    for label, color in zip(labels, color_list):
        legend_elements.append(Patch(facecolor=color, edgecolor='none', label=label))

    if n_columns is not None:
        if type(n_columns) != int:
            n_columns = len(color_list)
    else:
        n_columns = 1
        
    ax.legend(handles=legend_elements, loc=loc, frameon=False, ncol=n_columns)