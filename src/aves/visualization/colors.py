import numpy as np
import matplotlib.colors as colors
import seaborn as sns

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


def color_legend(ax, color_list, bins, norm=None, sizes=None, orientation='horizontal'):
    if bins is None or colors is None:
        raise Exception('bins and colors are required if size is not None (histogram)')
            
    if sizes is not None:
        bar_width = (bins[1:] - bins[0:-1])
        if orientation == 'horizontal':
            ax.bar(bins[:-1], sizes, width=bar_width, align='edge', color=color_list, edgecolor='none')
            ax.set_xticks(bins)
        else:
            ax.barh(bins[:-1], sizes, height=bar_width, align='edge', color=color_list, edgecolor='none')
            ax.set_yticks(bins)
        sns.despine(ax=ax, top=True, bottom=True, left=True, right=True)
    elif norm is not None:
        cbar_norm = colors.BoundaryNorm(bins, len(bins) - 1)
        cmap = colors.ListedColormap(color_list)
        colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=cbar_norm,
                                ticks=bins,
                                spacing='proportional',
                                orientation=orientation)
        sns.despine(ax=ax, top=True, bottom=True, left=True, right=True)
    else:
        raise Exception('Invalid legend type. norm and size are None')    
    