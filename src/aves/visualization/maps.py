import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.colors as colors
import matplotlib.patheffects as path_effects
from mpl_toolkits.axes_grid.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import from_levels_and_colors
from matplotlib import colorbar
from mapclassify import FisherJenks
from aves.visualization.colors import color_legend, MidpointNormalize, colormap_from_palette
from aves.features.geo import kde_from_points


def bubble_map(ax, geodf, category=None, size=None, scale=1, sort_categories=False, palette=None, color=None, add_legend=True, edge_color='white', alpha=1.0):
    if category is not None:
        n_colors = len(geodf[category].unique())
        geodf = geodf[pd.notnull(geodf[category])]
        
        if sort_categories:
            geodf = geodf.sort_values(category)
    else:
        n_colors = 1

    marker = 'o'
    
    if size is not None:
        if type(size) == str:
            marker_size = geodf[size] 
        else:
            marker_size = float(size)
    else:
        marker_size = 1
        
    if palette is None:
        palette = 'cool'
        
    if category is not None:
        cmap = colormap_from_palette(palette, n_colors=n_colors)
        color = None
    else:
        cmap = None
        if color is None:
            color = colors.rgb2hex(sns.color_palette(palette, n_colors=1)[0])
        
    return (geodf.plot(categorical=True, 
       column=category,
       ax=ax, 
       marker=marker, 
       markersize=marker_size * scale,
       edgecolor=edge_color,
       alpha=alpha,
       cmap=cmap, 
       facecolor=color,
       legend=add_legend))


def dot_map(ax, geodf, category=None, size=10, palette=None, add_legend=True, sort_categories=False):
    return bubble_map(ax, geodf, category=category, size=float(size), palette=palette, add_legend=add_legend, sort_categories=sort_categories, edge_color='none')
    
    
def choropleth_map(ax, geodf, column, k=10, cmap=None, default_divergent='RdBu_r', default_negative='Blues_r', default_positive='Reds', palette_type='light',
                    cbar_label=None, cbar_width=2.4, cbar_height=0.15, cbar_location='upper left', cbar_orientation='horizontal', cbar_pad=0.05,
                    cbar_bbox_to_anchor=(0.0, 0.0, 1.0, 1.0), cbar_bbox_transform=None, legend_type='colorbar', edgecolor='white',
                    palette_center=None, fisher_jenks=False):
    
    if cbar_bbox_transform is None:
        cbar_bbox_transform = ax.transAxes
        
    geodf = geodf[pd.notnull(geodf[column])].copy()
        
    if cbar_location != 'out':
        cbar_ax = inset_axes(ax, width=cbar_width, height=cbar_height, loc=cbar_location, bbox_to_anchor=cbar_bbox_to_anchor, bbox_transform=cbar_bbox_transform, borderpad=0)
    else:
        divider = make_axes_locatable(ax)
        cbar_main = divider.append_axes('bottom' if cbar_orientation == 'horizontal' else 'right', 
                                      size=cbar_height if cbar_orientation == 'horizontal' else cbar_width, 
                                      pad=cbar_pad)
        cbar_main.set_axis_off()
        cbar_ax = inset_axes(cbar_main, width=cbar_width, height=cbar_height, loc='center', bbox_to_anchor=(0.0, 0.0, 1.0, 1.0), bbox_transform=cbar_main.transAxes, borderpad=0)
    
    min_value, max_value = geodf[column].min(), geodf[column].max()
    
    if fisher_jenks:
        binning = FisherJenks(geodf[column], k=k)
        bins = np.insert(binning.bins, 0, geodf[column].min())
        geodf = geodf.assign(__bin__=binning.yb)
    else:
        bins = np.linspace(min_value, max_value + (max_value - min_value) * 0.001, num=k + 1)
        geodf = geodf.assign(__bin__=lambda x: pd.cut(x[column], bins=bins, include_lowest=True, labels=False).astype(np.int))

    cmap_name = None
    norm = None
    midpoint = 0.0
    using_divergent = False
    
    if palette_center is not None:
        midpoint = palette_center
    
    if cmap is None:
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
        if not isinstance(cmap, colors.Colormap):
            if colors.is_color_like(cmap):
                if palette_type == 'light':
                    cmap = colors.ListedColormap(sns.light_palette(cmap, n_colors=k))
                else:
                    cmap = colors.ListedColormap(sns.dark_palette(cmap, n_colors=k))
            else:
                cmap_name = cmap
                cmap = None

    if norm is None:
        if palette_center is None:
            norm = colors.Normalize(vmin=min_value, vmax=max_value)
        else:
            norm = MidpointNormalize(vmin=min_value, vmax=max_value, midpoint=midpoint)
            using_divergent = True
    
    if cmap is None:
        cmap = colors.ListedColormap(sns.color_palette(palette=cmap_name, n_colors=k))
    
    def pick_color(b1, b0):
        if not using_divergent or np.sign(b1) == np.sign(b0):
            res = float(norm(0.5 * (b1 + b0)))
        else:
            res = float(norm(midpoint))
        return res
    
    built_palette = [cmap(pick_color(b1, b0)) for b0, b1 in zip(bins[:-1], bins[1:])]
    
    if legend_type == 'hist':
        color_legend(cbar_ax, built_palette, bins, sizes=np.histogram(geodf[column], bins=bins)[0], orientation=cbar_orientation)
    elif legend_type == 'colorbar':
        color_legend(cbar_ax, built_palette, bins, norm=colors.BoundaryNorm(bins, k), orientation=cbar_orientation)
            
    for idx, group in geodf.groupby('__bin__'):
        group.plot(ax=ax, facecolor=built_palette[idx], edgecolor=edgecolor)
        
    return ax, cbar_ax


def heat_map(ax, geodf, low_threshold=0, max_threshold=1.0, n_levels=5, alpha=1.0, palette='magma', kernel='cosine', norm=2, bandwidth=1e-2, grid_points=2**6, weight_column=None, return_heat=False):
    heat = kde_from_points(geodf, kernel=kernel, norm=norm, bandwidth=bandwidth, grid_points=grid_points, weight_column=weight_column)
    
    norm_heat = (heat[2] / heat[2].max())
    cmap = colormap_from_palette(palette, n_colors=n_levels)

    levels = np.linspace(low_threshold, max_threshold, n_levels)
    
    if not return_heat:
        return ax.contourf(heat[0], heat[1], norm_heat, levels, alpha=alpha, cmap=cmap)
    else:
        return ax.contourf(heat[0], heat[1], norm_heat, levels, alpha=alpha, cmap=cmap), heat
    

def add_labels_from_dataframe(ax, geodf, use_index=True, column=None, font_size=11, font_weight='bold', color='white', outline=True, outline_color='black', outline_width=2):
    labels = []
    for idx, row in geodf.iterrows():
        centroid = row.geometry.centroid
        if use_index:
            label = idx
        else:
            label = row[column]
            
        t = ax.text(centroid.x, centroid.y, label, horizontalalignment='center', fontsize=font_size, fontweight=font_weight, color=color)
        if outline:
            t.set_path_effects([path_effects.Stroke(linewidth=outline_width, foreground=outline_color), path_effects.Normal()])
        labels.append(t)
        
    return labels