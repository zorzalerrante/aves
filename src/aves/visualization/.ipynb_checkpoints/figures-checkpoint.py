import matplotlib.pyplot as plt

def figure_from_geodataframe(geodf, height=5, bbox=None, remove_axes=False, set_limits=True):
    if bbox is None:
        bbox = geodf.total_bounds
        
    aspect = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1])
    fig, ax = plt.subplots(figsize=(height * aspect, height))
    
    if set_limits:
        ax.set_xlim([bbox[0], bbox[2]])
        ax.set_ylim([bbox[1], bbox[3]])
    
    if remove_axes:
        ax.set_axis_off()
        
    return fig, ax

def small_multiples_from_geodataframe(geodf, n_variables, height=5, col_wrap=5, bbox=None, sharex=True, sharey=True, remove_axes=False, set_limits=True, flatten_axes=True, equal_aspect=True):
    if n_variables <= 1:
        return figure_from_geodataframe(geodf, height=height, bbox=bbox, remove_axes=remove_axes, set_limits=set_limits)
    
    if bbox is None:
        bbox = geodf.total_bounds
        
    aspect = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1])
    
    n_columns = min(col_wrap, n_variables)
    n_rows = n_variables // n_columns
    if n_rows * n_columns < n_variables:
        n_rows += 1
        
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(n_columns * height * aspect, n_rows * height), sharex=sharex, sharey=sharey)
    flattened = axes.flatten()
    
    if set_limits:
        for ax in flattened:
            ax.set_xlim([bbox[0], bbox[2]])
            ax.set_ylim([bbox[1], bbox[3]])
    
    if remove_axes:
        for ax in flattened:
            ax.set_axis_off()
    else:
        # deactivate only unneeded axes
        for i in range(n_variables, len(axes)):
            flattened[i].set_axis_off()
    
    if equal_aspect:
        for ax in flattened:
            ax.set_aspect('equal')
        
    
    if flatten_axes:
        return fig, flattened
    
    return fig, axes

def tighten_figure(fig):
    fig.tight_layout(pad=0, h_pad=0, w_pad=0)
    fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1)