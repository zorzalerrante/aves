# %% [markdown]
# La descripción y ejecución de este código está disponible en el stream:
#
# **Zorzal TV #1: Mapas en Python**
# https://www.twitch.tv/videos/1057304044?t=0h7m7s

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import geopandas as gpd
import contextily as cx

from sklearn.preprocessing import minmax_scale
from matplotlib import colorbar
from matplotlib_scalebar.scalebar import ScaleBar # https://github.com/ppinard/matplotlib-scalebar/

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

sns.set(context='paper', font='Fira Sans Extra Condensed', style='ticks', palette='colorblind', font_scale=1.0)


# %%
from aves.data import DATA_PATH, eod, census
from aves.features.utils import normalize_rows
from aves.features.geo import to_point_geodataframe, clip_area_geodataframe
from aves.features.weights import weighted_mean
from aves.visualization.figures import figure_from_geodataframe, small_multiples_from_geodataframe, tighten_figure
from aves.visualization.maps import dot_map, choropleth_map, bubble_map, heat_map, add_labels_from_dataframe, GeographicalNodeLink
from aves.visualization.colors import color_legend, colormap_from_palette
from aves.visualization.fdeb import FDB


# %%
zones = gpd.read_file(DATA_PATH / 'processed/scl_zonas_urbanas.json').set_index('ID')
zones.head()

# %%
zones.plot()

# %%
viajes = eod.read_trips()
print(len(viajes))

# descartamos sectores que no sean relevantes en los orígenes y destinos de los viajes
viajes = viajes[(viajes['SectorOrigen'] != 'Exterior a RM') 
                & (viajes['SectorDestino'] != 'Exterior a RM')
                & (viajes['SectorOrigen'] != 'Extensión Sur-Poniente') 
                & (viajes['SectorDestino'] != 'Extensión Sur-Poniente')
                & pd.notnull(viajes['SectorOrigen'])
                & pd.notnull(viajes['SectorDestino'])
                # también descartamos viajes que hayan sido imputados en la encuesta
                & (viajes['Imputada'] == 0)
                # y finalmente descartamos viajes cuya distancia indique que son viajes cortísimos o bien demasiado largos para el tamaño de la ciudad
                & (viajes['DistManhattan'].between(500, 45000))]

print(len(viajes))

# %%
personas = eod.read_people()
personas.head()

# %%
viajes_persona = viajes.merge(personas)
viajes_persona.head()

# %%
viajes_persona['PesoLaboral'] = viajes_persona['FactorLaboralNormal'] * viajes_persona['Factor_LaboralNormal']

# %%
viajes_persona = viajes_persona[pd.notnull(viajes_persona['PesoLaboral'])]
len(viajes_persona)

# %%
print('{} viajes expandidos a {}'.format(len(viajes_persona), int(viajes_persona['PesoLaboral'].sum())))

# %% [markdown]
# ## 1. ¿Cuál es la distribución geográfica de los viajes al trabajo desde el hogar de acuerdo al modo de transporte?

# %%
viajes_persona[['OrigenCoordX', 'OrigenCoordY']].head()

# %%
origenes_viajes = to_point_geodataframe(viajes_persona, 'OrigenCoordX', 'OrigenCoordY', crs='epsg:32719')
origenes_viajes.head()

# %%
zones = zones.to_crs(origenes_viajes.crs)
zones.plot()


# %%
fig, ax = figure_from_geodataframe(zones, height=6, remove_axes=True)
zones.plot(ax=ax, color='#efefef', edgecolor='#abacab', linewidth=1)
origenes_viajes.plot(ax=ax, markersize=1, marker='.', alpha=0.5)
tighten_figure(fig)

# %%
# los parámetros lsuffix y rsuffix indican el sufijo a agregar a las columnas de cada tabla
print(len(origenes_viajes))
origenes_urbanos = gpd.sjoin(origenes_viajes, zones, op='within', lsuffix='_l', rsuffix='_r')
print(len(origenes_urbanos))

# %%
fig, ax = figure_from_geodataframe(zones, height=6, remove_axes=True)
zones.plot(ax=ax, color='#efefef', edgecolor='#abacab', linewidth=1)
origenes_urbanos.plot(ax=ax, markersize=1, marker='.', alpha=0.5)
tighten_figure(fig)

# %%
origenes_urbanos.ModoDifusion.value_counts(normalize=True)

# %%
origenes_a_graficar = origenes_urbanos[(origenes_urbanos.Proposito == 'Al trabajo') &
                                       (origenes_urbanos.ModoDifusion.isin(['Bip!', 'Auto', 'Caminata', 'Bicicleta']))]


# %%
fig, ax = figure_from_geodataframe(zones, height=6, remove_axes=True)
zones.plot(ax=ax, color='#efefef', edgecolor='#abacab', linewidth=1)
dot_map(ax, origenes_a_graficar, category='ModoDifusion', size=10, palette='Set3')
tighten_figure(fig)

# %%
origenes_urbanos['PesoLaboral'].describe()

# %%
origenes_urbanos['PesoVisual'] = minmax_scale(origenes_urbanos['PesoLaboral'], (0.01, 1.0))
origenes_urbanos['PesoVisual'].describe()

# %%

fig, ax = figure_from_geodataframe(zones, height=6, remove_axes=True)
zones.plot(ax=ax, color='#efefef', edgecolor='white', linewidth=1)
bubble_map(ax, 
    origenes_urbanos[(origenes_urbanos.Proposito == 'Al trabajo') & origenes_urbanos.ModoDifusion.isin(['Bip!', 'Auto', 'Caminata', 'Bicicleta'])], 
    category='ModoDifusion', 
    size='PesoVisual', 
    scale=500, 
    sort_categories=False, 
    palette='husl', 
    alpha=0.45, 
    edge_color='none')
tighten_figure(fig)

# %%
fig, axes = small_multiples_from_geodataframe(zones, 4, height=6, col_wrap=2, remove_axes=True)
colors = sns.color_palette('cool', n_colors=4)

# el método zip nos permite iterar sobre tres listas simultáneamente
for ax, modo, color in zip(axes, ['Auto', 'Bip!', 'Caminata', 'Bicicleta'], colors):
    zones.plot(ax=ax, color='#efefef', edgecolor='#abacab', linewidth=0.5)
    ax.set_title(modo)
      
    origenes_a_graficar = origenes_urbanos[(origenes_urbanos.Proposito == 'Al trabajo') &
                                           (origenes_urbanos.ModoDifusion == modo)]
    
    bubble_map(ax, origenes_a_graficar, size='PesoVisual', scale=500, color=color, edge_color='none', alpha=0.45)    
    
tighten_figure(fig)

# %%
fig, ax = figure_from_geodataframe(zones, height=6, remove_axes=True)

zones.plot(ax=ax, color='#efefef', edgecolor='white', linewidth=1)

heat_map(ax, origenes_urbanos[origenes_urbanos.Proposito == 'Al trabajo'], 
         weight_column='PesoLaboral', alpha=0.75, palette='inferno', n_levels=10,
         # área de influencia
         bandwidth=1000
)

tighten_figure(fig)

# %%
fig, axes = small_multiples_from_geodataframe(zones, 4, height=6, col_wrap=2, remove_axes=True)

for ax, modo in zip(axes, ['Auto', 'Bip!', 'Caminata', 'Bicicleta']):
    zones.plot(ax=ax, color='#efefef', edgecolor='#abacab', linewidth=0.5)
    ax.set_title(modo)
      
    origenes_a_graficar = origenes_urbanos[(origenes_urbanos.Proposito == 'Al trabajo') &
                                           (origenes_urbanos.ModoDifusion == modo)]
    
    heat_map(ax, origenes_a_graficar, n_levels=10,
             weight_column='PesoLaboral', alpha=0.75, palette='inferno',
             # área de influencia
             bandwidth=1000,
             # no pintar áreas con valores muy bajos
             low_threshold=0.05
    )

tighten_figure(fig)

cax = fig.add_axes([0.25, -0.012, 0.5, 0.01])
color_legend(cax, colormap_from_palette('inferno', n_colors=10), remove_axes=True)
cax.set_title('Magnitud relativa de los viajes, de menor a mayor', loc='center', fontsize=8)


# %% [markdown]
# ## 2. ¿Cuán lejos queda el trabajo de acuerdo al lugar de residencia?

# %%
viajes_trabajo = viajes_persona[(viajes_persona.Proposito == 'Al trabajo') &
                                (pd.notnull(viajes_persona.PesoLaboral))]
print(len(viajes_trabajo), viajes_trabajo.PesoLaboral.sum())

# %%
viajes_trabajo['DistEuclidiana'].describe()

# %%
viajes_trabajo['DistEuclidiana'].mean(), weighted_mean(viajes_trabajo, 'DistEuclidiana', 'PesoLaboral')

# %%
distancia_zonas = (viajes_trabajo
                   .groupby(['ZonaOrigen'])
                   .apply(lambda x: weighted_mean(x, 'DistEuclidiana', 'PesoLaboral'))
                   .rename('distancia_al_trabajo')
)

distancia_zonas

# %%
distancia_zonas.plot(kind='kde')
plt.xlim([0, distancia_zonas.max()])
plt.title('Distancia al Trabajo por Zonas')
plt.xlabel('Distancia')
plt.ylabel('Densidad (KDE)')
sns.despine()

# %%
fig, ax = figure_from_geodataframe(zones, height=8, remove_axes=True)

ax, cax = choropleth_map(ax, zones.join(distancia_zonas, how='inner'), 'distancia_al_trabajo', 
                         k=6, legend_type='hist', binning='fisher_jenks',
                         cbar_location='lower center', cbar_height=0.4, cbar_width=6)

cax.set_title('Distancia al Trabajo')

tighten_figure(fig);


# %%


# %%
comunas = census.read_census_map('comuna').to_crs(zones.crs)
comunas.plot()

# %%
comunas_urbanas = comunas[comunas['COMUNA'].isin(zones['Com'].unique())].drop('NOM_COMUNA', axis=1).copy()
comunas_urbanas['NombreComuna'] = comunas_urbanas['COMUNA'].map(dict(zip(zones['Com'], zones['Comuna'])))
comunas_urbanas.plot()

# %%

bounding_box = zones.total_bounds
bounding_box

# %%
comunas_urbanas = clip_area_geodataframe(comunas_urbanas, zones.total_bounds, buffer=500)
comunas_urbanas.plot()

# %%

fig, ax = figure_from_geodataframe(zones, height=8, remove_axes=True)

ax, cax = choropleth_map(ax, zones.join(distancia_zonas, how='inner'), 'distancia_al_trabajo', 
                         k=6, legend_type='hist', binning='quantiles',
                         cbar_location='lower center', cbar_height=0.1, cbar_width=6, linewidth=0)

comunas_urbanas.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.5)

cax.set_title('Distancia al Trabajo')

# %%

fig, ax = figure_from_geodataframe(zones, height=8, remove_axes=True)

ax, cax = choropleth_map(ax, zones.join(distancia_zonas, how='inner'), 'distancia_al_trabajo', 
                         k=6, legend_type='colorbar', binning='quantiles', cmap='RdPu',
                         cbar_location='lower center', cbar_height=0.1, cbar_width=6, linewidth=0, )

comunas_urbanas.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.5)

add_labels_from_dataframe(ax, comunas_urbanas, 'NombreComuna', font_size=9, outline_width=1)

cax.set_title('Distancia al Trabajo')
fig.tight_layout()

# %%
fig, ax = figure_from_geodataframe(zones, height=15, remove_axes=True)

cx.add_basemap(ax, crs=zones.crs.to_string(), source="../data/processed/scl_toner_12.tif", interpolation='hanning', zorder=0)

ax, cax = choropleth_map(ax, zones.join(distancia_zonas, how='inner'), 'distancia_al_trabajo', 
                         k=6, legend_type='colorbar', binning='quantiles', cmap='RdPu',
                         cbar_location='center right', cbar_height=4, cbar_width=0.2, linewidth=0, 
                         cbar_orientation='vertical', alpha=0.8)

cax.set_title('Distancia al Trabajo')

ax.add_artist(ScaleBar(1, location='lower right', color='#abacab'))

x, y, arrow_length = 0.95, 0.1, 0.05
ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
            arrowprops=dict(facecolor='#444444', width=5, headwidth=15),
            ha='center', va='center', fontsize=20, fontname='Fira Sans Extra Condensed', color='#444444',
            xycoords=ax.transAxes);


# %%
fig, axes = small_multiples_from_geodataframe(zones, 4, height=6, col_wrap=2, remove_axes=True)

for ax, modo in zip(axes, ['Auto', 'Bip!', 'Caminata', 'Bicicleta']):
    #zones.plot(ax=ax, color='#efefef', edgecolor='#abacab', linewidth=0.5)
    ax.set_title(f'Viajes al trabajo en {modo}')

    cx.add_basemap(ax, crs=zones.crs.to_string(), source="../data/processed/scl_toner_12.tif", interpolation='hanning', zorder=0)
      
    origenes_a_graficar = origenes_urbanos[(origenes_urbanos.Proposito == 'Al trabajo') &
                                           (origenes_urbanos.ModoDifusion == modo)]
    
    heat_map(ax, origenes_a_graficar, n_levels=10,
             weight_column='PesoLaboral', alpha=0.75, palette='inferno',
             # área de influencia
             bandwidth=750,
             # no pintar áreas con valores muy bajos
             low_threshold=0.01,
             legend_type=None,
             return_heat=True
    )

fig.tight_layout()

cax = fig.add_axes([0.25, -0.012, 0.5, 0.01])
cax.set_title('Magnitud relativa de los viajes, de menor a mayor', loc='center', fontsize=8)
cax.set_axis_off()
cb3 = colorbar.ColorbarBase(cax, cmap=colormap_from_palette('inferno', n_colors=10), alpha=0.75,
                                #norm=norm,
                                ticks=range(10),
                                spacing='uniform',
                                orientation='horizontal')

# %% [markdown]
# # 3. ¿Cómo se conecta la ciudad de acuerdo a las relaciones origen-destino?

# %%
matriz = (viajes_trabajo[(viajes_trabajo['ComunaOrigen'].isin(comunas_urbanas['NombreComuna']))
                                  & (viajes_trabajo['ComunaDestino'].isin(comunas_urbanas['NombreComuna']))]
                    .groupby(['ComunaOrigen', 'ComunaDestino'])
                    .agg(n_viajes=('PesoLaboral', 'sum'))
                    .reset_index()
    )
matriz.head()


# %%
fig, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(matriz.set_index(['ComunaOrigen', 'ComunaDestino'])['n_viajes'].unstack(fill_value=0).pipe(normalize_rows), cmap='inferno_r', linewidth=1)

# %%
comunas_urbanas.head()

# %%
geonodelink = GeographicalNodeLink.from_edgelist_and_geodataframe(
    matriz[matriz['n_viajes'] > matriz['n_viajes'].quantile(0.25)],
    comunas_urbanas,
    source='ComunaOrigen', 
    target='ComunaDestino',
    node_column='NombreComuna',
    weight='n_viajes')
# %%
fig, ax = figure_from_geodataframe(zones, height=6, set_limits=True, remove_axes=True)

comunas_urbanas.plot(ax=ax, facecolor='none', edgecolor='#abacab', zorder=0)

geonodelink.plot_nodes(ax, color='white', edgecolor='black', size=250, zorder=5, use_weights='in-degree', min_size=5)

geonodelink.plot_weighted_edges(ax, palette='plasma', log_transform=False, weight_bins=4,
                                min_linewidth=1.0, linewidth=4, min_alpha=0.5, alpha=0.95, 
                                with_arrows=True, arrow_shrink=3, arrow_scale=10, zorder=4)

ax.set_title('Viajes al trabajo en Santiago (en días laborales, EOD 2012)');

# %%
matriz_zonas = (viajes_trabajo[(viajes_trabajo['ZonaOrigen'] != viajes_trabajo['ZonaDestino'])
                             & (viajes_trabajo['ZonaOrigen'].isin(zones.index))
                             & (viajes_trabajo['ZonaDestino'].isin(zones.index))]
                    .groupby(['ComunaOrigen', 'ZonaOrigen', 'ZonaDestino'])
                    .agg(n_viajes=('PesoLaboral', 'sum'))
                    .sort_values('n_viajes', ascending=False)
                    .assign(cumsum_viajes=lambda x: x['n_viajes'].cumsum())
                    .assign(cumsum_viajes=lambda x: x['cumsum_viajes'] / x['cumsum_viajes'].max())
                    .reset_index()
)

matriz_zonas.head()

# %%
matriz_zonas['cumsum_viajes'].plot()

# %%
matriz_zonas = matriz_zonas[matriz_zonas['cumsum_viajes'] <= 0.5]
matriz_zonas.shape

# %%
merged_zones = zones.reset_index().dissolve('ID')

# %%
zone_nodelink = GeographicalNodeLink.from_edgelist_and_geodataframe(
    matriz_zonas,
    merged_zones,
    source='ZonaOrigen', 
    target='ZonaDestino',
    weight='n_viajes')


# %%
fig, ax = figure_from_geodataframe(zones, height=6, set_limits=True, remove_axes=True)

zones.plot(ax=ax, facecolor='none', edgecolor='#abacab', zorder=0)

comunas_urbanas.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.5, zorder=0)

zone_nodelink.plot_nodes(ax, color='white', edgecolor='black', size=300, zorder=5, use_weights='in-degree', min_size=3)

zone_nodelink.plot_weighted_edges(ax, palette='plasma', log_transform=False, weight_bins=4,
                                min_linewidth=1.0, linewidth=4, min_alpha=0.5, alpha=0.95, 
                                with_arrows=True, arrow_shrink=3, arrow_scale=10, zorder=4)

ax.set_title('Viajes al trabajo en Santiago (en días laborales, EOD 2012)');

# %%
bundled_zone_network = FDB(zone_nodelink, 
# más alto más resistencia
K=1, 
# más alto más jiggly las líneas
S=500, 
I=10,
compatibility_threshold=0.6)

# %%

fig, ax = figure_from_geodataframe(zones, height=8, set_limits=True, remove_axes=True)

zones.plot(ax=ax, facecolor='none', edgecolor='#abacab', zorder=0)

comunas_urbanas.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.5, zorder=0)

zone_nodelink.plot_nodes(ax, color='white', edgecolor='black', size=250, zorder=5, use_weights='in-degree', min_size=5)

zone_nodelink.plot_weighted_edges(ax, 
                                palette='plasma', log_transform=False, weight_bins=10,
                                min_linewidth=0.5, linewidth=1.5, min_alpha=0.5, alpha=0.9, 
                                with_arrows=True, arrow_shrink=3, arrow_scale=10, zorder=4)

# %%
fig, ax = figure_from_geodataframe(zones, height=8, set_limits=True, remove_axes=True)

cx.add_basemap(ax, crs=zones.crs.to_string(), source="../data/processed/scl_toner_12.tif", interpolation='hanning', zorder=0)

zone_nodelink.plot_nodes(ax, color='white', edgecolor='black', size=250, zorder=5, use_weights='in-degree', min_size=5)

zone_nodelink.plot_weighted_edges(ax, 
                                palette='plasma', log_transform=False, weight_bins=10,
                                min_linewidth=0.5, linewidth=1.5, min_alpha=0.5, alpha=0.9, 
                                with_arrows=True, arrow_shrink=3, arrow_scale=10, zorder=4)
