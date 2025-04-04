# 🐦 `aves`: Análisis y Visualización, Educación y Soporte

Este repositorio contiene datos, código y notebooks relacionados con cursos (Visualización de Información, Ciencia de Datos Geográficos) y el trabajo diario de [Eduardo Graells-Garrido](http://www.datagramas.cl) y [Daniela Opitz](https://calipsotornasol.github.io/). 

Todavía no existe una documentación exhaustiva para `aves`, ya que su uso es primariamente interno, pero estos ejemplos muestran cómo se utilizan sus funciones. En lo que respecta a visualización, se mantiene el esquema típico que se utiliza en `matplotlib` y `seaborn`, las bibliotecas de visualización de bajo nivel más utilizadas en Python. De cierto modo, `aves` es un conjunto de herramientas de bajos niveles de abstracción, es decir, utiliza un paradigma imperativo, donde le damos instrucciones específicas al programa (**cómo** hacerlo); en contraste, una herramienta de alto nivel se enfoca en **qué** hacer, ocultando los detalles de implementación.

Para comprender la funcionalidad del código puedes explorar la carpeta `notebooks`. Sin embargo, los notebooks se preocupan de trabajar conceptos que en ocasiones están más allá del alcance de `aves`, ya que los utilizo en los cursos que dicto. Esos conceptos incluyen trabajar con `DataFrames` de `pandas` o utilizar técnicas de visualización implementadas en bibliotecas como `geopandas`, `matplotlib` y `seaborn` (que `aves` utiliza de manera interna).

## Ejemplos 

### Visualización de Tablas

```python
from aves.visualization.tables import barchart

fig, ax = plt.subplots(1, 1, figsize=(12, 4))

barchart(ax, modo_comuna, stacked=True, sort_categories=True, sort_items=True)

ax.set_title("Uso de Modo de Transporte en Viajes al Trabajo (Día Laboral)", loc="left")
ax.set_ylim([0, 1])
ax.set_xlabel("")
ax.set_ylabel("Fracción de los Viajes")
```

![](reports/figures/example_barchart.png)

```python
from aves.visualization.tables import scatterplot

fig, ax = plt.subplots(1, 1, figsize=(12, 6))

scatterplot(
    ax,
    modo_comuna_ingreso,
    "ingreso",
    "Bip!",
    annotate=True,
    avoid_collisions=True,
    text_args=dict(fontsize="x-small"),
    scatter_args=dict(color="purple"),
)

ax.set_xlabel("Ingreso Promedio por Hogar")
ax.set_ylabel("Proporción de Uso de Transporte Público")
ax.set_title(
    "Relación entre Uso de Transporte Público e Ingreso por Comunas de RM (Fuente: EOD2012)",
    loc="left",
)
ax.grid(alpha=0.5)
ax.ticklabel_format(style="plain")

sns.despine(ax=ax, left=True, bottom=True, right=True, top=True)

```

![](reports/figures/example_scatterplot.png)

### Visualización de Datos Geográficos
```python
from aves.visualization.figures import GeoFacetGrid
from aves.visualization.maps import choropleth_map

grid = GeoFacetGrid(zones.join(distancia_zonas, how="inner"), height=7)

grid.add_basemap("../data/processed/scl_toner_12.tif")
grid.add_layer(
    choropleth_map,
    "distancia_al_trabajo",
    k=6,
    linewidth=0.5,
    edgecolor="black",
    binning="fisher_jenks",
    palette="RdPu",
    alpha=0.75,
    cbar_args=dict(
        label="Distancia (m)",
        height="20%",
        width="1%",
        orientation="vertical",
        location="center right",
        label_size="x-small",
        bbox_to_anchor=(0.0, 0.0, 0.9, 1.0),
    ),
)
grid.add_map_elements()
grid.set_title("Distancia al Trabajo")
```

![](reports/figures/example_choropleth.png)

```python
from aves.visualization.figures import GeoFacetGrid
from aves.visualization.maps import heat_map

grid = GeoFacetGrid(
    origenes_urbanos[origenes_urbanos["Proposito"] == "Al trabajo"],
    context=zones,
    col="ModoDifusion",
    col_wrap=2,
    col_order=["Auto", "Bip!", "Caminata", "Bicicleta"],
    height=7,
)

grid.add_basemap("../data/processed/scl_toner_12.tif")
grid.add_layer(
    heat_map,
    weight="PesoLaboral",
    n_levels=10,
    bandwidth=1000,
    low_threshold=0.05,
    alpha=0.75,
    palette="inferno",
)
grid.add_global_colorbar(
    "inferno",
    10,
    title="Intensidad de Viajes (de menos a más)",
    orientation="horizontal",
)

```

![](reports/figures/example_geofacetgrid.png)

### Visualización de Redes

```python
from aves.models.network import Network
from aves.visualization.networks import NodeLink

network = Network.from_edgelist(edgelist, directed=False)
nodelink = NodeLink(network)
nodelink.layout_nodes()
nodelink.set_node_drawing(method="plain", weights=network.node_degree("total"))

fig, ax = plt.subplots(figsize=(16, 16))

nodelink.plot(ax, nodes=dict(node_size=150, edgecolor="black", linewidth=1))

ax.set_axis_off()
ax.set_aspect("equal")
```

![](reports/figures/example_nodelink.png)

```python
from aves.models.network import Network
from aves.visualization.networks import NodeLink

network = Network.from_edgelist(edgelist, directed=False)
nodelink = NodeLink(network)
heb = nodelink.bundle_edges(method="hierarchical")

nodelink.set_node_drawing(
    "labeled",
    radial=True,
    offset=0.1,
    weights=network.node_degree("total"),
    categories=heb.get_node_memberships(1),
)
nodelink.set_edge_drawing(
    "community-gradient", node_communities=heb.get_node_memberships(1)
)

fig, ax = plt.subplots(figsize=(12, 12))

nodelink.plot(
    ax,
    nodes=dict(
        node_size=150, palette="plasma", edgecolor="none", alpha=0.75, fontsize="medium"
    ),
    edges=dict(color="#abacab", palette="plasma", alpha=0.5),
)

ax.set_axis_off()
ax.set_aspect("equal")
```

![](reports/figures/example_heb.png)

### Visualización de Redes con Contexto Geográfico

```python
from aves.visualization.figures import GeoFacetGrid
from aves.models.network import Network
from aves.visualization.networks import NodeLink

zone_od_network = Network.from_edgelist(
    matriz_zonas, source="ZonaOrigen", target="ZonaDestino", weight="n_viajes"
)
zone_nodelink = NodeLink(zone_od_network)
zone_nodelink.layout_nodes(method="geographical", geodataframe=merged_zones)
zone_nodelink.bundle_edges(
    method="force-directed", K=1, S=500, I=30, compatibility_threshold=0.65, C=6
)
zone_nodelink.set_node_drawing("plain", weights=zone_od_network.node_degree("in"))
zone_nodelink.set_edge_drawing(method="origin-destination")


def plot_network(ax, geo_data, *args, **kwargs):
    zone_nodelink.plot(ax, *args, **kwargs)


grid = GeoFacetGrid(zones, context=zones, height=7)
grid.add_layer(zones, facecolor="#efefef", edgecolor="white")
grid.add_layer(comunas_urbanas, facecolor="none", edgecolor="#abacab")
grid.add_layer(
    plot_network,
    nodes=dict(color="white", edgecolor="black", node_size=100, alpha=0.95),
    edges=dict(linewidth=0.5, alpha=0.25),
)
grid.set_title("Viajes al trabajo en Santiago (en días laborales, EOD 2012)")
```

![](reports/figures/example_geo_fdb.png)

### Frecuencia y Tendencia de Palabras usando Bubble Plots

El dataframe `unisex_names` se calcula a partir del dataset guaguas (ver sección datasets).

```python
from aves.visualization.tables.bubbles import bubble_plot

fig, ax = plt.subplots(figsize=(16, 9))

bubble_plot(
    ax,
    unisex_names.reset_index(),
    "tendency",
    "n",
    label_column="nombre",
    palette="cool",
    max_label_size=56,
    starting_y_range=60, margin=2
)

ax.set_axis_off()
ax.set_title(
    "Nombres compartidos por hombres y mujeres (1920-2020, Registro Civil de Chile)"
)
ax.annotate(
    "Más usado por mujeres →",
    (0.95, 0.01),
    xycoords="axes fraction",
    ha="right",
    va="bottom",
    fontsize="medium",
    color="#abacab",
)
ax.annotate(
    "← Más usado por hombres",
    (0.05, 0.01),
    xycoords="axes fraction",
    ha="left",
    va="bottom",
    fontsize="medium",
    color="#abacab",
)
ax.annotate(
    "Fuente: guaguas, por @RivaQuiroga.",
    (0.5, 0.01),
    xycoords="axes fraction",
    ha="center",
    va="bottom",
    fontsize="medium",
    color="#abacab",
)

fig.set_facecolor("#efefef")
fig.tight_layout()
```

![](reports/figures/example_bubbleplot.png)

## Configuración y requisitos

### Paso 1: Preparación

Se recomienda usar sistema operativo Linux, o bien el [Windows Subsystem for Linux](https://docs.microsoft.com/es-es/windows/wsl/install-win10) en caso de usar Windows. En cuanto a distribución, se recomienda Ubuntu 22.04.  


Para instalar las bibliotecas necesarias para el funcionamiento de `aves`, abre la consola (_shell_) de Ubuntu y ejecuta el siguiente comando:

```sh
sudo apt install make libxcursor1 libgdk-pixbuf2.0-dev libxdamage-dev osmctools gcc libarchive-dev libxcomposite-dev
```

Además, para administrar el entorno de ejecución de aves necesitas una instalación de `conda` ([Miniconda](https://docs.conda.io/en/latest/miniconda.html) es una buena alternativa) y de `mamba`. Primero debes instalar `conda`, y una vez que la tengas, puedes ejecutar:

```sh
conda install mamba -n base -c conda-forge
```

¿Por qué `mamba`? Es una versión más eficiente de `conda`. ¡Te ahorrará muchos minutos de instalación!

Una vez que se ha instalado `conda` y `mamba`, es necesario modificar el archivo `.condarc` que está en la carpeta raíz de la cuenta de usuario/a del sistema. El archivo debe contener lo siguiente:

```
channels:
  - conda-forge
ssl_verify: true
channel_priority: strict
```

### Paso 2: Creación del entorno

Descarga o clona el repositorio usando el comando `git clone`, luego accede al directorio `aves` desde la terminal e instala el entorno de desarrollo con los siguientes comandos:

```sh
make create-env
make install-package
```

Ello creará un entorno llamado `aves` que puedes utilizar a través del comando `conda activate aves`. Para dejar de trabajar en el entorno, usa el comando `conda deactivate`.

### Paso 3: Ejecución en Jupyter

El principal modo de uso de aves es a través de los notebooks de Jupyter.

Es posible que ya tengas un entorno de `conda` en el que ejecutes Jupyter. En ese caso, puedes agregar el entorno de `aves` como _kernel_ ejecutando este comando desde el entorno que contiene Jupyter:

```sh
make install-kernel
```

Así quedará habilitado acceder al entorno de `aves` desde Jupyter.


## Actualización de dependencias

Para añadir o actualizar dependencias:

1. Agrega el nombre (y la versión si es necesaria) a la lista en `environment.yml`.
2. Ejecuta `conda env update --name aves --file environment.yml  --prune`.
3. Actualiza el archivo `environment.lock.yml` ejecutando `conda env export > environment.lock.yml`.

## Créditos

Parte del tiempo dedicado a este código ha sido financiado por el proyecto **ANID Fondecyt de Iniciación #11180913** de Eduardo Graells-Garrido, **ANID Fondecyt de Iniciación #11220799** de Daniela Opitz y Fondo de Instalación FCFM de Eduardo Graells-Garrido.

### Personas y contribuciones

* Han contribuido código y documentación: [Natalia Meza](https://www.linkedin.com/in/natalia-meza-m-4a0124253/), [Matilde Rivas](https://cl.linkedin.com/in/matilde-rivas), [Daniela Campos Fischer](https://www.linkedin.com/in/daniela-campos-fischer/), [Tabita Catalán](https://github.com/tabitaCatalan/), [Vera Sativa](https://github.com/verasativa).
* El módulo `aves.features.twokenize` es una versión modificada de [ark-twokenize](https://github.com/myleott/ark-twokenize-py) de [Myle Ott](https://github.com/myleott).
* Este repositorio fue creado gracias al template de _Cookie Cutter / Data Science with Conda_ hecho por [Patricio Reyes](https://github.com/pareyesv/).
* Gran parte de la funcionalidad de `aves` es proporcionada por las bibliotecas `matplotlib`, `seaborn`, `pandas`, `geopandas`, `contextily`, `graph-tool`, `scikit-learn`, `pysal` y otras. 
* Para los notebooks de mapas: Data by [OpenStreetMap](http://openstreetmap.org/), under [ODbL](http://www.openstreetmap.org/copyright).

### Datasets

Este repositorio incluye los siguientes datasets:

* [Encuesta Origen-Destino, Santiago 2012](http://datos.gob.cl/dataset/31616) (por SECTRA).
* [Arenas' Jazz Network](http://konect.uni-koblenz.de/networks/arenas-jazz).
* Shapefiles del [Censo 2017 de Chile](http://www.censo2017.cl/servicio-de-mapas/) para la Región Metropolitana. En [este repositorio de Diego Caro](https://github.com/diegocaro/chile_census_2017_shapefiles) pueden encontrar todas las regiones del país.
* Inscripciones de nombres en el Registro Civil de Chile a través del dataset [guaguas](https://github.com/rivaquiroga/guaguas) preparado por [Riva Quiroga](https://twitter.com/rivaquiroga).

