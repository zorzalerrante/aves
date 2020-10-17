# AVES: Analysis & Visualization -- Education & Support

Por **Eduardo Graells-Garrido**, Barcelona Supercomputing Center (BSC). <eduardo.graells@bsc.es>

Este repositorio contiene datos, código y notebooks relacionados con mi [curso de Visualización de Información](http://datagramas.cl/courses/infovis). Lo he estructurado en un paquete llamado `aves`, sigla descrita en el título de este documento. 

El repositorio incluye los siguientes datasets:

  * [Encuesta Origen-Destino, Santiago 2012](http://datos.gob.cl/dataset/31616) (por SECTRA).
  * [Arenas' Jazz Network](http://konect.uni-koblenz.de/networks/arenas-jazz).

Y las siguientes unidades de práctica en notebooks:

  1. Los Datos y las Herramientas: Encuesta Origen-Destino + Python
  1. Tablas: análisis de la EOD.
  1. Mapas: uso de `geopandas` y análisis de la EOD.
  1. Redes: uso de `graph-tool` y análisis de la EOD.

## Configuración y Requisitos

Después de clonar el repositorio, debes instalar el entorno de conda:

```sh
make conda-create-env
make install-package
```

Es posible que ya tengas un entorno de conda en el que ejecutes Jupyter. En ese caso, puedes agregar este entorno como kernel ejecutando este comando desde el entorno que contiene Jupyter:

```sh
python -m ipykernel install --user  --name aves --display-name 'AVES'
```

Para replicar los gráficos de los notebooks, debes instalar la fuente [Fira Sans](https://bboxtype.com/typefaces/FiraSans/#!layout=specimen) y [Fira Code](https://github.com/tonsky/FiraCode).

Si usas Windows, te recomiendo instalar el [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/install-win10).

## Actualización de Dependencias

Para añadir o actualizar dependencias:

1. Agrega el nombre (y la versión si es necesaria) a la lista en `environment.yml`.
2. Ejecuta `conda env update --name aves --file environment.yml  --prune`.
3. Actualiza el archivo `environment.lock.yml` ejecutando `conda env export > environment.lock.yml`.

## Créditos

Este repositorio fue creado gracias al template de _Cookie Cutter / Data Science with Conda_ hecho por [Patricio Reyes](https://github.com/pareyesv/).
