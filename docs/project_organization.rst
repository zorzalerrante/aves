

.. _proj_structure:

=================
Project Structure
=================

The organization of the project is the following:

.. code-block:: text

        ├── LICENSE
        ├── Makefile            <- Makefile with commands like `make data` or `make train`
        ├── README.md           <- The top-level README for developers using this project.
        ├── data
        │   ├── external        <- Data from third party sources.
        │   ├── interim         <- Intermediate data that has been transformed.
        │   ├── processed       <- The final, canonical data sets for modeling.
        │   └── raw             <- The original, immutable data dump.
        │
        ├── docs                <- A default Sphinx project; see sphinx-doc.org for details
        │
        ├── models              <- Trained and serialized models, model predictions, or model summaries
        │
        ├── notebooks           <- Jupyter notebooks. Naming convention is a number (for ordering),
        │                          the creator's initials, and a short `-` delimited description, e.g.
        │                          `1.0-jqp-initial-data-exploration`.
        │
        ├── references          <- Data dictionaries, manuals, and all other explanatory materials.
        │
        ├── reports             <- Generated analysis as HTML, PDF, LaTeX, etc.
        │   └── figures         <- Generated graphics and figures to be used in reporting
        │
        ├── requirements.txt    <- The requirements file for reproducing the analysis environment, e.g.
        │                          generated with `pip freeze > requirements.txt`
        │
        ├── environment.yml     <- The Anaconda environment requirements file for reproducing the analysis environment.
        │                          This file is used by Anaconda to create the project environment.
        │
        ├── src                 <- Source code for use in this project.
        │   └── aves        <- Main package
        │       ├── __init__.py <- Makes src a Python module
        │       │
        │       ├── data        <- Scripts to download or generate data
        │       │   │
        │       │   └── make_dataset.py
        │       │
        │       ├── features    <- Scripts to turn raw data into features for modeling
        │       │   └── build_features.py
        │       │
        │       ├── models      <- Scripts to train models and then use trained models to make
        │       │   │                 predictions
        │       │   ├── predict_model.py
        │       │   └── train_model.py
        │       │
        │       └── visualization  <- Scripts to create exploratory and results oriented visualizations
        │           └── visualize.py
        │
        └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org

.. ----------------------------------------------------------------------------

Project based on the
`cookiecutter data science project template <https://drivendata.github.io/cookiecutter-data-science/>`_


.. |License| image:: https://img.shields.io/badge/License-MIT-blue.svg
   :target: https://github.com/carnby/aves/blob/master/LICENSE
   :alt: Project License

