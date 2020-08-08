# AVES: Analysis & Visualization -- Education & Support

Module to support my Information Visualization course.

## Development Setup

```sh
# Create conda environment, install dependencies on it and activate it
conda create --name aves --file environment.yml
conda activate aves

# Setup pre-commit and pre-push hooks
pre-commit install -t pre-commit
pre-commit install -t pre-push
```

Install the [Fira Sans set of fonts](https://bboxtype.com/typefaces/FiraSans/#!layout=specimen).

## Update dependencies

To add new dependencies or to update existing ones:

1. Add the name (and version if needed) to the list of dependencies in `environment.yml`
2. run `conda env update --name aves --file environment.yml  --prune`
3. Update the file `environment.lock.yml` by running `conda env export > environment.lock.yml`

## Credits

This package was created with the [BSCCNS/cookiecutter-data-science](https://github.com/BSCCNS/) project template, prepared by [Patricio Reyes](http://twitter.com/pareyesv).
