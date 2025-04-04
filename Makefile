.PHONY: .check_yesno owner-git owner-github clean create-env update-env delete-env install-package

#################################################################################
# GLOBALS                                                                       #
#################################################################################

SHELL=/bin/bash

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROFILE = default
PROJECT_NAME = AVES: Analysis, Visualization and Educational Support
PACKAGE_NAME = aves
ENV_NAME = aves
SRC_CODE_FOLDER = src/aves
PYTHON_INTERPRETER = python
CURRENT_ENV := $(CONDA_DEFAULT_ENV)

# Check for package managers in order of preference: mamba, micromamba, conda
MAMBA_CMD := $(shell command -v mamba 2>/dev/null)
MICROMAMBA_CMD := $(shell command -v micromamba 2>/dev/null)
CONDA_CMD := $(shell command -v conda 2>/dev/null)

# Determine which package manager to use
ifdef MAMBA_CMD
    PKG_MGR := $(MAMBA_CMD)
    HAS_PKG_MGR := True
    PKG_MGR_NAME := mamba
else ifdef MICROMAMBA_CMD
    PKG_MGR := $(MICROMAMBA_CMD)
    HAS_PKG_MGR := True
    PKG_MGR_NAME := micromamba
else ifdef CONDA_CMD
    PKG_MGR := $(CONDA_CMD)
    HAS_PKG_MGR := True
    PKG_MGR_NAME := conda
    @printf ">>> Utilizando conda, se sugiere instalar mamba\n"
else
    HAS_PKG_MGR := False
    PKG_MGR_NAME := none
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

# yes/no prompt
.check_yesno:
	@echo -n "If you cloned this repo, there is no need to do it. Are you sure? [y/N] " && read ans && [ $${ans:-N} = y ]

## Init version control. Do it only if you are the project owner
owner-git: .check_yesno
	@if [ ! -d ".git" ]; then \
	git init ; \
	versioneer install ; \
	git add . ; \
	git commit -m "INIT: Initial Commit" ; \
	fi

## Set github remote and push. Do it only if you are the project owner
owner-github: .check_yesno
	@if [ ! -d ".git" ]; then \
	make owner-git ; \
	fi

	@read -p "Enter remote repo https: " remote ; \
	git remote add origin $$remote ; \
	git push -u origin master ; \

## Delete all compiled Python files
clean:
	find . -name "*.pyc" -exec rm {} \;

## create conda environment
create-env:
ifeq (True,$(HAS_PKG_MGR))
	@printf ">>> Creating '$(ENV_NAME)' environment using $(PKG_MGR_NAME). This could take a few minutes ...\n\n"
	@PIP_NO_DEPS=1 $(PKG_MGR) env create --name $(ENV_NAME) --file environment.yml
	@printf ">>> Adding the project to the environment...\n\n"
else
	@printf ">>> No package manager found. Please install mamba, micromamba, or conda first.\n"
endif

## delete conda environment
delete-env:
ifeq (True,$(HAS_PKG_MGR))
	@printf ">>> Deleting '$(ENV_NAME)' environment using $(PKG_MGR_NAME). This could take a few minutes ...\n\n"
	@$(PKG_MGR) env remove --name $(ENV_NAME)
	@printf ">>> Done.\n\n"
else
	@printf ">>> No package manager found. Please install mamba, micromamba, or conda first.\n"
endif

## update conda environment
update-env:
ifeq (True,$(HAS_PKG_MGR))
	@printf ">>> Updating '$(ENV_NAME)' environment using $(PKG_MGR_NAME). This could take a few minutes ...\n\n"
	@PIP_NO_DEPS=1 $(PKG_MGR) env update --name $(ENV_NAME) --file environment.yml --prune
	@printf ">>> Updated.\n\n"
else
	@printf ">>> No package manager found. Please install mamba, micromamba, or conda first.\n"
endif

## install package in editable mode
install-package:
ifeq (True,$(HAS_PKG_MGR))
	$(PKG_MGR) run --name '$(ENV_NAME)' python -m pip install --editable . --config-settings editable_mode=compat
else
	@printf ">>> No package manager found. Please install mamba, micromamba, or conda first.\n"
endif

## uninstall package
uninstall-package:
ifeq (True,$(HAS_PKG_MGR))
	$(PKG_MGR) run --name '$(ENV_NAME)' python -m pip uninstall --yes '$(PACKAGE_NAME)'
else
	@printf ">>> No package manager found. Please install mamba, micromamba, or conda first.\n"
endif

## install jupyter notebook kernel
install-kernel:
ifeq (True,$(HAS_PKG_MGR))
	$(PKG_MGR) run --name '$(ENV_NAME)' python -m ipykernel install --user --name '$(ENV_NAME)' --display-name "Python ($(ENV_NAME))"
else
	@printf ">>> No package manager found. Please install mamba, micromamba, or conda first.\n"
endif

## download data from external sources
download-external:
	sh ./scripts/download_casen_2017.sh
	sh ./scripts/download_casen_2020.sh
	sh ./scripts/download_presidenciales_2021_primera_vuelta.sh
	sh ./scripts/download_others.sh
	sh ./scripts/download_plebiscito_2020.sh
	sh ./scripts/download_wiki2vec.sh

## download OSM data
download-osm:
	sh ./scripts/download_osm.sh

# generate documentation
html:
	$(MAKE) -C docs html

sentence-transformers:
ifeq (True,$(HAS_PKG_MGR))
	$(PKG_MGR) run --name '$(ENV_NAME)' pip install git+https://github.com/UKPLab/sentence-transformers.git
else
	@printf ">>> No package manager found. Please install mamba, micromamba, or conda first.\n"
endif

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := show-help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: show-help
show-help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
