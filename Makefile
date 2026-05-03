SHELL := /bin/bash

INPUT_DIR  := notebooks
NOTEBOOK  ?= notebooks/StatisticsFoundations.ipynb
OUTPUT    ?= docs/lectures/

NOTEBOOKS := \
	notebooks/InstallationGuide.ipynb \
	notebooks/Introduction.ipynb \
	notebooks/StatisticsFoundations.ipynb \
	notebooks/MLConcepts.ipynb \
	notebooks/LinearRegression.ipynb \
	notebooks/ParameterEstimation.ipynb \
	notebooks/TimeSeries.ipynb \
	notebooks/TimeSeriesModels.ipynb

deploy:
	mkdocs gh-deploy

build:
	mkdocs build

convert:
	jupyter nbconvert --to markdown $(NOTEBOOK) --output-dir=$(OUTPUT)

convert-all:
	@for nb in $(NOTEBOOKS); do \
		jupyter nbconvert --to markdown $$nb --output-dir=$(OUTPUT); \
	done

all: build deploy