SHELL := /bin/bash

INPUT_DIR  := notebooks
NOTEBOOK ?= notebooks/*.ipynb
OUTPUT   ?= docs/lectures/

deploy:
	mkdocs gh-deploy

build:
	mkdocs build

convert:
	jupyter nbconvert --to markdown $(NOTEBOOK) --output-dir=$(OUTPUT)

all:
	build deploy