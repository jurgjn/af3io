# Data and panels for the paper (see design/specs/tests.md):
#   make data    re-run the long tests, which record their tables in data/ (slow)
#   make panels  draw panels from the tables in data/

PYTHON ?= python
PANELS := $(patsubst %.py,%.pdf,$(wildcard panels/*.py))

.PHONY: all data panels
all: panels

panels: $(PANELS)

panels/%.pdf: panels/%.py $(wildcard data/*.parquet)
	$(PYTHON) $<

data:
	cd .. && $(PYTHON) -m pytest -m long
