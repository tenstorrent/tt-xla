# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Minimal makefile for Sphinx documentation.
# Assumes docs deps are installed — see docs/build_docs.sh for the
# install + build path used by CI and first-time local builds.

SPHINXOPTS    ?= -W
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
PORT          ?= 8888

.PHONY: help default clean html server

default: html

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	@rm -rf $(BUILDDIR)

html:
	@$(SPHINXBUILD) "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

server: html
	@echo "Navigate to: http://localhost:$(PORT)/index.html"
	@cd $(BUILDDIR)/html && python -m http.server $(PORT)
