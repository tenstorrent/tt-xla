# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

.PHONY: build clean help

build:
	bash -c "source venv/activate && \
		cmake -G Ninja -B build && \
		cmake --build build"

clean:
	rm -rf build
	rm -rf venv/bin
	rm -rf venv/include
	rm -rf venv/lib
	rm -rf venv/share
	rm -rf venv/.lock
	rm -rf venv/pyvenv.cfg

help:
	@echo "Available targets:"
	@echo "  build  - Full build from scratch"
	@echo "  clean  - Remove build and venv/bin directories"
	@echo "  help   - Display this help message"
