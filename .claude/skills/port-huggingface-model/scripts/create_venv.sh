#!/usr/bin/env bash

# Directory name for the virtual environment
VENV_DIR=".venv"

if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/activate" ]; then
    echo "Virtual environment already exists at '$VENV_DIR'. Skipping creation."
else
    echo "Creating Python 3 virtual environment at '$VENV_DIR'..."
    python3 -m venv "$VENV_DIR"

    if [ $? -eq 0 ]; then
        echo "Virtual environment created successfully."
    else
        echo "Error: Failed to create virtual environment." >&2
        exit 1
    fi
fi

source $VENV_DIR/bin/activate

pip install pre-commit
pre-commit install

pip install pytest

pip install pjrt-plugin-tt --extra-index-url https://pypi.eng.aws.tenstorrent.com/
