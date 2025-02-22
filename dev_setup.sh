#!/bin/bash

# Uninstall existing installation
pip uninstall -y fast-cody

# Clean build artifacts
rm -rf build/
rm -rf dist/
rm -rf *.egg-info/
rm -rf src/*.egg-info/

# Install in editable mode using setup.py directly
python setup.py develop