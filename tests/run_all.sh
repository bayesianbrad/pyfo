#!/usr/bin/env bash
set -e

echo "Running compiler tests"
python test_compiler.py

echo "Running inference tests"
python test_inference.py

echo "Running util tests"
python test_utils.py

