#!/usr/bin/env bash

python -m pip install -r requirements.txt
python -m pip install -r requirements-test.txt
python -m pip install -r requirements-doc.txt

python -m pip install pre-commit
pre-commit install --install-hooks

python -m pip install -e .
