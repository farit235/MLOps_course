#!/usr/bin/env bash

set -ex

mypy src
black src --check
isort --check-only --profile black src
flake8 --ignore E203,W503,E231  src