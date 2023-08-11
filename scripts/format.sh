#!/usr/bin/env bash

set -ex

black src
isort --profile black src