#!/bin/bash

PROJECT_PATH=$(realpath $(dirname "${BASH_SOURCE[0]}"))

pip3 install --upgrade pip
pip3 install -r requirements.txt

git submodule init && git submodule update
cd 3rdparty/FedCor && pip3 install -e .
cd ...

