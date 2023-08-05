#!/bin/bash

PROJECT_PATH=$(realpath $(dirname "${BASH_SOURCE[0]}"))

pip3 install --upgrade pip
pip3 install -r requirements.txt

git submodule init && git submodule update
ln -sf $PROJECT_PATH/3rdparty/FedCor/src $PROJECT_PATH/src/data/fedcor
