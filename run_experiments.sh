#!/bin/bash

MY_PYTHON="python"

# build datasets
cd data/
cd raw/

$MY_PYTHON raw.py

cd ..

$MY_PYTHON mnist_rotations.py

cd ..

# model "GEM"
$MY_PYTHON main.py
