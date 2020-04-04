#!/bin/bash

MY_PYTHON="python"

# build datasets
cd data/

$MY_PYTHON raw.py

$MY_PYTHON mnist_rotations.py

cd ..

# model "GEM"
$MY_PYTHON main.py
