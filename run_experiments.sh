#!/bin/bash

MY_PYTHON="python"

# build datasets
cd data/
cd raw/

$MY_PYTHON raw.py

cd ..

$MY_PYTHON mnist_rotations.py \
	--o mnist_rotations.pt\
	--seed 0 \
	--min_rot 0 \
	--max_rot 180 \
	--n_tasks 20

cd ..

# model "GEM"
$MY_PYTHON main.py
