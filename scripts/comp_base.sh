#!/bin/bash

python -m hdf5_compressor.compress_hdf5 --compress --dataset_dir ./data/sample --output_dir ./data/out --nproc 5 --quality 50 
