#!/usr/bin/bash
CUDA_VISIBLE_DEVICES=$1 python3 baselinetest.py --ca --mode $2 --cs $3 --output_dir $4
