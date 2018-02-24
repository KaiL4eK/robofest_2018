#!/bin/bash

python train.py -w weights_best.h5 -a -l 1e-5 2>&1 | tee learn.log
