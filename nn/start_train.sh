#!/bin/bash

python train.py -w weights_best.h5 2>&1 | tee learn.log
