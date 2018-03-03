#!/bin/bash

rsync -avzLPh -e 'ssh -p 9992' userquadro@uni1:~/keras_NN/weights_best.h5 ./
