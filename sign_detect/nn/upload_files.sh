#!/bin/bash

rsync -avzLPh -e 'ssh -p 9992' \
	train.py data.py net.py start_train.sh npy \
	userquadro@uni:~/keras_NN/
