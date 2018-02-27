#!/bin/bash

rsync -avzLC -e 'ssh -p 22' --progress ./ nvidia@jetson:~/catkin_ws/src/robofest_jtsn/ \
--exclude='upload.sh' --exclude='download.sh' --exclude='*.webm' --exclude='*.npy' --exclude='*.png' --exclude='*.sh' \
--exclude='data' --exclude='*.pyc'
