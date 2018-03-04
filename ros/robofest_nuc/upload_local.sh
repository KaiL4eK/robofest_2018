#!/bin/bash

rsync -avzLC --progress ./ ~/catkin_ws/src/robofest_nuc/ \
--exclude='upload*.sh' --exclude='download.sh'
