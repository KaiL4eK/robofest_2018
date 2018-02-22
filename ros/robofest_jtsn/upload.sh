#!/bin/bash

rsync -avzC -e 'ssh -p 22' --progress ./ nvidia@jetson:~/catkin_ws/src/robofest_jtsn/ \
--exclude='upload.sh' --exclude='download.sh'
