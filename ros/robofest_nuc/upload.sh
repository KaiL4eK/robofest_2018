#!/bin/bash

rsync -avzLC -e 'ssh -p 9992' --progress ./ user-sau-nuc@nuc:~/catkin_ws/src/robofest_nuc/ \
--exclude='upload.sh' --exclude='download.sh'
