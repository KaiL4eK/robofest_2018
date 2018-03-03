#!/bin/bash

NUC=nuc_ros
rsync -avzLC -e 'ssh -p 9992' --progress ./ user-sau-nuc@$NUC:~/catkin_ws/src/robofest_nuc/ \
--exclude='upload.sh' --exclude='download.sh'
