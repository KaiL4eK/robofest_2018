#!/bin/bash

rsync -rvzC -e 'ssh -p 9992' --progress user-sau-nuc@nuc:~/catkin_ws/src/robofest_nuc/ ./
