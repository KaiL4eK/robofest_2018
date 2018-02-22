#!/bin/bash

rsync -rvzC -e 'ssh -p 22' --progress nvidia@jetson:~/catkin_ws/src/robofest_jtsn/ ./
