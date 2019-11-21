#!/bin/bash
# 
# 
# Author: alex
# Created Time: 2019年11月21日 星期四 15时13分33秒
sudo docker rm -f rk1808
sudo docker run -t -i --privileged --rm \
    --name rk1808 \
    -v /dev/bus/usb:/dev/bus/usb \
    -v "$PWD":/src \
    -w /src \
    rknn-toolkit:1.1.0 \
    /bin/bash
