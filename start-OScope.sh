#!/usr/bin/env bash
### Run this from ssh root, the display (from USB-C DP Alt) will automatically show feed ###
export DISPLAY=:0
export XAUTHORITY=/home/radxa/.Xauthority

cd /home/radxa/Documents/OScope
source venv2/bin/activate
exec python3 ./hpp02.py --camera /dev/video0