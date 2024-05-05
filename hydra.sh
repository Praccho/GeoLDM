#!/bin/sh

python3 main.py --base configs/ddpm.yaml --train --logdir logs

while true
do
    # Clear the local terminal
    clear
    find *.out || cat
done