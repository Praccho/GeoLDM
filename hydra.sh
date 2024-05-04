#!/bin/sh
sudo apt install python3-pip
sudo pip3 install -r requirements.txt
sudo pip3 install satlaspretrain-models
python3 data/satlas_model.py