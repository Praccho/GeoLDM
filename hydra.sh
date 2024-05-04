#!/bin/sh
sudo apt install python3-pip
sudo pip3 uninstall -r requirements.txt
sudo pip3 uninstall satlaspretrain-models
python3 data/satlas_model.py