#!/bin/sh

python3 main.py --base configs/ddpm.yaml -- train

while true
do
    # Connect to the remote server, execute the training script, and capture the output
    output=$(ssh $HOST "bash $TRAINING_SCRIPT_PATH")

    # Clear the local terminal
    clear

    # Print the latest output
    echo "$output"

    # Sleep for specified interval before the next update
    sleep $REFRESH_INTERVAL
done