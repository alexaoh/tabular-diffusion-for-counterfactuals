#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)" # Finds the directory we are working in.

# Generate data for diffusion model. 
nice python $DIR/../train_diffusion.py -s 1234 -d AD -t True -g True # Still need to add hyperparameters here!
nice python $DIR/../train_diffusion.py -s 1234 -d CH -t True -g True # Still need to add hyperparameters here!
nice python $DIR/../train_diffusion.py -s 1234 -d DI -t True -g True # Still need to add hyperparameters here!
# Find the hyperparameters in the code from TabDDPM!
