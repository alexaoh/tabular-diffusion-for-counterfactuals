#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)" # Finds the directory we are working in.

# Generate data from trees in MCCE. The hyperparameters are left as default in all of these for now. 
nice python $DIR/../mccepy/generate_data_AD.py -s 1234 
nice python $DIR/../mccepy/generate_data_CH.py -s 1234
nice python $DIR/../mccepy/generate_data_DI.py -s 1234
