#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)" # Finds the directory we are working in.

############################### First we generate data from all three models for each of the five different seeds.
# Generate data from diffusion model. 
nice python $DIR/../train_diffusion.py -s 1234 -d CH -t True -g True # Still need to add hyperparameters here!
nice python $DIR/../train_diffusion.py -s 4500 -d CH -t True -g True # Still need to add hyperparameters here!
nice python $DIR/../train_diffusion.py -s 2018 -d CH -t True -g True # Still need to add hyperparameters here!
nice python $DIR/../train_diffusion.py -s 1999 -d CH -t True -g True # Still need to add hyperparameters here!
nice python $DIR/../train_diffusion.py -s 2023 -d CH -t True -g True # Still need to add hyperparameters here!
# Find these hyperparameters from TabDDPM code!

# Generate data from TVAE. What about the hyperparameters here? Make sure epochs and batch_size (at least) are the same!
nice python $DIR/../TVAE/generate_data_CH.py -s 1234 -t True # Hyperparameters?
nice python $DIR/../TVAE/generate_data_CH.py -s 4500 -t True # Hyperparameters?
nice python $DIR/../TVAE/generate_data_CH.py -s 2018 -t True # Hyperparameters?
nice python $DIR/../TVAE/generate_data_CH.py -s 1999 -t True # Hyperparameters?
nice python $DIR/../TVAE/generate_data_CH.py -s 2023 -t True # Hyperparameters?

# Generate data from MCCE trees. What about hyperparameters here? Something I need to check out!
nice python $DIR/../mccepy/generate_data_CH.py -s 1234 -t True # Hyperparameters?
nice python $DIR/../mccepy/generate_data_CH.py -s 4500 -t True # Hyperparameters?
nice python $DIR/../mccepy/generate_data_CH.py -s 2018 -t True # Hyperparameters?
nice python $DIR/../mccepy/generate_data_CH.py -s 1999 -t True # Hyperparameters?
nice python $DIR/../mccepy/generate_data_CH.py -s 2023 -t True # Hyperparameters?

############################### Next we run the ML efficacy script to calculate the metrics. 
nice python $DIR/../ml_efficacy_catBoost_CH.py -s 12345


############################### Finally, we average the values over each of the seeds. 
nice python $DIR/../make_descriptive_tables.py 
