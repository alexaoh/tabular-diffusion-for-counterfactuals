#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)" # Finds the directory we are working in.

############################### First we generate data from all three models for each of the five different seeds.
# Generate data from diffusion model. 
nice python $DIR/../train_diffusion.py -s 1234 -d DI -t True -g True # Still need to add hyperparameters here!
nice python $DIR/../train_diffusion.py -s 4500 -d DI -t True -g True # Still need to addd hyperparameters here!
nice python $DIR/../train_diffusion.py -s 2018 -d DI -t True -g True # Still need to add hyperparameters here!
nice python $DIR/../train_diffusion.py -s 1999 -d DI -t True -g True # Still need to add hyperparameters here!
nice python $DIR/../train_diffusion.py -s 2023 -d DI -t True -g True # Still need to add hyperparameters here!
# Find these hyperparameters from TabDDPM code!

# Generate data from TVAE. What about the hyperparameters here? Make sure epochs and batch_size (at least) are the same!
nice python $DIR/../TVAE/generate_data_DI.py -s 1234 -t True # Hyperparameters?
nice python $DIR/../TVAE/generate_data_DI.py -s 4500 -t True # Hyperparameters?
nice python $DIR/../TVAE/generate_data_DI.py -s 2018 -t True # Hyperparameters?
nice python $DIR/../TVAE/generate_data_DI.py -s 1999 -t True # Hyperparameters?
nice python $DIR/../TVAE/generate_data_DI.py -s 2023 -t True # Hyperparameters?

# Generate data from MCCE trees. What about hyperparameters here? Something I need to check out!
nice python $DIR/../mccepy/generate_data_DI.py -s 1234
nice python $DIR/../mccepy/generate_data_DI.py -s 4500
nice python $DIR/../mccepy/generate_data_DI.py -s 2018
nice python $DIR/../mccepy/generate_data_DI.py -s 1999
nice python $DIR/../mccepy/generate_data_DI.py -s 2023

############################### Next we run the ML efficacy script to calculate the metrics. 
nice python $DIR/../ML_efficacy_catBoost_DI.py -s 12345
nice python $DIR/../ML_efficacy_catBoost_DI.py -s 4500
nice python $DIR/../ML_efficacy_catBoost_DI.py -s 2018
nice python $DIR/../ML_efficacy_catBoost_DI.py -s 1999
nice python $DIR/../ML_efficacy_catBoost_DI.py -s 2023

############################### Finally, we average the values over each of the seeds. 
#nice python $DIR/../make_tables.py 
