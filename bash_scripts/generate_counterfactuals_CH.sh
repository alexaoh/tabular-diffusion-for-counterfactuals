#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)" # Finds the directory we are working in.

# First we find the different factuals. 
nice python $DIR/../find_factuals_CH.py -s 1234 --num-factuals 100 --train --save
nice python $DIR/../find_factuals_CH.py -s 4500 --num-factuals 100 --train --save
nice python $DIR/../find_factuals_CH.py -s 2018 --num-factuals 100 --train --save
nice python $DIR/../find_factuals_CH.py -s 1999 --num-factuals 100 --train --save
nice python $DIR/../find_factuals_CH.py -s 2023 --num-factuals 100 --train --save

# Second we generate counterfactuals from MCCE.
nice python $DIR/../generate_counterfactuals/CH_MCCE_generate_counterfactuals.py -s 1234
nice python $DIR/../generate_counterfactuals/CH_MCCE_generate_counterfactuals.py -s 4500
nice python $DIR/../generate_counterfactuals/CH_MCCE_generate_counterfactuals.py -s 2018
nice python $DIR/../generate_counterfactuals/CH_MCCE_generate_counterfactuals.py -s 1999
nice python $DIR/../generate_counterfactuals/CH_MCCE_generate_counterfactuals.py -s 2023

# Find possible counterfactuals from TVAE, after training the model. 
nice python $DIR/../TVAE/generate_data_CH.py -s 1234 --num-samples 1000000 --savename "K10000"
nice python $DIR/../TVAE/generate_data_CH.py -s 4500 --num-samples 1000000 --savename "K10000"
nice python $DIR/../TVAE/generate_data_CH.py -s 2018 --num-samples 1000000 --savename "K10000"
nice python $DIR/../TVAE/generate_data_CH.py -s 1999 --num-samples 1000000 --savename "K10000"
nice python $DIR/../TVAE/generate_data_CH.py -s 2023 --num-samples 1000000 --savename "K10000"

# Then generate counterfactuals from TVAE. 
nice python $DIR/../generate_counterfactuals/CH_TVAE_generate_counterfactuals.py -s 1234
nice python $DIR/../generate_counterfactuals/CH_TVAE_generate_counterfactuals.py -s 4500
nice python $DIR/../generate_counterfactuals/CH_TVAE_generate_counterfactuals.py -s 2018
nice python $DIR/../generate_counterfactuals/CH_TVAE_generate_counterfactuals.py -s 1999
nice python $DIR/../generate_counterfactuals/CH_TVAE_generate_counterfactuals.py -s 2023

# Find possible counterfactuals from TabDDPM, after training the model. 

nice python $DIR/../train_diffusion.py -s 1234 -d CH -g True -T 1000 -e 200 -b 128 --mlp-blocks 512 1024 1024 1024 1024 512 --dropout-ps 0 0 0 0 0 0 --early-stop-tolerance 10 --num-samples 1000000 --savename "TabDDPM_K10000_"
nice python $DIR/../train_diffusion.py -s 4500 -d CH -g True -T 1000 -e 200 -b 128 --mlp-blocks 512 1024 1024 1024 1024 512 --dropout-ps 0 0 0 0 0 0 --early-stop-tolerance 10 --num-samples 1000000 --savename "TabDDPM_K10000_"
nice python $DIR/../train_diffusion.py -s 2018 -d CH -g True -T 1000 -e 200 -b 128 --mlp-blocks 512 1024 1024 1024 1024 512 --dropout-ps 0 0 0 0 0 0 --early-stop-tolerance 10 --num-samples 1000000 --savename "TabDDPM_K10000_"
nice python $DIR/../train_diffusion.py -s 1999 -d CH -g True -T 1000 -e 200 -b 128 --mlp-blocks 512 1024 1024 1024 1024 512 --dropout-ps 0 0 0 0 0 0 --early-stop-tolerance 10 --num-samples 1000000 --savename "TabDDPM_K10000_"
nice python $DIR/../train_diffusion.py -s 2023 -d CH -g True -T 1000 -e 200 -b 128 --mlp-blocks 512 1024 1024 1024 1024 512 --dropout-ps 0 0 0 0 0 0 --early-stop-tolerance 10 --num-samples 1000000 --savename "TabDDPM_K10000_"

# Then generate counterfactuals from TabDDPM.
nice python $DIR/../generate_counterfactuals/CH_TabDDPM_generate_counterfactuals.py -s 1234
nice python $DIR/../generate_counterfactuals/CH_TabDDPM_generate_counterfactuals.py -s 4500
nice python $DIR/../generate_counterfactuals/CH_TabDDPM_generate_counterfactuals.py -s 2018
nice python $DIR/../generate_counterfactuals/CH_TabDDPM_generate_counterfactuals.py -s 1999
nice python $DIR/../generate_counterfactuals/CH_TabDDPM_generate_counterfactuals.py -s 2023
 
# After this we use the "make_tables.py"-script to make the tables that appear in the report. 
