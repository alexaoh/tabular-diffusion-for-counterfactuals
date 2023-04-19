#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)" # Finds the directory we are working in.

# First we find the different factuals. 
nice python $DIR/../find_factuals_AD.py -s 1234 --num-factuals 100 --train --save
nice python $DIR/../find_factuals_AD.py -s 4500 --num-factuals 100 --train --save
nice python $DIR/../find_factuals_AD.py -s 2018 --num-factuals 100 --train --save
nice python $DIR/../find_factuals_AD.py -s 1999 --num-factuals 100 --train --save
nice python $DIR/../find_factuals_AD.py -s 2023 --num-factuals 100 --train --save

# Second we generate counterfactuals from MCCE.
nice python $DIR/../generate_counterfactuals/AD_MCCE_generate_counterfactuals.py -s 1234
nice python $DIR/../generate_counterfactuals/AD_MCCE_generate_counterfactuals.py -s 4500
nice python $DIR/../generate_counterfactuals/AD_MCCE_generate_counterfactuals.py -s 2018
nice python $DIR/../generate_counterfactuals/AD_MCCE_generate_counterfactuals.py -s 1999
nice python $DIR/../generate_counterfactuals/AD_MCCE_generate_counterfactuals.py -s 2023

# Find possible counterfactuals from TVAE, after training the model. 
nice python $DIR/../TVAE/generate_data_AD.py -s 1234 --num-samples 1000000 --savename "K10000"
nice python $DIR/../TVAE/generate_data_AD.py -s 4500 --num-samples 1000000 --savename "K10000"
nice python $DIR/../TVAE/generate_data_AD.py -s 2018 --num-samples 1000000 --savename "K10000"
nice python $DIR/../TVAE/generate_data_AD.py -s 1999 --num-samples 1000000 --savename "K10000"
nice python $DIR/../TVAE/generate_data_AD.py -s 2023 --num-samples 1000000 --savename "K10000"

# Then generate counterfactuals from TVAE. 
nice python $DIR/../generate_counterfactuals/AD_TVAE_generate_counterfactuals.py -s 1234
nice python $DIR/../generate_counterfactuals/AD_TVAE_generate_counterfactuals.py -s 4500
nice python $DIR/../generate_counterfactuals/AD_TVAE_generate_counterfactuals.py -s 2018
nice python $DIR/../generate_counterfactuals/AD_TVAE_generate_counterfactuals.py -s 1999
nice python $DIR/../generate_counterfactuals/AD_TVAE_generate_counterfactuals.py -s 2023

# Find possible counterfactuals from TabDDPM, after training the model. This finds possible counterfactuals from p(X|y).
nice python $DIR/../train_diffusion.py -s 1234 -d AD -T 1000 -e 200 -b 256 --mlp-blocks 256 1024 1024 1024 1024 256 --dropout-ps 0 0 0 0 0 0 --early-stop-tolerance 10 --is-class-cond --dont-plot-losses --savename "TabDDPM_K10000_" --num-samples 1000000 --dont-train
nice python $DIR/../train_diffusion.py -s 4500 -d AD -T 1000 -e 200 -b 256 --mlp-blocks 256 1024 1024 1024 1024 256 --dropout-ps 0 0 0 0 0 0 --early-stop-tolerance 10 --is-class-cond --dont-plot-losses --savename "TabDDPM_K10000_" --num-samples 1000000 --dont-train
nice python $DIR/../train_diffusion.py -s 2018 -d AD -T 1000 -e 200 -b 256 --mlp-blocks 256 1024 1024 1024 1024 256 --dropout-ps 0 0 0 0 0 0 --early-stop-tolerance 10 --is-class-cond --dont-plot-losses --savename "TabDDPM_K10000_" --num-samples 1000000 --dont-train
nice python $DIR/../train_diffusion.py -s 1999 -d AD -T 1000 -e 200 -b 256 --mlp-blocks 256 1024 1024 1024 1024 256 --dropout-ps 0 0 0 0 0 0 --early-stop-tolerance 10 --is-class-cond --dont-plot-losses --savename "TabDDPM_K10000_" --num-samples 1000000 --dont-train
nice python $DIR/../train_diffusion.py -s 2023 -d AD -T 1000 -e 200 -b 256 --mlp-blocks 256 1024 1024 1024 1024 256 --dropout-ps 0 0 0 0 0 0 --early-stop-tolerance 10 --is-class-cond --dont-plot-losses --savename "TabDDPM_K10000_" --num-samples 1000000 --dont-train

# Find possible counterfactuals from TabDDPM, after training the model. This finds possible counterfactuals from p(X,y).
nice python $DIR/../train_diffusion.py -s 1234 -d AD -T 1000 -e 200 -b 256 --mlp-blocks 256 1024 1024 1024 1024 256 --dropout-ps 0 0 0 0 0 0 --early-stop-tolerance 10 --dont-plot-losses --savename "TabDDPM_K10000_joint_" --num-samples 1000000 --dont-train
nice python $DIR/../train_diffusion.py -s 4500 -d AD -T 1000 -e 200 -b 256 --mlp-blocks 256 1024 1024 1024 1024 256 --dropout-ps 0 0 0 0 0 0 --early-stop-tolerance 10 --dont-plot-losses --savename "TabDDPM_K10000_joint_" --num-samples 1000000 --dont-train
nice python $DIR/../train_diffusion.py -s 2018 -d AD -T 1000 -e 200 -b 256 --mlp-blocks 256 1024 1024 1024 1024 256 --dropout-ps 0 0 0 0 0 0 --early-stop-tolerance 10 --dont-plot-losses --savename "TabDDPM_K10000_joint_" --num-samples 1000000 --dont-train
nice python $DIR/../train_diffusion.py -s 1999 -d AD -T 1000 -e 200 -b 256 --mlp-blocks 256 1024 1024 1024 1024 256 --dropout-ps 0 0 0 0 0 0 --early-stop-tolerance 10 --dont-plot-losses --savename "TabDDPM_K10000_joint_" --num-samples 1000000 --dont-train
nice python $DIR/../train_diffusion.py -s 2023 -d AD -T 1000 -e 200 -b 256 --mlp-blocks 256 1024 1024 1024 1024 256 --dropout-ps 0 0 0 0 0 0 --early-stop-tolerance 10 --dont-plot-losses --savename "TabDDPM_K10000_joint_" --num-samples 1000000 --dont-train

# Then generate counterfactuals from TabDDPM. This generates counterfactuals from p(X|y).
nice python $DIR/../generate_counterfactuals/AD_TabDDPM_generate_counterfactuals.py -s 1234
nice python $DIR/../generate_counterfactuals/AD_TabDDPM_generate_counterfactuals.py -s 4500
nice python $DIR/../generate_counterfactuals/AD_TabDDPM_generate_counterfactuals.py -s 2018
nice python $DIR/../generate_counterfactuals/AD_TabDDPM_generate_counterfactuals.py -s 1999
nice python $DIR/../generate_counterfactuals/AD_TabDDPM_generate_counterfactuals.py -s 2023

# Then generate counterfactuals from TabDDPM. This generates counterfactuals from p(X,y).
nice python $DIR/../generate_counterfactuals/AD_TabDDPM_generate_counterfactuals.py -s 1234 --joint
nice python $DIR/../generate_counterfactuals/AD_TabDDPM_generate_counterfactuals.py -s 4500 --joint
nice python $DIR/../generate_counterfactuals/AD_TabDDPM_generate_counterfactuals.py -s 2018 --joint
nice python $DIR/../generate_counterfactuals/AD_TabDDPM_generate_counterfactuals.py -s 1999 --joint
nice python $DIR/../generate_counterfactuals/AD_TabDDPM_generate_counterfactuals.py -s 2023 --joint
 
# After this we use the "make_tables.py"-script to make the tables that appear in the report. 
