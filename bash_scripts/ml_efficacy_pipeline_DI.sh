#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)" # Finds the directory we are working in.

############################### First we generate data from all three models for each of the five different seeds.
# Generate data from diffusion model. This generates from p(X|y).
nice python $DIR/../train_diffusion.py -s 1234 -d DI -T 1000 -e 200 -b 64 --mlp-blocks 128 512 --dropout-ps 0 0 --early-stop-tolerance 10 --is-class-cond --dont-plot-losses
nice python $DIR/../train_diffusion.py -s 4500 -d DI -T 1000 -e 200 -b 64 --mlp-blocks 128 512 --dropout-ps 0 0 --early-stop-tolerance 10 --is-class-cond --dont-plot-losses
nice python $DIR/../train_diffusion.py -s 2018 -d DI -T 1000 -e 200 -b 64 --mlp-blocks 128 512 --dropout-ps 0 0 --early-stop-tolerance 10 --is-class-cond --dont-plot-losses
nice python $DIR/../train_diffusion.py -s 1999 -d DI -T 1000 -e 200 -b 64 --mlp-blocks 128 512 --dropout-ps 0 0 --early-stop-tolerance 10 --is-class-cond --dont-plot-losses
nice python $DIR/../train_diffusion.py -s 2023 -d DI -T 1000 -e 200 -b 64 --mlp-blocks 128 512 --dropout-ps 0 0 --early-stop-tolerance 10 --is-class-cond --dont-plot-losses

# Generate data from diffusion model. This generates from p(X,y).
nice python $DIR/../train_diffusion.py -s 1234 -d DI -T 1000 -e 200 -b 64 --mlp-blocks 128 512 --dropout-ps 0 0 --early-stop-tolerance 10 --dont-plot-losses
nice python $DIR/../train_diffusion.py -s 4500 -d DI -T 1000 -e 200 -b 64 --mlp-blocks 128 512 --dropout-ps 0 0 --early-stop-tolerance 10 --dont-plot-losses
nice python $DIR/../train_diffusion.py -s 2018 -d DI -T 1000 -e 200 -b 64 --mlp-blocks 128 512 --dropout-ps 0 0 --early-stop-tolerance 10 --dont-plot-losses
nice python $DIR/../train_diffusion.py -s 1999 -d DI -T 1000 -e 200 -b 64 --mlp-blocks 128 512 --dropout-ps 0 0 --early-stop-tolerance 10 --dont-plot-losses
nice python $DIR/../train_diffusion.py -s 2023 -d DI -T 1000 -e 200 -b 64 --mlp-blocks 128 512 --dropout-ps 0 0 --early-stop-tolerance 10 --dont-plot-losses

# Generate data from TVAE. 
nice python $DIR/../TVAE/generate_data_DI.py -s 1234 --train --compress-dims 128 128 --decompress-dims 128 128 -b 64 -e 200 --loss-factor 2 --embedding-dim 128
nice python $DIR/../TVAE/generate_data_DI.py -s 4500 --train --compress-dims 128 128 --decompress-dims 128 128 -b 64 -e 200 --loss-factor 2 --embedding-dim 128
nice python $DIR/../TVAE/generate_data_DI.py -s 2018 --train --compress-dims 128 128 --decompress-dims 128 128 -b 64 -e 200 --loss-factor 2 --embedding-dim 128
nice python $DIR/../TVAE/generate_data_DI.py -s 1999 --train --compress-dims 128 128 --decompress-dims 128 128 -b 64 -e 200 --loss-factor 2 --embedding-dim 128
nice python $DIR/../TVAE/generate_data_DI.py -s 2023 --train --compress-dims 128 128 --decompress-dims 128 128 -b 64 -e 200 --loss-factor 2 --embedding-dim 128

# Generate data from MCCE trees. The hyperparameters are left as default in all these.
nice python $DIR/../mccepy/generate_data_DI.py -s 1234
nice python $DIR/../mccepy/generate_data_DI.py -s 4500
nice python $DIR/../mccepy/generate_data_DI.py -s 2018
nice python $DIR/../mccepy/generate_data_DI.py -s 1999
nice python $DIR/../mccepy/generate_data_DI.py -s 2023

############################### Next we run the ML efficacy script to calculate the metrics. 
nice python $DIR/../ML_efficacy_catBoost_DI.py -s 1234
nice python $DIR/../ML_efficacy_catBoost_DI.py -s 4500
nice python $DIR/../ML_efficacy_catBoost_DI.py -s 2018
nice python $DIR/../ML_efficacy_catBoost_DI.py -s 1999
nice python $DIR/../ML_efficacy_catBoost_DI.py -s 2023
