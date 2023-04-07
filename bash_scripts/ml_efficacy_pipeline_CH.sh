#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)" # Finds the directory we are working in.

############################### First we generate data from all three models for each of the five different seeds.
# Generate data from diffusion model. 
nice python $DIR/../train_diffusion.py -s 1234 -d CH -t True -g True -T 1000 -e 200 -b 128 --mlp-blocks 512 1024 1024 1024 1024 512 --dropout-ps 0 0 0 0 0 0 --early-stop-tolerance 10
nice python $DIR/../train_diffusion.py -s 4500 -d CH -t True -g True -T 1000 -e 200 -b 128 --mlp-blocks 512 1024 1024 1024 1024 512 --dropout-ps 0 0 0 0 0 0 --early-stop-tolerance 10
nice python $DIR/../train_diffusion.py -s 2018 -d CH -t True -g True -T 1000 -e 200 -b 128 --mlp-blocks 512 1024 1024 1024 1024 512 --dropout-ps 0 0 0 0 0 0 --early-stop-tolerance 10
nice python $DIR/../train_diffusion.py -s 1999 -d CH -t True -g True -T 1000 -e 200 -b 128 --mlp-blocks 512 1024 1024 1024 1024 512 --dropout-ps 0 0 0 0 0 0 --early-stop-tolerance 10
nice python $DIR/../train_diffusion.py -s 2023 -d CH -t True -g True -T 1000 -e 200 -b 128 --mlp-blocks 512 1024 1024 1024 1024 512 --dropout-ps 0 0 0 0 0 0 --early-stop-tolerance 10

# Generate data from TVAE. 
nice python $DIR/../TVAE/generate_data_CH.py -s 1234 --train --compress-dims 256 512 --decompress-dims 256 512 -b 128 -e 200 --loss-factor 2 --embedding-dim 256
nice python $DIR/../TVAE/generate_data_CH.py -s 4500 --train --compress-dims 256 512 --decompress-dims 256 512 -b 128 -e 200 --loss-factor 2 --embedding-dim 256
nice python $DIR/../TVAE/generate_data_CH.py -s 2018 --train --compress-dims 256 512 --decompress-dims 256 512 -b 128 -e 200 --loss-factor 2 --embedding-dim 256
nice python $DIR/../TVAE/generate_data_CH.py -s 1999 --train --compress-dims 256 512 --decompress-dims 256 512 -b 128 -e 200 --loss-factor 2 --embedding-dim 256
nice python $DIR/../TVAE/generate_data_CH.py -s 2023 --train --compress-dims 256 512 --decompress-dims 256 512 -b 128 -e 200 --loss-factor 2 --embedding-dim 256

# Generate data from MCCE trees. The hyperparameters are left as default in all these.
nice python $DIR/../mccepy/generate_data_CH.py -s 1234
nice python $DIR/../mccepy/generate_data_CH.py -s 4500
nice python $DIR/../mccepy/generate_data_CH.py -s 2018
nice python $DIR/../mccepy/generate_data_CH.py -s 1999
nice python $DIR/../mccepy/generate_data_CH.py -s 2023

############################### Next we run the ML efficacy script to calculate the metrics. 
nice python $DIR/../ML_efficacy_catBoost_CH.py -s 1234
nice python $DIR/../ML_efficacy_catBoost_CH.py -s 4500
nice python $DIR/../ML_efficacy_catBoost_CH.py -s 2018
nice python $DIR/../ML_efficacy_catBoost_CH.py -s 1999
nice python $DIR/../ML_efficacy_catBoost_CH.py -s 2023
