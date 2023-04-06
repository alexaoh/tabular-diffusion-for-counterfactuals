#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)" # Finds the directory we are working in.

# Generate data from TVAE. 
nice python $DIR/../TVAE/generate_data_AD.py -s 1234 -t True --compress-dims 128 128 --decompress-dims 128 128 -b 256 -e 200 --loss-factor 2 --embedding-dim 128
# Default TVAE hyperparameters are used above, except for batch-size and epochs, which is set to match our diffusion models.
nice python $DIR/../TVAE/generate_data_CH.py -s 1234 -t True --compress-dims 256 512 --decompress-dims 256 512 -b 256 -e 200 --loss-factor 2 --embedding-dim 256
# Parameters found in TabDDPM are used above. 
nice python $DIR/../TVAE/generate_data_DI.py -s 1234 -t True --compress-dims 128 128 --decompress-dims 128 128 -b 256 -e 200 --loss-factor 2 --embedding-dim 128
# Default TVAE hyperparameters are used above, except for batch-size and epochs, which is set to match our diffusion models.

# Generate possible counterfactuals after training the model as well. 
