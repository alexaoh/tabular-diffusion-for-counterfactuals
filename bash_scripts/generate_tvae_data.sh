#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)" # Finds the directory we are working in.

# Generate data from TVAE. 
nice python $DIR/../TVAE/generate_data_AD.py -s 1234 -t True --compress-dims 256 512 --decompress-dims 256 512 -b 4096 -e 300 --loss-factor 1 --embedding-dim 512
# -e 30000
nice python $DIR/../TVAE/generate_data_CH.py -s 1234 -t True --compress-dims 256 512 --decompress-dims 256 512 -b 256 -e 300 --loss-factor 2 --embedding-dim 512
# -e 30000
nice python $DIR/../TVAE/generate_data_DI.py -s 1234 -t True --compress-dims 54 512 512 512 --decompress-dims 54 512 512 512 -b 256 -e 300 --loss-factor 6 --embedding-dim 512
# -e 20000
