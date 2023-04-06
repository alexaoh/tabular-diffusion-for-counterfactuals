#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)" # Finds the directory we are working in.

# Generate data from diffusion model. 
nice python $DIR/../train_diffusion.py -s 1234 -d AD -t True -g True -T 1000 -e 200 -b 256 --mlp-blocks 256 1024 1024 1024 1024 256 --dropout-ps 0 0 0 0 0 0 --early-stop-tolerance 10
nice python $DIR/../train_diffusion.py -s 1234 -d CH -t True -g True -T 1000 -e 200 -b 128 --mlp-blocks 512 1024 1024 1024 1024 512 --dropout-ps 0 0 0 0 0 0 --early-stop-tolerance 10
nice python $DIR/../train_diffusion.py -s 1234 -d DI -t True -g True -T 1000 -e 200 -b 64 --mlp-blocks 128 512 --dropout-ps 0 0 --early-stop-tolerance 10

# Generate possible counterfactuals after training the model as well. 
nice python $DIR/../train_diffusion.py -s 1234 -d AD -g True -T 1000 -e 200 -b 256 --mlp-blocks 256 1024 1024 1024 1024 256 --dropout-ps 0 0 0 0 0 0 --early-stop-tolerance 10 --num-samples 1000000 --savename "TabDDPM_K10000_"
nice python $DIR/../train_diffusion.py -s 1234 -d CH -g True -T 1000 -e 200 -b 128 --mlp-blocks 512 1024 1024 1024 1024 512 --dropout-ps 0 0 0 0 0 0 --early-stop-tolerance 10 --num-samples 1000000 --savename "TabDDPM_K10000_"
nice python $DIR/../train_diffusion.py -s 1234 -d DI -g True -T 1000 -e 200 -b 64 --mlp-blocks 128 512 --dropout-ps 0 0 --early-stop-tolerance 10 --num-samples 1000000 --savename "TabDDPM_K10000_"
