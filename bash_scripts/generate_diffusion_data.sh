#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)" # Finds the directory we are working in.

# Generate data from diffusion model. This models a density conditional to the response, p(X|y)
nice python $DIR/../train_diffusion.py -s 1234 -d AD -T 1000 -e 200 -b 256 --mlp-blocks 256 1024 1024 1024 1024 256 --dropout-ps 0 0 0 0 0 0 --early-stop-tolerance 10 --is-class-cond --dont-plot-losses
nice python $DIR/../train_diffusion.py -s 1234 -d CH -T 1000 -e 200 -b 128 --mlp-blocks 512 1024 1024 1024 1024 512 --dropout-ps 0 0 0 0 0 0 --early-stop-tolerance 10 --is-class-cond --dont-plot-losses
nice python $DIR/../train_diffusion.py -s 1234 -d DI -T 1000 -e 200 -b 64 --mlp-blocks 128 512 --dropout-ps 0 0 --early-stop-tolerance 10 --is-class-cond --dont-plot-losses

# Generate data from diffusion model. This models the joint density p(X,y).
nice python $DIR/../train_diffusion.py -s 1234 -d AD -T 1000 -e 200 -b 256 --mlp-blocks 256 1024 1024 1024 1024 256 --dropout-ps 0 0 0 0 0 0 --early-stop-tolerance 10 --dont-plot-losses
nice python $DIR/../train_diffusion.py -s 1234 -d CH -T 1000 -e 200 -b 128 --mlp-blocks 512 1024 1024 1024 1024 512 --dropout-ps 0 0 0 0 0 0 --early-stop-tolerance 10 --dont-plot-losses
nice python $DIR/../train_diffusion.py -s 1234 -d DI -T 1000 -e 200 -b 64 --mlp-blocks 128 512 --dropout-ps 0 0 --early-stop-tolerance 10 --dont-plot-losses
