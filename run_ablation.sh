#!/bin/bash

# Script to reproduce results
seed1=$RANDOM
seed2=$RANDOM
seed3=$RANDOM
seed4=$RANDOM
seed5=$RANDOM

# hidden layers 0, 2, 6, 14
for seed in {$seed1,$seed2,$seed3,$seed4,$seed5}
do
  CUDA_VISIBLE_DEVICES=0 python main.py --seed $seed --env "Ant-v2" --hidden_layers 0&
  CUDA_VISIBLE_DEVICES=0 python main.py --seed $seed --env "BipedalWalker-v3" --hidden_layers 0&
  CUDA_VISIBLE_DEVICES=1 python main.py --seed $seed --env "HalfCheetah-v2" --hidden_layers 0&
  CUDA_VISIBLE_DEVICES=1 python main.py --seed $seed --env "Hopper-v2" --hidden_layers 0&
  CUDA_VISIBLE_DEVICES=2 python main.py --seed $seed --env "Reacher-v2" --hidden_layers 0&
  CUDA_VISIBLE_DEVICES=2 python main.py --seed $seed --env "Walker2d-v2" --hidden_layers 0
done




