#!/bin/bash

# Script to reproduce results
seed1=$RANDOM
seed2=$RANDOM
seed3=$RANDOM
seed4=$RANDOM
seed5=$RANDOM

for reward in {1,2,3,4,5,6,7,8}
do
  for seed in {$seed1,$seed2,$seed3,$seed4,$seed5}
  do
    CUDA_VISIBLE_DEVICES=1 python3 main.py --seed $seed1 --env "Ant-v2" --policy "TD3" --reward_type $reward&
    CUDA_VISIBLE_DEVICES=2 python3 main.py --seed $seed1 --env "HalfCheetah-v2" --policy "TD3" --reward_type $reward&
    CUDA_VISIBLE_DEVICES=3 python3 main.py --seed $seed1 --env "Hopper-v2" --policy "TD3" --reward_type $reward&
    CUDA_VISIBLE_DEVICES=1 python3 main.py --seed $seed1 --env "Reacher-v2" --policy "TD3" --reward_type $reward&
    CUDA_VISIBLE_DEVICES=2 python3 main.py --seed $seed1 --env "Walker2d-v2" --policy "TD3" --reward_type $reward&
    CUDA_VISIBLE_DEVICES=3 python3 main.py --seed $seed1 --env "BipedalWalker-v3" --policy "TD3" --reward_type $reward
  done
done


