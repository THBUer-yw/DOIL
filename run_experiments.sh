#!/bin/bash

# Script to reproduce results
seed1=$RANDOM
seed2=$RANDOM
seed3=$RANDOM
seed4=$RANDOM
seed5=$RANDOM

CUDA_VISIBLE_DEVICES=0 python3 main.py --seed $seed1 --env "Ant-v2" --policy "TD3"&
CUDA_VISIBLE_DEVICES=0 python3 main.py --seed $seed2 --env "Ant-v2" --policy "TD3"&
CUDA_VISIBLE_DEVICES=0 python3 main.py --seed $seed3 --env "Ant-v2" --policy "TD3"&
CUDA_VISIBLE_DEVICES=0 python3 main.py --seed $seed4 --env "Ant-v2" --policy "TD3"&
CUDA_VISIBLE_DEVICES=0 python3 main.py --seed $seed5 --env "Ant-v2" --policy "TD3"

CUDA_VISIBLE_DEVICES=1 python3 main.py --seed $seed1 --env "HalfCheetah-v2" --policy "TD3"&
CUDA_VISIBLE_DEVICES=1 python3 main.py --seed $seed2 --env "HalfCheetah-v2" --policy "TD3"&
CUDA_VISIBLE_DEVICES=1 python3 main.py --seed $seed3 --env "HalfCheetah-v2" --policy "TD3"&
CUDA_VISIBLE_DEVICES=1 python3 main.py --seed $seed4 --env "HalfCheetah-v2" --policy "TD3"&
CUDA_VISIBLE_DEVICES=1 python3 main.py --seed $seed5 --env "HalfCheetah-v2" --policy "TD3"

CUDA_VISIBLE_DEVICES=2 python3 main.py --seed $seed1 --env "Hopper-v2" --policy "TD3"&
CUDA_VISIBLE_DEVICES=2 python3 main.py --seed $seed2 --env "Hopper-v2" --policy "TD3"&
CUDA_VISIBLE_DEVICES=2 python3 main.py --seed $seed3 --env "Hopper-v2" --policy "TD3"&
CUDA_VISIBLE_DEVICES=2 python3 main.py --seed $seed4 --env "Hopper-v2" --policy "TD3"&
CUDA_VISIBLE_DEVICES=2 python3 main.py --seed $seed5 --env "Hopper-v2" --policy "TD3"

CUDA_VISIBLE_DEVICES=3 python3 main.py --seed $seed1 --env "Reacher-v2" --policy "TD3" --start_steps 1000&
CUDA_VISIBLE_DEVICES=3 python3 main.py --seed $seed2 --env "Reacher-v2" --policy "TD3" --start_steps 1000&
CUDA_VISIBLE_DEVICES=3 python3 main.py --seed $seed3 --env "Reacher-v2" --policy "TD3" --start_steps 1000&
CUDA_VISIBLE_DEVICES=3 python3 main.py --seed $seed4 --env "Reacher-v2" --policy "TD3" --start_steps 1000&
CUDA_VISIBLE_DEVICES=3 python3 main.py --seed $seed5 --env "Reacher-v2" --policy "TD3" --start_steps 1000

CUDA_VISIBLE_DEVICES=0 python3 main.py --seed $seed1 --env "Walker2d-v2" --policy "TD3" --start_steps 5e4&
CUDA_VISIBLE_DEVICES=0 python3 main.py --seed $seed2 --env "Walker2d-v2" --policy "TD3" --start_steps 5e4&
CUDA_VISIBLE_DEVICES=0 python3 main.py --seed $seed3 --env "Walker2d-v2" --policy "TD3" --start_steps 5e4&
CUDA_VISIBLE_DEVICES=0 python3 main.py --seed $seed4 --env "Walker2d-v2" --policy "TD3" --start_steps 5e4&
CUDA_VISIBLE_DEVICES=0 python3 main.py --seed $seed5 --env "Walker2d-v2" --policy "TD3" --start_steps 5e4

CUDA_VISIBLE_DEVICES=1 python3 main.py --seed $seed1 --env "Humanoid-v2" --policy "TD3"&
CUDA_VISIBLE_DEVICES=1 python3 main.py --seed $seed2 --env "Humanoid-v2" --policy "TD3"&
CUDA_VISIBLE_DEVICES=1 python3 main.py --seed $seed3 --env "Humanoid-v2" --policy "TD3"&
CUDA_VISIBLE_DEVICES=1 python3 main.py --seed $seed4 --env "Humanoid-v2" --policy "TD3"&
CUDA_VISIBLE_DEVICES=1 python3 main.py --seed $seed5 --env "Humanoid-v2" --policy "TD3"

CUDA_VISIBLE_DEVICES=2 python3 main.py --seed $seed1 --env "InvertedPendulum-v2" --policy "TD3" --start_steps 1000&
CUDA_VISIBLE_DEVICES=2 python3 main.py --seed $seed2 --env "InvertedPendulum-v2" --policy "TD3" --start_steps 1000&
CUDA_VISIBLE_DEVICES=2 python3 main.py --seed $seed3 --env "InvertedPendulum-v2" --policy "TD3" --start_steps 1000&
CUDA_VISIBLE_DEVICES=2 python3 main.py --seed $seed4 --env "InvertedPendulum-v2" --policy "TD3" --start_steps 1000&
CUDA_VISIBLE_DEVICES=2 python3 main.py --seed $seed5 --env "InvertedPendulum-v2" --policy "TD3" --start_steps 1000

CUDA_VISIBLE_DEVICES=3 python3 main.py --seed $seed1 --env "InvertedDoublePendulum-v2" --policy "TD3" --start_steps 1000&
CUDA_VISIBLE_DEVICES=3 python3 main.py --seed $seed2 --env "InvertedDoublePendulum-v2" --policy "TD3" --start_steps 1000&
CUDA_VISIBLE_DEVICES=3 python3 main.py --seed $seed3 --env "InvertedDoublePendulum-v2" --policy "TD3" --start_steps 1000&
CUDA_VISIBLE_DEVICES=3 python3 main.py --seed $seed4 --env "InvertedDoublePendulum-v2" --policy "TD3" --start_steps 1000&
CUDA_VISIBLE_DEVICES=3 python3 main.py --seed $seed5 --env "InvertedDoublePendulum-v2" --policy "TD3" --start_steps 1000




