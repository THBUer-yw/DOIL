import numpy as np
import torch
import gym
import argparse
import os
import shutil
from tensorboardX import SummaryWriter

import utils
import TD3
import OurDDPG
import DDPG


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="HalfCheetah-v2")          # OpenAI gym environment name
    parser.add_argument("--seed", default=1979, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--eval_episodes", default=10, type=int)       # How often (time steps) we evaluate
    args = parser.parse_args()


    env = gym.make(args.env)
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action
    }

    if args.policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        policy = TD3.TD3(**kwargs)
    elif args.policy == "OurDDPG":
        policy = OurDDPG.DDPG(**kwargs)
    elif args.policy == "DDPG":
        policy = DDPG.DDPG(**kwargs)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_file_name = args.env+"_"+"seed_"+str(args.seed)
    policy.load(f"./models/{model_file_name}", device)

    data_file_name = "trajs_"+args.env.lower()[:-3]+".pt"
    if os.path.exists("./gail_experts/"):
        shutil.rmtree("./gail_experts/")
    os.makedirs("./gail_experts/")

    if args.env == "Ant-v2" or "HalfCheetah-v2" or "Hopper-v2" or "Walker2d-v2" or "Humanoid-v2":
        max_lengh = 1000
    if args.env == "Reacher-v2":
        max_lengh = 50

    states = torch.zeros([args.eval_episodes, max_lengh, kwargs["state_dim"]])
    next_states = torch.zeros([args.eval_episodes, max_lengh, kwargs["state_dim"]])
    actions = torch.zeros([args.eval_episodes, max_lengh, kwargs["action_dim"]])
    rewards = torch.zeros([args.eval_episodes, max_lengh, 1])
    dones = torch.zeros([args.eval_episodes, max_lengh, 1])
    lengths = []

    episodes_reward = []

    for i in range(args.eval_episodes):
        state, done = env.reset(), False
        path_length = 0
        path_reward = 0
        while not done:
            action = policy.select_action(np.array(state))

            states[i][path_length] = torch.tensor(state)
            actions[i][path_length] = torch.tensor(action)

            state, reward, done, _ = env.step(action)

            next_states[i][path_length] = torch.tensor(state)
            rewards[i][path_length] = torch.tensor(reward)
            dones[i][path_length] = torch.tensor(done)

            path_length += 1
            path_reward += reward

        lengths.append(path_length)
        episodes_reward.append(path_reward)
        print(f"eval_episode:{i+1}, path_length:{path_length}, path_reward:{path_reward}")

    avg_reward = np.mean(np.array(episodes_reward))
    reward_std = np.std(np.array(episodes_reward))
    lengths = torch.tensor(np.array(lengths))

    expert_data = {
        "states": states,
        "next_states": next_states,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
        "lengths": lengths
    }

    torch.save(expert_data, os.path.join("./gail_experts/", data_file_name), _use_new_zipfile_serialization=False)

    print("---------------------------------------")
    print(f"env:{args.env}, evaluation over last {args.eval_episodes} episodes, mean_reward:{avg_reward:.1f}, reward_std:{reward_std:.1f}")
    print("---------------------------------------")