import numpy as np
import torch
import gym
import argparse
import os
import time

import TD3
import Dense_TD3

# ant 5825.0 115.6 halfcheetah 11049.1 136.2 hopper 3707.3 11.8 reacher -3.8 1.8 walker2d 4729.4 23.0
# InvertedDoublePendulum 9359.8 0.1 bipedalwalker 295.4 1.2

# ant 6405.6 221.6  halfcheetah 14053.2 100.9 hopper 3776.9 26.4 walker2d 4806.8 12.4

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3", help="Policy name (TD3, DDPG or OurDDPG)")
    parser.add_argument("--env", default="BipedalWalker-v3", help="OpenAI gym environment name")
    parser.add_argument('--hidden_layers', type=int, default=3, help='numbers of hidden layers')
    parser.add_argument("--seed", default=0, type=int, help="Sets Gym, PyTorch and Numpy seeds")
    parser.add_argument("--eval_episodes", default=20, type=int, help="How often (time steps) we evaluate")
    parser.add_argument("--use_dense_network", type=int, default=0, help="whether use densenet")
    parser.add_argument("--random", default=0, type=int, help="evaluate the random policy")
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
        "max_action": max_action,
        "use_cuda": False
    }

    if args.policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["args"] = args
        if args.use_dense_network:
            policy = Dense_TD3.TD3(**kwargs)
            print("Using the dense net!")
        else:
            policy = TD3.TD3(**kwargs)
            print("Using the MLP!")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_file_name = args.env+"_"+"seed_"+str(args.seed)
    if not args.random:
        policy.load(f"./models_mlp/{model_file_name}", device)

    data_file_name = "trajs_"+args.env.lower()[:-3]+".pt"

    if args.env == "Ant-v2" or "HalfCheetah-v2" or "Hopper-v2" or "Walker2d-v2" or "InvertedDoublePendulum-v2":
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
            if args.random:
                action = env.action_space.sample()
            else:
                action = policy.select_action(np.array(state))

            states[i][path_length] = torch.tensor(state)
            actions[i][path_length] = torch.tensor(action)

            state, reward, done, _ = env.step(action)
            # env.render()
            # time.sleep(0.02)

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

    torch.save(expert_data, os.path.join("./gail_experts_mlp/", data_file_name), _use_new_zipfile_serialization=False)

    print("---------------------------------------")
    print(f"env:{args.env}, evaluation over last {args.eval_episodes} episodes, mean_reward:{avg_reward:.1f}, reward_std:{reward_std:.1f}")
    print("---------------------------------------")