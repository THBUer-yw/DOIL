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
import gail


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + 100)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			action = policy.select_action(np.array(state))
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"env:{env_name}, evaluation over last {eval_episodes} episodes: {avg_reward:.1f}")
	print("---------------------------------------")
	return avg_reward


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--batch_size", default=256, type=int, help="Batch size for both actor and critic")
	parser.add_argument("--discount", default=0.99, help="Discount factor")
	parser.add_argument("--eval_freq", default=5e3, type=int, help="How often (time steps) we evaluate")
	parser.add_argument("--env", default="HalfCheetah-v2", help="OpenAI gym environment name")
	parser.add_argument("--expl_noise", default=0.1, help="Std of Gaussian exploration noise")
	parser.add_argument('--gail', action='store_true', default=True, help='do imitation learning with gail')
	parser.add_argument('--gail-batch-size', type=int, default=128, help='gail batch size (default: 128)')
	parser.add_argument('--gail-experts-dir', default='./gail_experts', help='directory that contains expert demonstrations for gail')
	parser.add_argument('--gail-epoch', type=int, default=2, help='gail epochs (default: 5)')
	parser.add_argument('--max_horizon', type=int, default=2048, help='steps interval for training dicriminator')
	parser.add_argument("--load_model", default="", help="Model load file name")
	parser.add_argument("--noise_clip", default=0.5, help="Range to clip target policy noise")
	parser.add_argument('--num_trajs', type=int, default=5)
	parser.add_argument("--policy", default="TD3", help="Policy name (TD3, DDPG or OurDDPG)")
	parser.add_argument("--policy_noise", default=0.2, help="Noise added to target policy during critic update")
	parser.add_argument("--policy_freq", default=2, type=int, help="Frequency of delayed policy updates")
	parser.add_argument("--reward_type", type=int, default=1, help="different types of rewards")
	parser.add_argument("--seed", default=0, type=int, help="Sets Gym, PyTorch and Numpy seeds")
	parser.add_argument("--start_steps", default=2e4, type=int, help="Time steps initial random policy is used")
	parser.add_argument("--save_model", default=True, action="store_true", help="Save model and optimizer parameters")
	parser.add_argument('--states_only', type=int, default=0, help="if do imitation learning using only states tuple")
	parser.add_argument('--subsample_frequency', help='subsample for each trajectory', type=int, default=1)
	parser.add_argument("--total_steps", default=1e6, type=int, help="Max time steps to run environment")
	parser.add_argument("--tau", default=0.005, help="Target network update rate")
	parser.add_argument("--use_lr_decay", default=False, help="decay the learning rate for optimizer")
	parser.add_argument("--wdail", type=int, default=0, help="train the agent with wdail method")
	args = parser.parse_args()

	# file_name = f"{args.policy}_{args.env}_{args.seed}"
	# print("---------------------------------------")
	# print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	# print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	log_save_name = utils.Log_save_name4td3(args)
	log_save_path = os.path.join("./runs", log_save_name)
	if os.path.exists(log_save_path):
		shutil.rmtree(log_save_path)

	writer = SummaryWriter(log_save_path)
	if os.path.exists("./results/" + log_save_name):
		shutil.rmtree("./results/" + log_save_name)
	os.makedirs("./results/" + log_save_name)
	log_file = os.path.join("./results/" + log_save_name, "train_log.txt")

	env = gym.make(args.env)
	# Set seeds
	env.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	if args.use_lr_decay:
		lr = utils.LearningRate.get_instance()
		lr.lr = 10 ** (-3)
		lr.decay_factor = 0.5
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
	}

	# Initialize policy
	if args.policy == "TD3":
		# Target policy smoothing is scaled wrt the action scale
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		policy = TD3.TD3(**kwargs)
	elif args.policy == "OurDDPG":
		policy = OurDDPG.DDPG(**kwargs)
	elif args.policy == "DDPG":
		policy = DDPG.DDPG(**kwargs)

	# if args.load_model != "":
	# 	policy_file = file_name if args.load_model == "default" else args.load_model
	# 	policy.load(f"./models/{policy_file}")

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
	
	# Evaluate untrained policy
	evaluations = [eval_policy(policy, args.env, args.seed)]

	state, done = env.reset(), False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0
	max_eval_rewards = -1e3
	horizon = 0

	for t in range(int(args.total_steps)):
		
		episode_timesteps += 1
		horizon += 1

		# Select action randomly or according to policy
		if t < args.start_steps:
			action = env.action_space.sample()
		else:
			action = (policy.select_action(np.array(state))+np.random.normal(0, max_action * args.expl_noise, size=action_dim)).clip(-max_action, max_action)

		# Perform action
		next_state, reward, done, _ = env.step(action) 
		done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

		# Store data in replay buffer
		replay_buffer.add(state, action, next_state, reward, done_bool)

		state = next_state
		episode_reward += reward

		# Train agent after collecting sufficient data
		if t >= args.start_steps:
			if args.gail and horizon >= args.max_horizon:
				horizon = 0
				if args.states_only:
					discr = gail.Discriminator(kwargs["state_dim"]+kwargs["state_dim"], 100, states_only=args.states_only)
				else:
					discr = gail.Discriminator(kwargs["state_dim"]+kwargs["action_dim"], 100, states_only=args.states_only)

				file_name = os.path.join(args.gail_experts_dir, "trajs_{}.pt".format(args.env.split('-')[0].lower()))

				expert_dataset = gail.ExpertDataset(file_name, num_trajectories=args.num_trajs, subsample_frequency=args.subsample_frequency, states_only=args.states_only)
				args.gail_batch_size = min(args.gail_batch_size, len(expert_dataset))
				drop_last = len(expert_dataset) > args.gail_batch_size
				gail_train_loader = torch.utils.data.DataLoader(dataset=expert_dataset, batch_size=args.gail_batch_size, shuffle=True, drop_last=drop_last)

				expert_losses, policy_losses, dis_losses, dis_gps, dis_total_losses = [], [], [], [], []
				for _ in range(args.gail_epoch):
					if args.wdail:
						expert_loss, policy_loss, dis_loss, dis_gp, dis_total_loss = discr.update_wdail(gail_train_loader, replay_buffer)
					else:
						expert_loss, policy_loss, dis_loss, dis_gp, dis_total_loss = discr.update(gail_train_loader, replay_buffer)

					expert_losses.append(expert_loss)
					policy_losses.append(policy_loss)
					dis_losses.append(dis_loss)
					dis_gps.append(dis_gp)
					dis_total_losses.append(dis_total_loss)

				writer.add_scalar("discriminator/expert_loss", np.mean(np.array(expert_losses)), t+1)
				writer.add_scalar("discriminator/policy_loss", np.mean(np.array(policy_losses)), t+1)
				writer.add_scalar("discriminator/dis_loss", np.mean(np.array(dis_losses)), t+1)
				writer.add_scalar("discriminator/dis_gradient", np.mean(np.array(dis_gps)), t+1)
				writer.add_scalar("discriminator/total_loss", np.mean(np.array(dis_total_losses)), t+1)

			policy.train(args, replay_buffer, discr)

		writer.add_scalar("train/rewrad", episode_reward+1, t+1)
		writer.add_scalar("train/path_length", episode_timesteps, t+1)

		if done:
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			print(f"env:{args.env}, current steps: {t+1}, path_len: {episode_timesteps}, reward: {episode_reward:.1f}")
			# Reset environment
			state, done = env.reset(), False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1 

		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			file_name = args.env+"_"+"seed_"+str(args.seed)
			mean_eval_rewards = eval_policy(policy, args.env, args.seed)
			writer.add_scalar("eval/mean_reward", mean_eval_rewards, t+1)
			writer.add_scalar("eval/max_eval_reward", max_eval_rewards, t+1)
			if mean_eval_rewards > max_eval_rewards and args.save_model and not args.gail:
				max_eval_rewards = mean_eval_rewards
				policy.save(f"./models/{file_name}")
