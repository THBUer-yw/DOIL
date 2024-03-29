import numpy as np
import torch
import gym
import argparse
import os
import shutil
import time
from tensorboardX import SummaryWriter

import utils
import TD3
import Dense_TD3
import gail


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes):
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
	return avg_reward


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--batch_size", default=256, type=int, help="Batch size for both actor and critic")
	parser.add_argument("--discount", default=0.99, help="Discount factor")
	parser.add_argument("--decay_steps", default=1e5, help="Discount factor")
	parser.add_argument("--eval_freq", default=3e3, type=int, help="How often (time steps) we evaluate")
	parser.add_argument("--eval_episodes", default=5, type=int, help="How many episodes for each evaluation")
	parser.add_argument("--env", default="HalfCheetah-v2", help="OpenAI gym environment name")
	parser.add_argument("--expl_noise", default=0.1, help="Std of Gaussian exploration noise")
	parser.add_argument('--gail', type=int, default=1, help='do imitation learning with gail')
	parser.add_argument('--gail_batch_size', type=int, default=128, help='gail batch size (default: 128)')
	parser.add_argument('--gail_experts_dir', default='./gail_experts_dense', help='directory that contains expert demonstrations for gail')
	parser.add_argument('--gail_epoch', type=int, default=50, help='gail epochs (default: 5)')
	parser.add_argument('--gail_prepoch', type=int, default=100, help='gail prepochs (default: 50)')
	parser.add_argument('--hidden_layers', type=int, default=2, help='numbers of hidden layers')
	parser.add_argument('--max_horizon', type=int, default=2048, help='steps interval for training dicriminator')
	parser.add_argument("--load_model", default="", help="Model load file name")
	parser.add_argument("--noise_clip", default=0.5, help="Range to clip target policy noise")
	parser.add_argument('--num_trajs', type=int, default=5)
	parser.add_argument("--policy", default="TD3", help="Policy name (TD3, DDPG or OurDDPG)")
	parser.add_argument("--policy_noise", default=0.2, help="Noise added to target policy during critic update")
	parser.add_argument("--policy_freq", default=2, type=int, help="Frequency of delayed policy updates")
	parser.add_argument("--reward_type", type=int, default=3, help="different types of rewards")
	parser.add_argument("--seed", default=0, type=int, help="Sets Gym, PyTorch and Numpy seeds")
	parser.add_argument("--start_steps", default=2e4, type=int, help="Time steps initial random policy is used")
	parser.add_argument("--save_model", default=True, action="store_true", help="Save model and optimizer parameters")
	parser.add_argument('--states_only', type=int, default=0, help="if do imitation learning using only states tuple")
	parser.add_argument('--subsample_frequency', help='subsample for each trajectory', type=int, default=1)
	parser.add_argument("--total_steps", default=1e6, type=int, help="Max time steps to run environment")
	parser.add_argument("--tau", default=0.005, help="Target network update rate")
	parser.add_argument("--use_lr_decay", type=int, default=0, help="decay the learning rate for optimizer")
	parser.add_argument("--use_cuda", type=int, default=1, help="whether use GPU")
	parser.add_argument("--use_dense_network", type=int, default=0, help="whether use densenet")
	parser.add_argument("--wdail", type=int, default=0, help="train the agent with wdail method")
	parser.add_argument("--warm_times", type=int, default=10, help="warm times for the discriminator")
	args = parser.parse_args()

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	if args.gail:
		log_save_name = utils.Log_save_name4gail(args)
	else:
		log_save_name = utils.Log_save_name4td3(args)
	log_save_path = os.path.join("./runs", log_save_name)
	if os.path.exists(log_save_path):
		shutil.rmtree(log_save_path)

	writer = SummaryWriter(log_save_path)
	if os.path.exists("./results/" + log_save_name):
		shutil.rmtree("./results/" + log_save_name)
	os.makedirs("./results/" + log_save_name)
	log_file = os.path.join("./results/" + log_save_name, "eval_log.txt")

	env = gym.make(args.env)
	# Set seeds
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
		"discount": args.discount,
		"tau": args.tau,
	}

	# Initialize policy
	if args.policy == "TD3":
		# Target policy smoothing is scaled wrt the action scale
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		kwargs["use_cuda"] = args.use_cuda
		kwargs["args"] = args
		if args.use_dense_network:
			policy = Dense_TD3.TD3(**kwargs)
			print("Using the dense net!")
		else:
			kwargs["num_hidden_layers"] = args.hidden_layers
			policy = TD3.TD3(**kwargs)
			print("Using the MLP!")

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim, args)

	if args.gail:
		file_name = os.path.join(args.gail_experts_dir, "trajs_{}.pt".format(args.env.split('-')[0].lower()))
		print("Loading expert trajectory data!")
		expert_dataset = gail.ExpertDataset(file_name, num_trajectories=args.num_trajs, subsample_frequency=args.subsample_frequency, states_only=args.states_only)
		args.gail_batch_size = min(args.gail_batch_size, len(expert_dataset))
		drop_last = len(expert_dataset) > args.gail_batch_size
		gail_train_loader = torch.utils.data.DataLoader(dataset=expert_dataset, batch_size=args.gail_batch_size, shuffle=True, drop_last=drop_last)

		if args.states_only:
			discr = gail.Discriminator(args, kwargs["state_dim"] + kwargs["state_dim"], 100)
		else:
			discr = gail.Discriminator(args, kwargs["state_dim"] + kwargs["action_dim"], 100)
	
	# Evaluate untrained policy
	evaluations = [eval_policy(policy, args.env, args.seed, args.eval_episodes)]

	state, done = env.reset(), False
	episode_reward = 0
	episode_timesteps = 0
	max_eval_rewards = -1e3
	train_discri = 0
	warm_start = True
	start_time = time.time()

	for t in range(int(args.total_steps)):
		
		episode_timesteps += 1

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
			if args.gail:
				if (t+1) % args.max_horizon == 0:
					train_discri += 1
					warm_start = False if train_discri > args.warm_times else True
					discri_train_epoch = args.gail_prepoch if warm_start else args.gail_epoch
					print(f"time_step:{t+1}, train discriminator:{train_discri}\n")
					dis_losses, dis_gps, dis_total_losses = [], [], []
					for _ in range(discri_train_epoch):
						if args.wdail:
							_, _, dis_loss, dis_gp, dis_total_loss = discr.update_wdail(gail_train_loader, replay_buffer, warm_start)
						else:
							_, _, dis_loss, dis_gp, dis_total_loss = discr.update(gail_train_loader, replay_buffer, warm_start)

						dis_losses.append(dis_loss)
						dis_gps.append(dis_gp)
						dis_total_losses.append(dis_total_loss)

					writer.add_scalar("discriminator/dis_loss", np.mean(np.array(dis_losses)), t+1)
					writer.add_scalar("discriminator/dis_gradient", np.mean(np.array(dis_gps)), t+1)
					writer.add_scalar("discriminator/total_loss", np.mean(np.array(dis_total_losses)), t+1)

				policy.train(replay_buffer, writer, t+1, discr)
			else:
				policy.train(replay_buffer, writer, t+1)

		writer.add_scalar("train/rewrad", episode_reward, t+1)

		if done:
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			# Reset environment
			state, done = env.reset(), False
			episode_reward = 0
			episode_timesteps = 0

		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			file_name = args.env+"_seed_"+str(args.seed)+"_dense_"+str(args.use_dense_network)+"_states_only_"+str(args.states_only)
			mean_eval_rewards = eval_policy(policy, args.env, args.seed, eval_episodes=args.eval_episodes)
			end_time = time.time()
			fps = (t+1)/(end_time-start_time)
			print(f"env:{args.env},train_steps:{t+1},**FPS**:{fps:.0f},evaluation over last {args.eval_episodes} episodes:{mean_eval_rewards:.1f}\n")
			writer.add_scalar("eval/mean_reward", mean_eval_rewards, t+1)
			with open(log_file, "a") as file:
				print("train_step:{}, evaluation using {} episodes, mean reward {:.2f}".format(t+1, args.eval_episodes, mean_eval_rewards), file=file)
			if mean_eval_rewards > max_eval_rewards and args.save_model:
				max_eval_rewards = mean_eval_rewards
				print("***********************")
				print("Saving model!")
				print("***********************")
				policy.save(f"./models/{file_name}")
