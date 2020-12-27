import numpy as np
import torch
import time


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		# self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.device = torch.device("cpu")

	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)


def Log_save_name4gail(args):
	time_str = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
	save_name = args.env_name + '_seed_{}_total_steps_{}_num_trajs_{}_subsample_frequency_{}_prepoch_{}_bcgail_{}_decay_{}_gamma_{}_' \
                                'red_{}_sail_{}_wdail_{}_states_only_{}_reward_type_{}'\
                    .format(args.seed,
                            args.num_env_steps,
                            args.num_trajs,
                            args.subsample_frequency,
                            args.gail_prepoch,
                            args.bcgail,
                            args.decay,
                            args.gailgamma,
                            args.red,
                            args.sail,
                            args.wdail,
                            args.states_only,
                            args.reward_type,
                            ) + "_" + time_str
	return save_name


def Log_save_name4td3(args):
	time_str = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
	save_name = args.env + '_seed_{}_total_steps_{}_start_steps_{}_eval_freq_{}_expl_noise_{}_gamma_{}_tau_{}_policy_freq_{}' .format(args.seed,
                            args.total_steps,
                            args.start_steps,
                            args.eval_freq,
                            args.expl_noise,
                            args.discount,
                            args.tau	,
                            args.policy_freq
                            ) + "_" + time_str
	return save_name