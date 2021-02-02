import numpy as np
import torch
import time


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, args, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")

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


class LearningRate:

	__instance = None

	def __init__(self):
		if LearningRate.__instance is not None:
			raise Exception("Singleton instantiation called twice")
		else:
			LearningRate.__instance = self
			self.lr = None
			self.decay_factor = None
			self.training_step = 0

	@staticmethod
	def get_instance():
		"""Get the singleton instance.

		Returns:
			(LearningRate)
		"""
		if LearningRate.__instance is None:
			LearningRate()
		return LearningRate.__instance

	def set_learning_rate(self, lr):
		self.lr = lr

	def get_learning_rate(self):
		return self.lr

	def increment_step(self):
		self.training_step += 1

	def get_step(self):
		return self.training_step

	def set_decay(self, d):
		self.decay_factor = d

	def decay(self):
		self.lr = self.lr * self.decay_factor


def Log_save_name4gail(args):
	time_str = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
	save_name = args.env + '_algo_{}_gail_{}_seed_{}_total_steps_{}_num_trajs_{}_subsample_frequency_{}_gail_epoch_{}_prepoch_{}_' \
						   'max_horizon_{}_warm_times_{}_start_steps_{}_wdail_{}_states_only_{}_reward_type_{}_use_dense_{}_hidden_layers_{}'\
                    .format(args.policy,
							args.gail,
							args.seed,
                            args.total_steps,
                            args.num_trajs,
                            args.subsample_frequency,
                            args.gail_epoch,
							args.gail_prepoch,
                            args.max_horizon,
							args.warm_times,
                            args.start_steps,
                            args.wdail,
                            args.states_only,
                            args.reward_type,
							args.use_dense_network,
							args.hidden_layers,
                            ) + "_" + time_str
	return save_name


def Log_save_name4td3(args):
	time_str = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
	save_name = args.env + '_algo_{}_gail_{}_seed_{}_total_steps_{}_max_horizon_{}_start_steps_{}_use_dense_{}' \
		.format(args.policy,
				args.gail,
				args.seed,
				args.total_steps,
				args.max_horizon,
				args.start_steps,
				args.use_dense_network,
				) + "_" + time_str
	return save_name