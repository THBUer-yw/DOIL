import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action, num_hidden_layers):
		super(Actor, self).__init__()
		self.num_hidden_layers = num_hidden_layers
		self.input_layer = nn.Linear(state_dim, 256)
		self.hidden_layers = nn.ModuleList([nn.Linear(256, 256) for _ in range(self.num_hidden_layers)])
		self.output_layer = nn.Linear(256, action_dim)

		self.max_action = max_action

	def forward(self, state):
		a = F.relu(self.input_layer(state))
		for i in range(self.num_hidden_layers):
			a = F.relu(self.hidden_layers[i](a))
		return self.max_action * torch.tanh(self.output_layer(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim, num_hidden_layers):
		super(Critic, self).__init__()
		self.num_hidden_layers = num_hidden_layers

		# Q1 architecture
		self.input_layer1 = nn.Linear(state_dim + action_dim, 256)
		self.hidden_layers1 = nn.ModuleList([nn.Linear(256, 256) for _ in range(self.num_hidden_layers)])
		self.output_layer1 = nn.Linear(256, 1)

		# Q2 architecture
		self.input_layer2 = nn.Linear(state_dim + action_dim, 256)
		self.hidden_layers2 = nn.ModuleList([nn.Linear(256, 256) for _ in range(self.num_hidden_layers)])
		self.output_layer2 = nn.Linear(256, 1)

	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.input_layer1(sa))
		for i in range(self.num_hidden_layers):
			q1 = F.relu(self.hidden_layers1[i](q1))
		q1 = self.output_layer1(q1)

		q2 = F.relu(self.input_layer2(sa))
		for i in range(self.num_hidden_layers):
			q2 = F.relu(self.hidden_layers2[i](q2))
		q2 = self.output_layer2(q2)
		return q1, q2

	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.input_layer1(sa))
		for i in range(self.num_hidden_layers):
			q1 = F.relu(self.hidden_layers1[i](q1))
		q1 = self.output_layer1(q1)
		return q1

# class Actor(nn.Module):
# 	def __init__(self, state_dim, action_dim, max_action, num_hidden_layers):
# 		super(Actor, self).__init__()
#
# 		self.l1 = nn.Linear(state_dim, 256)
# 		self.l2 = nn.Linear(256, 256)
# 		self.l3 = nn.Linear(256, action_dim)
#
# 		self.max_action = max_action
#
#
# 	def forward(self, state):
# 		a = F.relu(self.l1(state))
# 		a = F.relu(self.l2(a))
# 		return self.max_action * torch.tanh(self.l3(a))
#
#
# class Critic(nn.Module):
# 	def __init__(self, state_dim, action_dim, num_hidden_layers):
# 		super(Critic, self).__init__()
#
# 		# Q1 architecture
# 		self.l1 = nn.Linear(state_dim + action_dim, 256)
# 		self.l2 = nn.Linear(256, 256)
# 		self.l3 = nn.Linear(256, 1)
#
# 		# Q2 architecture
# 		self.l4 = nn.Linear(state_dim + action_dim, 256)
# 		self.l5 = nn.Linear(256, 256)
# 		self.l6 = nn.Linear(256, 1)
#
#
# 	def forward(self, state, action):
# 		sa = torch.cat([state, action], 1)
#
# 		q1 = F.relu(self.l1(sa))
# 		q1 = F.relu(self.l2(q1))
# 		q1 = self.l3(q1)
#
# 		q2 = F.relu(self.l4(sa))
# 		q2 = F.relu(self.l5(q2))
# 		q2 = self.l6(q2)
# 		return q1, q2
#
#
# 	def Q1(self, state, action):
# 		sa = torch.cat([state, action], 1)
#
# 		q1 = F.relu(self.l1(sa))
# 		q1 = F.relu(self.l2(q1))
# 		q1 = self.l3(q1)
# 		return q1

class TD3(object):
	def __init__(self, args, state_dim, action_dim, max_action, use_cuda, num_hidden_layers, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):

		self.device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
		self.args = args
		self.actor = Actor(state_dim, action_dim, max_action, num_hidden_layers).to(self.device)
		self.actor_target = copy.deepcopy(self.actor)
		self.critic = Critic(state_dim, action_dim, num_hidden_layers).to(self.device)
		self.critic_target = copy.deepcopy(self.critic)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq

		self.total_it = 0


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
		return self.actor(state).cpu().data.numpy().flatten()

	def train(self, replay_buffer, writer, steps, gail=None):
		self.total_it += 1
		# Sample replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(self.args.batch_size)
		if gail:
			reward = gail.predict_reward(state, action, self.args.discount, not_done, self.args.reward_type)
			writer.add_scalar("discriminator/gail_reward", np.mean(np.array(reward.to("cpu")), axis=0), steps)

		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
			
			next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + not_done * self.discount * target_Q

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:

			# Compute actor losse
			actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


	def load(self, filename, device):
		self.critic.load_state_dict(torch.load(filename + "_critic", map_location=device))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer", map_location=device))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor", map_location=device))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer", map_location=device))
		self.actor_target = copy.deepcopy(self.actor)

		