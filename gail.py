import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch import autograd
from baselines.common.running_mean_std import RunningMeanStd

class Discriminator(nn.Module):
    def __init__(self, args, input_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
        self.args = args
        self.states_only = self.args.states_only
        self.pre_train = 0

        self.trunk = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)).to(self.device)
        self.trunk.train()
        self.optimizer = torch.optim.Adam(self.trunk.parameters())

        self.returns = None
        self.ret_rms = RunningMeanStd(shape=())

    def compute_grad_pen(self,
                         expert_state,
                         expert_action,
                         policy_state,
                         policy_action,
                         lambda_=10):
        alpha = torch.rand(expert_state.size(0), 1)
        expert_data = torch.cat([expert_state, expert_action], dim=1)
        policy_data = torch.cat([policy_state, policy_action], dim=1)

        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        mixup_data = alpha * expert_data + (1 - alpha) * policy_data
        mixup_data.requires_grad = True

        disc = self.trunk(mixup_data)
        ones = torch.ones(disc.size()).to(disc.device)
        grad = autograd.grad(
            outputs=disc,
            inputs=mixup_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

    def update(self, expert_loader, replay_buffer, warm_start):
        self.train()
        loss = 0
        e_loss = 0
        p_loss = 0
        g_loss = 0
        gp = 0
        n = 0
        JS_LOSS = 0
        if warm_start:
            self.pre_train += 1
            print(f"warm start pretrain:{self.pre_train}")

        for expert_batch in expert_loader:
            states, actions, next_states, _, _ = replay_buffer.sample(batch_size=expert_loader.batch_size)
            if self.states_only:
                policy_state, policy_action = states, next_states
            else:
                policy_state, policy_action = states, actions

            policy_d = self.trunk(torch.cat([policy_state, policy_action], dim=1))

            expert_state, expert_action = expert_batch
            expert_state = torch.FloatTensor(expert_state).to(self.device)
            expert_action = expert_action.to(self.device)
            expert_d = self.trunk(torch.cat([expert_state, expert_action], dim=1))

            expert_loss = F.binary_cross_entropy_with_logits(expert_d, torch.ones(expert_d.size()).to(self.device))
            policy_loss = F.binary_cross_entropy_with_logits(policy_d, torch.zeros(policy_d.size()).to(self.device))

            gail_loss = expert_loss + policy_loss
            g_loss += (expert_loss + policy_loss).item()
            grad_pen = self.compute_grad_pen(expert_state, expert_action, policy_state, policy_action)

            e_loss += expert_loss.item()
            p_loss += policy_loss.item()
            loss += (gail_loss + grad_pen).item()
            gp += (grad_pen).item()
            n += 1

            self.optimizer.zero_grad()
            (gail_loss + grad_pen).backward()
            self.optimizer.step()

        return e_loss / n, p_loss / n, JS_LOSS / n, gp / n, loss / n

    def update_wdail(self, expert_loader, replay_buffer):
        self.train()
        states, actions, next_states, _, _ = replay_buffer.sample(batch_size=expert_loader.batch_size)

        loss = 0
        g_loss =0.0
        gp =0.0
        n = 0
        e_loss = 0
        p_loss = 0
        for expert_batch in expert_loader:
            if self.states_only:
                policy_state, policy_action = states, next_states
            else:
                policy_state, policy_action = states, actions

            policy_d = self.trunk( torch.cat([policy_state, policy_action], dim=1))

            expert_state, expert_action = expert_batch
            expert_state = torch.FloatTensor(expert_state).to(self.device)
            expert_action = expert_action.to(self.device)
            expert_d = self.trunk(torch.cat([expert_state, expert_action], dim=1))

            expert_loss = torch.mean(torch.tanh(expert_d)).to(self.device)
            policy_loss = torch.mean(torch.tanh(policy_d)).to(self.device)

            wd = expert_loss - policy_loss
            grad_pen = self.compute_grad_pen(expert_state, expert_action, policy_state, policy_action)

            e_loss += expert_loss.item()
            p_loss += policy_loss.item()
            loss += (wd + grad_pen).item()
            g_loss += (wd).item()  # wd indicates the distance between the expert and policy
            gp += (grad_pen).item()
            n += 1

            self.optimizer.zero_grad()
            (-wd + grad_pen).backward()
            self.optimizer.step()

        return e_loss/n, p_loss/n, g_loss/n, gp/n, loss / n

    def predict_reward(self, state, action, gamma, masks, reward_type, update_rms=True):
        with torch.no_grad():
            self.eval()
            d = self.trunk(torch.cat([state, action], dim=1))
            s = torch.sigmoid(d)

            # If traditional GAIL
            if reward_type == 1:  # Original function
                reward = s.log() - (1 - s).log()
            elif reward_type == 2:  # Linear function
                reward = s
            elif reward_type == 3:  # Logarithmic function
                reward = - (1 - s).log()
            elif reward_type == 4:  # Logarithmic function
                reward = s.log()
            elif reward_type == 5:  # Exponential function
                reward = torch.exp(s)
            elif reward_type == 6:  # Inverse proportional function
                reward = -1 / (s + 1e-8)
            elif reward_type == 7:  # Power function
                reward = torch.pow(s, 2)
            elif reward_type == 8:  # Power function
                reward = torch.pow(s, 0.5)


            if self.returns is None:
                self.returns = reward.clone()

            if update_rms:
                self.returns = self.returns * masks * gamma + reward
                self.ret_rms.update(self.returns.cpu().numpy())

            return reward / np.sqrt(self.ret_rms.var[0] + 1e-8)


class ExpertDataset(torch.utils.data.Dataset):
    def __init__(self, file_name, num_trajectories=4, subsample_frequency=20, train=True, start=0, states_only=False):
        all_trajectories = torch.load(file_name)
        states = all_trajectories["states"]
        next_states = torch.zeros((states.size(0), states.size(1), states.size(2)))
        for traj_num in range(states.size(0)):
            traj_next_states = torch.zeros(states.size(1), states.size(2))
            traj_states = states[traj_num]
            for step in range(1, states.size(1)):
                traj_next_states[step-1] = traj_states[step]
            last_next_states = torch.zeros(states.size(2))
            traj_next_states[-1] = last_next_states
            next_states[traj_num] = traj_next_states

        all_trajectories["next_states"] = next_states
        perm = torch.randperm(all_trajectories['states'].size(0))
        #idx = perm[:num_trajectories]
        idx = np.arange(num_trajectories) + start
        if not train:
            assert start > 0

        self.trajectories = {}

        # See https://github.com/pytorch/pytorch/issues/14886
        # .long() for fixing bug in torch v0.4.1
        start_idx = torch.randint(0, subsample_frequency, size=(num_trajectories, )).long()

        for k, v in all_trajectories.items():
            data = v[idx]

            if k != 'lengths':
                samples = []
                for i in range(num_trajectories):
                    samples.append(data[i, start_idx[i]::subsample_frequency])
                self.trajectories[k] = torch.stack(samples)
            else:
                self.trajectories[k] = data // subsample_frequency

        self.i2traj_idx = {}
        self.i2i = {}

        self.length = self.trajectories['lengths'].sum().item()

        traj_idx = 0
        i = 0
        self.states_only = states_only
        self.get_idx = []

        for j in range(self.length):

            while self.trajectories['lengths'][traj_idx].item() <= i:
                i -= self.trajectories['lengths'][traj_idx].item()
                traj_idx += 1

            self.get_idx.append((traj_idx, i))

            i += 1


    def __len__(self):
        return self.length

    def __getitem__(self, i):
        traj_idx, i = self.get_idx[i]
        if self.states_only:
            results = self.trajectories['states'][traj_idx][i], self.trajectories['next_states'][traj_idx][i]
        else:
            results = self.trajectories['states'][traj_idx][i], self.trajectories['actions'][traj_idx][i]
        return results
