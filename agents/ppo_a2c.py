import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random


class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCriticNetwork, self).__init__()

        self.shared_fc1 = nn.Linear(state_dim, hidden_dim)
        self.shared_fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.actor_head = nn.Linear(hidden_dim, action_dim)

        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = F.relu(self.shared_fc1(state))
        x = F.relu(self.shared_fc2(x))

        action_probs = F.softmax(self.actor_head(x), dim=-1)

        value = self.critic_head(x)

        return action_probs, value


class PPO_A2C:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=3e-4,
        gamma=0.99,
        eps_clip=0.2,
        k_epochs=4,
        batch_size=64,
        hidden_dim=128
    ):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.batch_size = batch_size

        self.policy = ActorCriticNetwork(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.policy_old = ActorCriticNetwork(state_dim, action_dim, hidden_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.memory = {
            'states': [],
            'actions': [],
            'logprobs': [],
            'masks': [],
            'rewards': [],
            'is_terminals': [],
            'values': []
        }

    def select_action(self, state, available_actions=None):
        state = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            action_probs, value = self.policy_old(state)

        if available_actions is not None:
            mask = torch.zeros_like(action_probs)
            mask[0, available_actions] = 1
            action_probs = action_probs * mask
            prob_sum = action_probs.sum()
            if prob_sum > 0 and not torch.isnan(prob_sum):
                action_probs = action_probs / prob_sum
            else:
                action_probs = mask / mask.sum()
        else:
            mask = torch.ones_like(action_probs)

        dist = torch.distributions.Categorical(action_probs)

        action = dist.sample() # ici on prend pas argmax, pour explorer plus 

        self.memory['states'].append(state)
        self.memory['actions'].append(action)
        self.memory['logprobs'].append(dist.log_prob(action))
        self.memory['masks'].append(mask)
        self.memory['values'].append(value)

        return action.item()

    def store_reward(self, reward, is_terminal):
        self.memory['rewards'].append(reward)
        self.memory['is_terminals'].append(is_terminal)

    def update(self):
        if len(self.memory['rewards']) == 0:
            return

        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.memory['rewards']),
                                      reversed(self.memory['is_terminals'])):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.FloatTensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        old_states = torch.cat(self.memory['states']).detach()
        old_actions = torch.cat(self.memory['actions']).detach()
        old_logprobs = torch.cat(self.memory['logprobs']).detach()
        old_masks = torch.cat(self.memory['masks']).detach()
        old_values = torch.cat(self.memory['values']).detach().squeeze()

        advantages = rewards - old_values

        for _ in range(self.k_epochs):
            action_probs, values = self.policy(old_states)
            # Appliquer le même masque que lors de la collecte
            action_probs = action_probs * old_masks
            prob_sums = action_probs.sum(dim=-1, keepdim=True)
            prob_sums = prob_sums.clamp(min=1e-8)
            action_probs = action_probs / prob_sums
            dist = torch.distributions.Categorical(action_probs)

            new_logprobs = dist.log_prob(old_actions)
            dist_entropy = dist.entropy()

            ratios = torch.exp(new_logprobs - old_logprobs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = 0.5 * F.mse_loss(values.squeeze(), rewards)

            loss = actor_loss + critic_loss - 0.01 * dist_entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

        self.clear_memory()

    def clear_memory(self):
        self.memory = {
            'states': [],
            'actions': [],
            'logprobs': [],
            'masks': [],
            'rewards': [],
            'is_terminals': [],
            'values': []
        }

    def save(self, checkpoint_path):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, checkpoint_path)

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy_old.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])