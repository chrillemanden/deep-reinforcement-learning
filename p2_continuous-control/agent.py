import random
import numpy as np

from collections import namedtuple, deque

# Pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Implementation imports
from model import Actor, Critic 
#from model import Critic
#import shared_memory

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4 
actor_lr = 5e-3
critic_lr = 5e-4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
	"""Fixed-size buffer to store experience tuples."""

	def __init__(self, action_size, buffer_size, batch_size, seed):
		"""Initialize a ReplayBuffer object.

		Params
		======
			action_size (int): dimension of each action
			buffer_size (int): maximum size of buffer
			batch_size (int): size of each training batch
			seed (int): random seed
		"""
		self.action_size = action_size
		self.memory = deque(maxlen=buffer_size)
		self.batch_size = batch_size
		self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
		self.seed = random.seed(seed)

	def add(self, state, action, reward, next_state, done):
		"""Add a new experience to memory."""
		e = self.experience(state, action, reward, next_state, done)
		self.memory.append(e)

	def sample(self):
		"""Randomly sample a batch of experiences from memory."""
		experiences = random.sample(self.memory, k=self.batch_size)

		states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
		actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
		rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
		next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
		dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

		return (states, actions, rewards, next_states, dones)

	def __len__(self):
		"""Return the current size of internal memory."""
		return len(self.memory)
		
		
shared_buffer = ReplayBuffer(4, BUFFER_SIZE, BATCH_SIZE, 0)

def copy_weights(source_network, target_network):
	"""Copy source network weights to target"""
	for target_param, source_param in zip(target_network.parameters(), source_network.parameters()):
		target_param.data.copy_(source_param.data)


class Agent():
	#''' DDPG agent '''
	def __init__(self, state_size, action_size, seed):
		super(Agent, self).__init__()
		self.state_size = state_size
		self.action_size = action_size
		hidden_layers = [50, 50]
		self.random_seed = random.seed(seed)
		
		self.actor_local = Actor(state_size, action_size, seed, hidden_layers).to(device)
		self.actor_target = Actor(state_size, action_size, seed, hidden_layers).to(device)
		self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=actor_lr)
		copy_weights(self.actor_local, self.actor_target)
		
		# Critic networks
		self.critic_local = Critic(state_size, action_size, seed, hidden_layers).to(device)
		self.critic_target = Critic(state_size, action_size, seed, hidden_layers).to(device)
		self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=critic_lr)
		copy_weights(self.critic_local, self.critic_target)

		self.t_step = 0

	def step(self, state, action, reward, next_state, done):
		# Write the replay buffer so it is shared between all agents!
		#self.memory.add(state, action, reward, next_state, done)
		#global shared memory
		#shared_memory.shared_buffer.add(state, action, reward, next_state, done)
		shared_buffer.add(state, action, reward, next_state, done)

		# update time steps
		self.t_s = (self.t_step + 1) % UPDATE_EVERY
		if self.t_step == 0:
			# time to learn again
			# provided that there are enough
			#if len(shared_memory.shared_buffer) > BATCH_SIZE:
			if len(shared_buffer) > BATCH_SIZE:
				#experiences = self.memory.sample()
				#experiences = shared_memory.shared_buffer.sample()
				experiences = shared_buffer.sample()
				self.learn(experiences, GAMMA)

	def act(self, state):
		# Make current state into a Tensor that can be passes as input to the network
		#state = torch.from_numpy(state).float().unsqueeze(0).to(device)
		#state = torch.from_numpy(np.expand_dims(state, 0)).float().to(device)
		#state = torch.from_numpy(np.expand_dims(state, 0)).float().to(device)
		state = torch.from_numpy(state).float().to(device)

		# Set network in evaluation mode to prevent things like dropout from happening
		self.actor_local.eval()

		# We don't need to backpropagate, so turn off the autograd engine
		with torch.no_grad():
			# Do a forward pass through the network
			action_values = self.actor_local(state).cpu().data.numpy()

		# Put network back into training mode
		self.actor_local.train()

		return np.clip(action_values, -1, 1)
		
		
		# state = torch.from_numpy(np.expand_dims(state, 0)).float().to(device)
		# actor_local.eval()
		# with torch.no_grad():
			# action = actor_local(state).cpu().data.numpy()
		# actor_local.train()
		# if add_noise:
			# action += noise.sample()
		# return np.clip(action, -1, 1)


	def learn(self, experiences, gamma):
		''' Q_targets = r + γ * critic_target(next_state, actor_target(next_state)) '''

		states, actions, rewards, next_states, dones = experiences

		# ------------------------ Update Critic Network ------------------------ #
		next_actions = self.actor_target(next_states)
		Q_targets_prime = self.critic_target(next_states, next_actions)

		# Compute y_i
		Q_targets = rewards + (gamma * Q_targets_prime * (1 - dones))

		# Compute the critic loss
		Q_expected = self.critic_local(states, actions)
		critic_loss = F.mse_loss(Q_expected, Q_targets)
		# Minimise the loss
		self.critic_optimizer.zero_grad() # Reset the gradients to prevent accumulation
		critic_loss.backward()            # Compute gradients
		torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
		self.critic_optimizer.step()      # Update weights

		# ------------------------ Update Actor Network ------------------------- #
		# Compute the actor loss
		actions_pred = self.actor_local(states)
		actor_loss = -self.critic_local(states, actions_pred).mean()

		# Minimise the loss
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()


		# ------------------------ Update Target Networks ----------------------- #
		self.soft_update(critic_local, critic_target, TAU)
		self.soft_update(actor_local, actor_target, TAU)
		
	def soft_update(self, local_model, target_model, tau):
		"""Soft update model parameters.
		θ_target = τ*θ_local + (1 - τ)*θ_target

		Params
		======
			local_model (PyTorch model): weights will be copied from
			target_model (PyTorch model): weights will be copied to
			tau (float): interpolation parameter 
		"""
		for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
			target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)