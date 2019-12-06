import numpy as np

# Pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def hidden_init(layer):
	fan_in = layer.weight.data.size()[0]
	lim = 1. / np.sqrt(fan_in)
	return (-lim, lim)


class Actor(nn.Module):
	"""Actor (Policy) Model."""

	def __init__(self, state_size, action_size, seed, hidden_layers, fc1_units=200, fc2_units=50):
		"""Initialize parameters and build model.
		Params
		======
			state_size (int): Dimension of each state
			action_size (int): Dimension of each action
			seed (int): Random seed
			hidden_layers: list of the sizes of the hidden layers
		"""
		#super(Actor, self).__init__()
		#self.seed = torch.manual_seed(seed) #Setting the seed for the random generator in Pytorch
		# Sets up all layers with linear transformations
		# Add the input layer to a hidden layer
		#self.hidden_layers = nn.ModuleList([nn.Linear(state_size, hidden_layers[0])])
		
		# Add the remaining hidden layers
		#layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
		#self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
		
		# Add the output layer
		#self.output = nn.Linear(hidden_layers[-1], action_size)
		
		super(Actor, self).__init__()
		self.seed = torch.manual_seed(seed)

		self.bn0 = nn.BatchNorm1d(state_size)
		self.fc1 = nn.Linear(state_size, fc1_units)
		self.bn1 = nn.BatchNorm1d(fc1_units)
		self.fc2 = nn.Linear(fc1_units, fc2_units)
		self.bn2 = nn.BatchNorm1d(fc2_units)
		self.fc3 = nn.Linear(fc2_units, action_size)
		self.reset_parameters()
		
		
	def reset_parameters(self):
		self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
		self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
		self.fc3.weight.data.uniform_(-3e-3, 3e-3)

	# def forward(self, state):
		# # Forward propagation
		# # relu is a non-linear activation function
		# # F is the pyTorch functional module
		# x = state
		# for linear in self.hidden_layers: # Pass through hidden layers
			# x = F.relu(linear(x))
			
		# return F.tanh(self.output(x))	 # Return the output of the output layer

	def forward(self, state):
		"""Build an actor (policy) network that maps states -> actions."""
		x = self.bn0(state)
		x = F.relu(self.bn1(self.fc1(x)))
		x = F.relu(self.bn2(self.fc2(x)))
		return torch.tanh(self.fc3(x))


class Critic(nn.Module):
	"""Actor (Policy) Model."""

	def __init__(self, state_size, action_size, seed, hidden_layers, fc1_units=400, fc2_units=50):
		"""Initialize parameters and build model.
		Params
		======
			state_size (int): Dimension of each state
			action_size (int): Dimension of each action
			seed (int): Random seed
			hidden_layers: list of the sizes of the hidden layers
		"""
		#super(Critic, self).__init__()
		#self.seed = torch.manual_seed(seed) #Setting the seed for the random generator in Pytorch
		# Sets up all layers with linear transformations
		# Add the input layer to a hidden layer
		#self.hidden_layers = nn.ModuleList([nn.Linear(state_size, hidden_layers[0])])
		
		# Add the remaining hidden layers
		#layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
		#self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
		
		# Add the output layer
		#self.output = nn.Linear(hidden_layers[-1], dim=1)
		#self.fool = nn.Linear(state_size, fc1_units)
		#self.fc2 = nn.Linear(fc1_units+action_size, fc2_units)
		#self.fc3 = nn.Linear(fc2_units, 1)
		
		super(Critic, self).__init__()
		self.seed   = torch.manual_seed(seed)
		self.fcs1   = nn.Linear(state_size, fc1_units)
		self.bn1    = nn.BatchNorm1d(fc1_units)
		self.d1     = nn.Dropout(p=0.1)
		self.fc2    = nn.Linear(fc1_units+action_size, fc2_units)
		self.d2     = nn.Dropout(p=0.1)
		self.fc3    = nn.Linear(fc2_units, 1)
		
		
	def reset_parameters(self):
		self.fcs1.weight.data.uniform_(*hidden_init(self.fc1))
		self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
		self.fc3.weight.data.uniform_(-3e-3, 3e-3)
		self.fcs1.bias.data.fill_(0.1)
		self.fc2.bias.data.fill_(0.1)
		self.fc3.bias.data.fill_(0.1)

	def forward(self, state, action):
		# Forward propagation
		# relu is a non-linear activation function
		# F is the pyTorch functional module

		# This is very important, gotta test this cat-thing
		# x = torch.cat([state, action], 1)
		# for linear in self.hidden_layers: # Pass through hidden layers
			# x = F.relu(linear(x))
			
		# return self.output(x) # Return the output of the output layer
		
		#if state.dim() == 1:
		#	state = torch.unsqueeze(state,0)
		
		#xp = F.relu(self.fcs1(state))
		#x = torch.cat((xp, action), dim=1)
		#x = self.fc2(x)
		#x = F.relu(x)
		#return self.fc3(x)
		
		
		if state.dim() == 1:
			state = torch.unsqueeze(state,0)

		xs = self.d1(self.bn1(F.relu(self.fcs1(state))))
		x = torch.cat((xs, action), dim=1)
		x = self.d2(F.relu(self.fc2(x)))
		return self.fc3(x)
		