import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim


class Actor(nn.Module):
    """Actor Network"""
    def __init__(self, obs_dim, act_dim, seed):
        """Initialize parameters and build model.
        Params
        ======
            obs_dim (int): Dimension of all observation
            act_dim (int): Dimension of each action
            seed (int): Random seed
        """
        super(Actor, self).__init__()
        hid0_size = 800
        hid1_size = 400

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.seed = torch.manual_seed(seed)

        self.fc0 = nn.Linear(obs_dim, hid0_size)
        self.fc1 = nn.Linear(hid0_size, hid1_size)
        self.fc2 = nn.Linear(hid1_size, act_dim)

        self.reset_parameters()

    def reset_parameters(self):
        self.fc0.weight.data.uniform_(*hidden_init(self.fc0))
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, obs):
        """Build an actor network that maps states -> actions."""
        hid0 = F.tanh(self.fc0(obs))
        hid1 = F.tanh(self.fc1(hid0))
        means = F.tanh(self.fc2(hid1))
        return means


class Critic(nn.Module):
    """Critic Network"""
    def __init__(self, obs_dim, act_dim, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(Critic, self).__init__()
        hid0_size = 800
        hid1_size = 400
        hid2_size = 400

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.seed = torch.manual_seed(seed)

        self.fc0 = nn.Linear(obs_dim, hid0_size)
        self.fc1 = nn.Linear(hid0_size, hid1_size)
        self.act_fc0 = nn.Linear(act_dim, hid1_size)
        self.fc2 = nn.Linear(hid1_size+hid1_size, hid2_size)
        self.fc3 = nn.Linear(hid2_size, 1)

        self.reset_parameters()

    def reset_parameters(self):
        self.fc0.weight.data.uniform_(*hidden_init(self.fc0))
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.act_fc0.weight.data.uniform_(*hidden_init(self.act_fc0))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, obs, action):
        """Build a critic network that maps (state,action) pair -> Q_value."""

        hid0 = F.selu((self.fc0(obs)))
        hid1 = F.selu(self.fc1(hid0))
        a1 = self.act_fc0(action)
        concat = torch.cat((hid1, a1), dim=1)
        hid2 = F.selu(self.fc2(concat))
        q = self.fc3(hid2)
        return q
