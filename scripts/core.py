import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
import torch.nn.functional as F


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

def normalization_trick(x):
    x = np.array(x, dtype=np.float32)
    global_sum = np.sum(x) 
    global_n = len(x)
    mean = global_sum / global_n

    global_sum_sq = np.sum((x - mean)**2)
    std = np.sqrt(global_sum_sq / global_n)  # compute global std    
    return mean, std

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    # adv = np.zeros(len(x))
    # last_gae_lam = 0
    # for i in reversed(range(len(x))):
    #     last_gae_lam = x[i] + discount*last_gae_lam
    #     adv[i] = last_gae_lam
    # return adv
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):
    
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        # self.log_std = torch.as_tensor(log_std).to(device="cuda:0")#torch.nn.Parameter(torch.as_tensor(log_std))
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
        # for p in self.mu_net.parameters():
        #     p.data.fill_(0)
        # print("stop")

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)
        # return Normal(mu.clip(min=-5.,max=5.), std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.



class MLPActorCritic(nn.Module):


    def __init__(self, observation_space, action_space, 
                 hidden_sizes=(128,128), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[1]

        # policy builder depends on action space
        # if isinstance(action_space, Box):
        # BP: I'm only working with continuous action spaces
        self.pi = MLPGaussianActor(obs_dim, action_space.shape[1], hidden_sizes, activation)
        # elif isinstance(action_space, Discrete):
        #     self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        # return a, v, logp_a
        return a, v.cpu().numpy(), logp_a.cpu().numpy()

    def act(self, obs):
        return self.step(obs)[0]

class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=11, stride=4, padding=5)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        return x

class GRUNetwork1(nn.Module):
    def __init__(self, input_size, hidden_size, gru_layers):
        super(GRUNetwork1, self).__init__()
        self.gru1 = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=gru_layers)

    def forward(self, x, hidden_state):
        gru_out, hidden_state = self.gru1(x, hidden_state)
        return gru_out, hidden_state
    
class GRUNetwork2(nn.Module):
    def __init__(self, input_size, hidden_size=256, gru_layers=1):
        super(GRUNetwork2, self).__init__()
        self.gru2 = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=gru_layers)

    def forward(self, x, hidden_state):
        gru_out, hidden_state = self.gru2(x, hidden_state)
        return gru_out, hidden_state