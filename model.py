import sys
sys.path.append('..')
from utils import *
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch import einsum
from einops import reduce

#Based on D2RL https://arxiv.org/abs/2010.09163
#Max layers 8

#MLPBase from https://github.com/zplizzi/pytorch-ppo/

# Initialize Policy weights for D2RL network

def init(module, weight_init, bias_init, gain=1):
    """Helper to initialize a layer weight and bias."""
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class Flatten(nn.Module):
    """Helper to flatten a tensor."""
    def forward(self, x):
        return x.view(x.size(0), -1)

class MLPBase(nn.Module):
    """Basic multi-layer linear model."""
    def __init__(self, num_inputs, num_outputs, hidden_size=64):
        super().__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        init2_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
            init2_(nn.Linear(hidden_size, num_outputs)))

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, 1)))

    def forward(self, x):
        value = self.critic(x)
        action_logits = self.actor(x)
        return value, action_logits


class D2RLNet(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_dim=512, num_layers=4):
        super(D2RLNet, self).__init__()

        self.num_layers= num_layers
        in_dim = num_inputs+hidden_dim
        self.apply(weights_init_)

        # Actor architecture
        self.l1_1 = nn.Linear(num_inputs, hidden_dim)
        self.l1_2 = nn.Linear(in_dim, hidden_dim)

        if num_layers > 2:
            self.l1_3 = nn.Linear(in_dim, hidden_dim)
            self.l1_4 = nn.Linear(in_dim, hidden_dim)

        if num_layers > 4:
            self.l1_5 = nn.Linear(in_dim, hidden_dim)
            self.l1_6 = nn.Linear(in_dim, hidden_dim)

        if num_layers == 8:
            self.l1_7 = nn.Linear(in_dim, hidden_dim)
            self.l1_8 = nn.Linear(in_dim, hidden_dim)

        self.out1 = nn.Linear(hidden_dim, num_outputs)

        # Critic architecture
        self.l2_1 = nn.Linear(num_inputs, hidden_dim)
        self.l2_2 = nn.Linear(in_dim, hidden_dim)

        if num_layers > 2:
            self.l2_3 = nn.Linear(in_dim, hidden_dim)
            self.l2_4 = nn.Linear(in_dim, hidden_dim)

        if num_layers > 4:
            self.l2_5 = nn.Linear(in_dim, hidden_dim)
            self.l2_6 = nn.Linear(in_dim, hidden_dim)

        if num_layers == 8:
            self.l2_7 = nn.Linear(in_dim, hidden_dim)
            self.l2_8 = nn.Linear(in_dim, hidden_dim)

        self.out2 = nn.Linear(hidden_dim, 1)

    def forward(self, network_input):
        xu = network_input

        #Actor

        x1 = F.relu(self.l1_1(xu))
        x1 = torch.cat([x1, xu], dim=1)

        x1 = F.relu(self.l1_2(x1))
        if not self.num_layers == 2:
            x1 = torch.cat([x1, xu], dim=1)

        if self.num_layers > 2:
            x1 = F.relu(self.l1_3(x1))
            x1 = torch.cat([x1, xu], dim=1)

            x1 = F.relu(self.l1_4(x1))
            if not self.num_layers == 4:
                x1 = torch.cat([x1, xu], dim=1)

        if self.num_layers > 4:
            x1 = F.relu(self.l1_5(x1))
            x1 = torch.cat([x1, xu], dim=1)

            x1 = F.relu(self.l1_6(x1))
            if not self.num_layers == 6:
                x1 = torch.cat([x1, xu], dim=1)

        if self.num_layers == 8:
            x1 = F.relu(self.l1_7(x1))
            x1 = torch.cat([x1, xu], dim=1)

            x1 = F.relu(self.l1_8(x1))

        action_logits = self.out1(x1)

        #Critic
        x2 = F.relu(self.l2_1(xu))
        x2 = torch.cat([x2, xu], dim=1)

        x2 = F.relu(self.l2_2(x2))
        if not self.num_layers == 2:
            x2 = torch.cat([x2, xu], dim=1)

        if self.num_layers > 2:
            x2 = F.relu(self.l2_3(x2))
            x2 = torch.cat([x2, xu], dim=1)

            x2 = F.relu(self.l2_4(x2))
            if not self.num_layers == 4:
                x2 = torch.cat([x2, xu], dim=1)

        if self.num_layers > 4:
            x2 = F.relu(self.l2_5(x2))
            x2 = torch.cat([x2, xu], dim=1)

            x2 = F.relu(self.l2_6(x2))
            if not self.num_layers == 6:
                x2 = torch.cat([x2, xu], dim=1)

        if self.num_layers == 8:
            x2 = F.relu(self.l2_7(x2))
            x2 = torch.cat([x2, xu], dim=1)

            x2 = F.relu(self.l2_8(x1))

        value = self.out2(x2)

        return value, action_logits

class CategoricalMasked(Categorical):

    def __init__(self, logits: torch.Tensor, mask: Optional[torch.Tensor]= None):
        self.mask = mask
        self.batch, self.nb_action = logits.size()
        if mask is None:
            super(CategoricalMasked, self).__init__(logits=logits)
        else:
            self.mask_value = torch.finfo(logits.dtype).min
            logits.masked_fill_(~self.mask, self.mask_value)
            super(CategoricalMasked, self).__init__(logits=logits)

    def entropy(self):
        if self.mask is None:
            return super().entropy()
        # Elementwise multiplication
        p_log_p = einsum("ij,ij->ij", self.logits, self.probs)
        # Compute the entropy with possible action only
        p_log_p = torch.where(
            self.mask,
            p_log_p,
            torch.tensor(0, dtype=p_log_p.dtype, device=p_log_p.device),
        )
        return -reduce(p_log_p, "b a -> b", "sum", b=self.batch, a=self.nb_action)
