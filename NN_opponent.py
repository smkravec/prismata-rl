import torch
from torch.distributions.categorical import Categorical
from model import MLPBase, D2RLNet, Discrete
import numpy as np

class NN_opponent:
    def __init__(self, model_dir=None, model_type='mlp', cards=4, hidden_dim=64, num_layers=4):
        self.hidden_dim=hidden_dim
        self.num_layers=num_layers
        self.cards= cards
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_dir=model_dir
        
        if self.cards==4:
            self.num_actions=14
            self.obs_shape=(30,)
            self.num_obs = 30
        elif self.cards==11:
            self.num_actions=32
            self.obs_shape=(82,)
            self.num_obs = 82
        else:
            raise ValueError('Cards Not Accepted')
            
        self.dist = Discrete(self.num_actions)
        
        if model_type == "mlp":
            self.model = MLPBase(self.num_obs, self.num_actions, self.dist, self.hidden_dim)
        elif args.model == "d2rl":
            self.model= D2RLNet(self.num_obs, self.num_actions, self.dist, self.hidden_dim, self.num_layers)
        else:
            raise ValueError('Model Not Supported')
        
        self.model.load_state_dict(torch.load(f'{self.model_dir}/model.h5', map_location=torch.device(self.device)))
        self.model.to(self.device)
        
        self.mean = np.load(f"{self.model_dir}/mean.npy")
        self.var = np.load(f"{self.model_dir}/var.npy")
    
    def getAction(self, gamestate):
        obs, legal, done = gamestate.toVector(), gamestate.getAbstractActionsVector(), gamestate.isGameOver()
        #Normalize observation
        obs=obs-self.mean
        obs= obs / np.sqrt(self.var + 1e-8)
        #get action probabilities
        obs_torch=torch.tensor(obs).to(self.device).float()
        legal=torch.tensor(legal).float()
        obs_torch=torch.reshape(obs_torch, (1,len(obs_torch)))
        legal=torch.reshape(legal,(1,len(legal)))
        _, probs = self.model(obs_torch)
        #mask illegal actions
        probs=probs.detach().cpu()
        probs=torch.mul(probs,legal)
        probs_sum=torch.sum(probs, dim=1)
        probs = torch.einsum('ij,i->ij', probs , 1/probs_sum)
        #sample action
        distribution = torch.distributions.Categorical(probs=probs)
        action = distribution.sample().item()
        return action