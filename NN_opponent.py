import torch
from model import MLPBase, D2RLNet, CategoricalMasked
import numpy as np
import prismataengine as p

class NN_opponent(p.PrismataPlayerPython):
    def __init__(self, model_dir=None, model_type='mlp', cards='4', hidden_dim=64, num_layers=4 , player='p2', one_hot = False):
        if player=='p1':
            self.player= p.Players.One
        elif player=='p2':
            self.player= p.Players.Two
        else:
            raise ValueError('Player Not Accepted, use p1 or p2')
        super().__init__(self.player)
        self.hidden_dim=hidden_dim
        self.num_layers=num_layers
        self.cards= cards
        #self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device="cpu" # CUDA is giving me an error declaring this second network, fix later
        self.model_dir=model_dir
        if one_hot == "True":
            self.one_hot = True
        elif one_hot == "False":
            self.one_hot = False
        else:
            raise ValueError('OneHot truth value not parseable')

        if self.cards=="4":
            self.num_actions=14
            if self.one_hot:
                self.num_obs=670
            else:
                self.num_obs = 30
        elif self.cards=="11":
            self.num_actions=32
            if self.one_hot:
                self.num_obs=1242
            else:
                self.num_obs = 82
        else:
            raise ValueError('Cards Not Accepted. Currently implemented: 4 and 11')

        if model_type == "mlp":
            self.model = MLPBase(self.num_obs, self.num_actions, self.hidden_dim)
        elif args.model == "d2rl":
            self.model= D2RLNet(self.num_obs, self.num_actions, self.hidden_dim, self.num_layers)
        else:
            raise ValueError('Model Not Supported. Currently implemented: mlp and d2rl')

        self.model.load_state_dict(torch.load(f'{self.model_dir}/model.h5', map_location=torch.device(self.device)))
        self.model.to(self.device)

        if not self.one_hot: #Assuming Obs_norm is off in onehot mode
            self.mean = np.load(f"{self.model_dir}/mean.npy")
            self.var = np.load(f"{self.model_dir}/var.npy")

    def getAction(self, gamestate):
        obs, legal, done = gamestate.toVector(), gamestate.getAbstractActionsVector(), gamestate.isGameOver()
        #Normalize observation
        if not self.one_hot:
            obs=obs-self.mean
            obs= obs / np.sqrt(self.var + 1e-8)
        else:
            obs=obs.astype('float16')
        #get action probabilities
        obs_torch=torch.tensor(obs).to(self.device).float()
        legal=torch.tensor(legal,  dtype=torch.bool).to(self.device)
        obs_torch=torch.reshape(obs_torch, (1,len(obs_torch)))
        legal=torch.reshape(legal,(1,len(legal)))
        _, logits = self.model(obs_torch)
        #mask illegal actions
        #sample action
        distribution = CategoricalMasked(logits=logits, mask=legal)
        action = distribution.sample().item()
        return action

    def getMove(self, prismata_gamestate, move):
        gamestate=p.GameState(prismata_gamestate, cards=int(self.cards), one_hot=self.one_hot)
        saveActivePlayer = gamestate.activePlayer
        while gamestate.activePlayer == saveActivePlayer:
            action_label = self.getAction(gamestate)
            actionPointer = gamestate.coerceAction(action_label)
            move.append(p.unsafeIntToAction(actionPointer))
            gamestate.doAction(action_label)
