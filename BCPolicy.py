import torch.nn as nn

class BCPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(512, 512)):
        super().__init__()

        layers = []
        in_dim = obs_dim

        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.ReLU()]
            in_dim = h
        
        layers += [nn.Linear(in_dim, act_dim), nn.Tanh()]
        self.net == nn.Sequential(*layers)

    def forward(self, obs):
        return self.net(obs)
    
obs_dim = 3 + 4 + 2
act_dim = 7
policy = BCPolicy(obs_dim, act_dim).cuda()