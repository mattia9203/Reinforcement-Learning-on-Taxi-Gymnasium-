import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, outputs):
        super(DQN, self).__init__()
        self.emb = nn.Embedding(500, 32)
        self.l1 = nn.Linear(32, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, outputs)

    def forward(self, x):
        x = F.relu(self.l1(self.emb(x)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x