import torch.nn as nn
import torch.nn.functional as F

class Decoder_linear(nn.Module):
    def __init__(self, embedding_dim, output_dim=2):
        super().__init__()
        self.layer1 = nn.Linear(embedding_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, embedding):
        h = self.layer1(embedding)
        h = self.softmax(h)
        return h
