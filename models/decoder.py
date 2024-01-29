import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, embedding_dim, output_size=2, hidden_size=128):
        super().__init__()
        self.layer1 = nn.Linear(embedding_dim, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, embedding):
        h = self.layer1(embedding)
        h = F.relu(h)
        h = self.layer2(h)
        prediction = F.softmax(h, dim=1)

        return prediction
