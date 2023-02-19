import torch.nn as nn


class ConvDiscussion(nn.Module):
    def __init__(self):
        super(ConvDiscussion, self).__init__()
        self.seq1 = nn.Sequential(
            nn.Conv2d(1, 1, 4, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(1, 1, 4),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )
        self.seq2 = nn.Sequential(
            nn.Linear(61 * 94, 768),
            nn.ReLU(),
        )
        
    def forward(self, x):
        x = self.seq1(x.unsqueeze(dim=1))
        x = x.view(x.size()[0], -1)
        x = self.seq2(x)
        return x


class ConvTechnicalDiscussion(nn.Module):
    def __init__(self, n_length):
        super(ConvTechnicalDiscussion, self).__init__()
        self.seq1 = nn.Sequential(
            nn.Conv2d(1, 1, 4, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )
        self.n_length = n_length
        if self.n_length != 1:
            self.seq2 = nn.Sequential(
                nn.Conv2d(1, 1, 4),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
            )
        fc_layers = {
            1: nn.Linear(375, 768),
            7: nn.Linear(660, 768),
            14: nn.Linear(1464, 768),
        }
        self.fc_layer = nn.Sequential(
            fc_layers[n_length],
            nn.ReLU(),
        )
        
    def forward(self, x):
        x = self.seq1(x.unsqueeze(dim=1))
        if self.n_length != 1:
            x = self.seq2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc_layer(x)
        return x
