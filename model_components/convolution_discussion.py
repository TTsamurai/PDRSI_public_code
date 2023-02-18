import torch.nn as nn


class ConvDiscussion(nn.Module):
    def __init__(self):
        super(ConvDiscussion, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, stride=2)
        self.dropout = nn.Dropout()
        self.conv1 = nn.Conv2d(1, 1, 4, stride=2)
        self.conv2 = nn.Conv2d(1, 1, 4)
        self.conv3 = nn.Conv2d(1, 1, 4)

        self.fc1 = nn.Linear(61 * 94, 768)

    def forward(self, x):
        # channel = 1
        x = x.unsqueeze(dim=1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        return x


class ConvTechnicalDiscussion(nn.Module):
    def __init__(self, n_length):
        super(ConvTechnicalDiscussion, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, stride=2)
        self.dropout = nn.Dropout()
        self.conv1 = nn.Conv2d(1, 1, 4, stride=2)
        self.conv2 = nn.Conv2d(1, 1, 4)
        self.conv3 = nn.Conv2d(1, 1, 4)
        self.n_length = n_length
        if n_length == 1:
            self.fc1 = nn.Linear(375, 768)
        elif n_length == 7:
            self.fc1 = nn.Linear(671, 768)
        elif n_length == 14:
            self.fc1 = nn.Linear(1464, 768)
        else:
            raise ValueError("N_length Check")

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        if self.n_length != 1:
            x = self.conv2(x)
            x = self.relu(x)
            x = self.pool(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        return x
