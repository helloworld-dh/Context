import torch.nn as nn

class Fc_change(nn.Module):
    def __init__(self, in_channels):
        super(Fc_change, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(2 * in_channels, in_channels),
            nn.ReLU(inplace=True),

            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True),

            nn.Linear(in_channels, 8)
        )

    def forward(self, x):
        return self.fc(x)