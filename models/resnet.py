import torch
import models.network as model
import torch.nn as nn
from models.fc_change import Fc_change

class ResNet(torch.nn.Module):
    def __init__(self,params):
        super(ResNet, self).__init__()
        # self.fc_change = Fc_change(in_channels=4096)

        if params.network=='resnet18':
            resnet = model.resnet18(params.network)
            resnet.fc = nn.Linear(in_features=resnet.fc.in_features, out_features= 100, bias=True)
        elif params.network=='resnet50':
            resnet = model.resnet18(params.network)
            resnet.fc = nn.Linear(in_features=resnet.fc.in_features, out_features= 100, bias=True)
        self.encoder = resnet

    # def forward(self, uniform_patch, random_patch):
    #     output_fc6_uniform = self.encoder(uniform_patch)
    #     output_fc6_random = self.encoder(random_patch)
    #     output = torch.cat((output_fc6_uniform, output_fc6_random), 1)
    #     output = self.fc_change(output)
    #     return output, output_fc6_uniform, output_fc6_random

    def forward(self, x):
        h = self.encoder(x)
        return h