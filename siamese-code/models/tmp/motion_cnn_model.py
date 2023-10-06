import torch.nn as nn
import torch.nn.functional as F
import os
from flow_resnet import *



class temporal_Res1(nn.Module):
    def __init__(self):
        super(temporal_Res1, self).__init__()
        self.res1 = flow_resnet50(pretrained= True, channel=10).cuda()
        self.fc8_1 = nn.Sequential(nn.Linear(1000, 500), nn.PReLU(),nn.Linear(500, 64)).cuda()


    def forward(self,ego_optical):
        ego_optical_ouput= self.res1(ego_optical)
        ego_optical_ouput=self.fc8_1(ego_optical_ouput)
        return ego_optical_ouput



class temporal_Res2(nn.Module):
    def __init__(self):
        super(temporal_Res2, self).__init__()
        self.res2 = flow_resnet50(pretrained=True, channel=10).cuda()
        self.fc8_2 = nn.Sequential(nn.Linear(1000, 500), nn.PReLU(),nn.Linear(500, 64)).cuda()


    def forward(self,front_optical):
        front_optical_output = self.res2(front_optical)
        front_optical_output=self.fc8_2(front_optical_output)
        return front_optical_output


class temporal_Res(nn.Module):
    def __init__(self):
        self.net1=temporal_Res1()
        self.net2=temporal_Res2()

    def forward(self,ego_optical,front_optical):
        ego_optical_output=self.net1(ego_optical)
        front_optical_output = self.net2(front_optical)
        return ego_optical_output,front_optical_output

