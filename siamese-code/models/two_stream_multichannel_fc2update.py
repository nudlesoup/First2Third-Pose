import torch
import torch.nn as nn
import models.multichannel_resnet
from models.multichannel_resnet import get_arch as Resnet

class Two_stream_ego(nn.Module):
    def __init__(self):
        super(Two_stream_ego, self).__init__()
        resnet50_ego = Resnet(50, 23)
        self.res1_ego = resnet50_ego(True)

    def forward(self,ego_mix):
        x1 = self.res1_ego(ego_mix)
        return x1


class Two_stream_fro(nn.Module):
    def __init__(self):
        super(Two_stream_fro, self).__init__()
        resnet50_fro = Resnet(50, 23)
        self.res1_fro = resnet50_fro(True)

    def forward(self,fro_mix):
	    x1 = self.res1_fro(fro_mix)
	    return x1



class Two_Stream_Res(nn.Module):
    def __init__(self):
        super(Two_Stream_Res, self).__init__()
        self.two_ego_model = Two_stream_ego()
        self.two_front_model = Two_stream_fro()
        self.fc8_1 = nn.Sequential(nn.Linear(1000, 500), nn.PReLU(), nn.Linear(500, 64))
        self.fc8_2 = nn.Sequential(nn.Linear(1000, 500), nn.PReLU(), nn.Linear(500, 64))



    def forward(self, ego_images, front_images,ego_optical,front_optical):
        ego_mix = torch.cat((ego_images, ego_optical), dim=1)
        fro_mix = torch.cat((front_images, front_optical), dim=1)
        ego_embed = self.two_ego_model(ego_mix)
        front_embed = self.two_front_model(fro_mix)
        ego_embed=self.fc8_1(ego_embed)
        front_embed = self.fc8_2(front_embed)
        return ego_embed,front_embed
