import torch
import torch.nn as nn
from models.image_resnet import *
import torch.nn as nn
from models.flow_resnet import *

class Two_res_spatial_res1(nn.Module):
    def __init__(self):
        super(Two_res_spatial_res1, self).__init__()
        self.res1sp = resnet50(pretrained= True)

    def forward(self,ego_image):
        x1 = self.res1sp(ego_image)
        return x1

class Two_res_spatial_res2(nn.Module):
    def __init__(self):
        super(Two_res_spatial_res2, self).__init__()
        self.res2sp = resnet50(pretrained=True)

    def forward(self,front_image):
        x1 = self.res2sp(front_image)
        return x1


class Two_res_temporal_res1(nn.Module):
    def __init__(self):
        super(Two_res_temporal_res1, self).__init__()
        self.res1 = flow_resnet18(pretrained= True)
    def forward(self,ego_optical):
        x1 = self.res1(ego_optical)
        return x1

class Two_res_temporal_res2(nn.Module):
    def __init__(self):
        super(Two_res_temporal_res2, self).__init__()
        self.res2 = flow_resnet18(pretrained=True)

    def forward(self,front_optical):
        x1 = self.res2(front_optical)
        return x1



class Two_Stream_Res(nn.Module):
    def __init__(self):
        super(Two_Stream_Res, self).__init__()
        self.two_ego_sp_model = Two_res_spatial_res1()
        self.two_front_sp_model = Two_res_spatial_res2()
        self.two_temporal_net1 = Two_res_temporal_res1()
        self.two_temporal_net2 = Two_res_temporal_res2()
        self.fc8_1 = nn.Sequential(nn.Linear(2000, 500), nn.PReLU(), nn.Linear(500, 64))
        self.fc8_2 = nn.Sequential(nn.Linear(2000, 500), nn.PReLU(), nn.Linear(500, 64))



    def forward(self, ego_images, front_images,ego_optical,front_optical):
        ego_images_output = self.two_ego_sp_model(ego_images)
        front_images_output = self.two_front_sp_model(front_images)
        ego_optical_output = self.two_temporal_net1(ego_optical)
        front_optical_output = self.two_temporal_net2(front_optical)
        ego_embed = torch.cat((ego_images_output, ego_optical_output), dim=1)
        front_embed = torch.cat((front_images_output, front_optical_output), dim=1)
        ego_embed=self.fc8_1(ego_embed)
        front_embed = self.fc8_1(front_embed)
        return ego_embed,front_embed
