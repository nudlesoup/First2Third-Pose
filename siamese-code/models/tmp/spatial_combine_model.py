import torch
import torch.nn as nn
from image_resnet import *

class spatial_Res1(nn.Module):
    def __init__(self):
        super(spatial_Res1, self).__init__()
        self.res1sp = resnet50(pretrained= True).cuda()
        self.fc8_sp1 = nn.Sequential(nn.Linear(1000, 500), nn.PReLU(), nn.Linear(500, 64)).cuda()

    def forward(self,ego_image):
        ego_image_output= self.res1sp(ego_image)
        ego_image_output=self.fc8_sp1(ego_image_output)
        return ego_image_output



class spatial_Res2(nn.Module):
    def __init__(self):
        super(spatial_Res2, self).__init__()
        self.res2sp = resnet50(pretrained=True).cuda()
        self.fc8_sp2 = nn.Sequential(nn.Linear(1000, 500), nn.PReLU(), nn.Linear(500, 64)).cuda()

    def forward(self,front_image):
        front_image_output = self.res2sp(front_image)
        front_image_output=self.fc8_sp2(front_image_output)
        return front_image_output



class Spatial_Combine(nn.Module):
    def __init__(self):
        super(Spatial_Combine, self).__init__()
        self.ego_sp_model = spatial_Res1()
        self.front_sp_model = spatial_Res2()

    def forward(self, ego_images, front_images):
        x1 = self.ego_sp_model(ego_images)
        x1 = x1.view(x1.size(0), -1)


        x2 = self.front_sp_model(front_images)
        x2 = x2.view(x2.size(0), -1)

        return x1,x2
