import torch.nn as nn
from image_resnet import *

class Two_temporal_CNN1(nn.Module):
    def __init__(self):
        super(Two_temporal_CNN1, self).__init__()
        self.features1 = nn.Sequential(
            nn.Conv2d(10, 96, kernel_size=7, stride=2, padding=1,bias=False),nn.PReLU(),nn.MaxPool2d(3, stride=2),
            nn.Conv2d(96,256, kernel_size=5, stride=2, padding=1, bias=False), nn.PReLU(), nn.MaxPool2d(3, stride=2),
            nn.Conv2d(256,512, kernel_size=3, stride=1, padding=1, bias=False), nn.PReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False), nn.PReLU(), nn.MaxPool2d(2, stride=2) )
        self.fc8_tp1 = nn.Sequential(nn.Linear(1000, 500), nn.PReLU(), nn.Linear(500, 64)).cuda()

    def forward(self,ego_optical):
        ego_optical_ouput= self.features1(ego_optical)
        return ego_optical_ouput

class Two_temporal_CNN2(nn.Module):
    def __init__(self):
        super(Two_temporal_CNN2, self).__init__()
        self.features2 = nn.Sequential(
            nn.Conv2d(10, 96, kernel_size=7, stride=2, padding=1, bias=False), nn.PReLU(), nn.MaxPool2d(3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=1, bias=False), nn.PReLU(), nn.MaxPool2d(3, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False), nn.PReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False), nn.PReLU(), nn.MaxPool2d(2, stride=2))
        self.fc8_tp2 = nn.Sequential(nn.Linear(1000, 500), nn.PReLU(), nn.Linear(500, 64)).cuda()

    def forward(self,front_optical):
        front_optical_ouput = self.features2(front_optical)
        return front_optical_ouput

class Two_spatial_Res1(nn.Module):
    def __init__(self):
        super(Two_spatial_Res1, self).__init__()
        self.res1sp = resnet50(pretrained= True).cuda()

    def forward(self,ego_image):
        ego_image_output= self.res1sp(ego_image)
        return ego_image_output


class Two_spatial_Res2(nn.Module):
    def __init__(self):
        super(Two_spatial_Res2, self).__init__()
        self.res2sp = resnet50(pretrained=True).cuda()

    def forward(self,front_image):
        front_image_output = self.res2sp(front_image)
        return front_image_output


class Two_Stream(nn.Module):
    def __init__(self):
        super(Two_Stream, self).__init__()
        self.ego_sp_model = Two_spatial_Res1()
        self.front_sp_model = Two_spatial_Res2()
        self.temporal_net1 = Two_temporal_CNN1()
        self.temporal_net2 = Two_temporal_CNN2()
        self.conv_5 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.PReLU())  # not_sure about 1st input
        self.conv_6 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False), nn.PReLU(),
                                    nn.MaxPool2d(2, stride=2))
        self.fc7 = nn.Sequential(nn.Linear(64 * 4 * 4, 1000), nn.PReLU())  # not_sure about 64,4,4

    def forward(self, ego_images, front_images,ego_optical,front_optical):
        ego_optical_output = self.temporal_net1(ego_optical)
        ego_optical_output = self.conv_5(ego_optical_output)
        ego_optical_output = self.conv_6(ego_optical_output)
        ego_optical_output = self.fc7(ego_optical_output)

        front_optical_output = self.temporal_net2(front_optical)
        front_optical_output = self.conv_5(front_optical_output)
        front_optical_output = self.conv_6(front_optical_output)
        front_optical_output = self.fc7(front_optical_output)

        x1 = self.ego_sp_model(ego_images)
        x1 = x1.view(x1.size(0), -1)

        x2 = self.front_sp_model(front_images)
        x2 = x2.view(x2.size(0), -1)



        return x1,x2