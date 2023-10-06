import torch.nn as nn

class temporal_CNN1(nn.Module):
    def __init__(self):
        super(temporal_CNN1, self).__init__()
        self.features1 = nn.Sequential(
            nn.Conv2d(10, 96, kernel_size=7, stride=2, padding=1,bias=False),nn.PReLU(),nn.MaxPool2d(3, stride=2),
            nn.Conv2d(96,256, kernel_size=5, stride=2, padding=1, bias=False), nn.PReLU(), nn.MaxPool2d(3, stride=2),
            nn.Conv2d(256,512, kernel_size=3, stride=1, padding=1, bias=False), nn.PReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False), nn.PReLU(), nn.MaxPool2d(2, stride=2) )

        self.fc8_tp1 = nn.Sequential(nn.Linear(1000, 500), nn.PReLU(), nn.Linear(500, 64)).cuda()
    def forward(self,ego_optical):
        ego_optical_ouput= self.features1(ego_optical)
        return ego_optical_ouput



class temporal_CNN2(nn.Module):
    def __init__(self):
        super(temporal_CNN2, self).__init__()
        self.features2 = nn.Sequential(
            nn.Conv2d(10, 96, kernel_size=7, stride=2, padding=1, bias=False), nn.PReLU(), nn.MaxPool2d(3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=1, bias=False), nn.PReLU(), nn.MaxPool2d(3, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False), nn.PReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False), nn.PReLU(), nn.MaxPool2d(2, stride=2))
        self.fc8_tp2 = nn.Sequential(nn.Linear(1000, 500), nn.PReLU(), nn.Linear(500, 64)).cuda()
    def forward(self,front_optical):
        front_optical_ouput = self.features2(front_optical)
        return front_optical_ouput


class temporal_CNN(nn.Module):
    def __init__(self):
        self.net1=temporal_CNN1()
        self.net2=temporal_CNN2()
        self.conv_5=nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False), nn.PReLU()) #not_sure about 1st input
        self.conv_6=nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False), nn.PReLU(), nn.MaxPool2d(2, stride=2))
        self.fc7 = nn.Sequential(nn.Linear(64 * 4 * 4, 1000),nn.PReLU()) #not_sure about 64,4,4

    def forward(self,ego_optical,front_optical):
        ego_optical_output=self.net1(ego_optical)
        ego_optical_output=self.conv_5(ego_optical_output)
        ego_optical_output=self.conv_6(ego_optical_output)
        ego_optical_output=self.fc7(ego_optical_output)
        ego_optical_output=self.net1.fc8_tp1(ego_optical_output)


        front_optical_output = self.net2(front_optical)
        front_optical_output = self.conv_5(front_optical_output)
        front_optical_output = self.conv_6(front_optical_output)
        front_optical_output = self.fc7(front_optical_output)
        front_optical_output = self.net2.fc8_tp2(front_optical_output)

        return ego_optical_output,front_optical_output

