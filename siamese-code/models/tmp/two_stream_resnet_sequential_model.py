import torch
import torch.nn as nn
from image_resnet import *
import torch.nn as nn
from flow_resnet import *

class Two_res_spatial_res1(nn.Module):
    def __init__(self):
        super(Two_res_spatial_res1, self).__init__()
        self.res1sp = resnet50(pretrained= True)

    def forward(self,ego_image):
        ego_image_output = []
        for batch in ego_image:
            with torch.no_grad():
                features = self.res1sp(batch)
            #features = features.reshape(features.size(0), -1)
            ego_image_output.append(features)
        ego_image_output = torch.stack(ego_image_output, dim=0)

        return ego_image_output

class Two_res_spatial_res2(nn.Module):
    def __init__(self):
        super(Two_res_spatial_res2, self).__init__()
        self.res2sp = resnet50(pretrained=True)

    def forward(self,front_image):
        #front_image_output = self.res2sp(front_image)
        front_image_output = []
        for batch in front_image:
            with torch.no_grad():
                features = self.res2sp(batch)
            # features = features.reshape(features.size(0), -1)
            front_image_output.append(features)
        front_image_output = torch.stack(front_image_output, dim=0)
        return front_image_output

class Two_res_temporal_res1(nn.Module):
    def __init__(self):
        super(Two_res_temporal_res1, self).__init__()
        self.res1 = flow_resnet50(pretrained= True, channel=100)

    def forward(self,ego_optical):
        #ego_optical_ouput= self.res1(ego_optical)
        ego_optical_output = []
        for batch in ego_optical:
            with torch.no_grad():
                features = self.res1(batch)
            # features = features.reshape(features.size(0), -1)
            ego_optical_output.append(features)
        ego_optical_ouput = torch.stack(ego_optical_output, dim=0)
        return ego_optical_output

class Two_res_temporal_res2(nn.Module):
    def __init__(self):
        super(Two_res_temporal_res2, self).__init__()
        self.res2 = flow_resnet50(pretrained=True, channel=100)

    def forward(self,front_optical):
        # front_optical_output = self.res2(front_optical)
        front_optical_output = []
        for batch in front_optical:
            with torch.no_grad():
                features = self.res2(batch)
            # features = features.reshape(features.size(0), -1)
            front_optical_output.append(features)
        front_optical_output = torch.stack(front_optical_output, dim=0)
        return front_optical_output


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, num_layers):
        super(DecoderRNN, self).__init__()

        self.lstm_upp = nn.LSTM((embed_size * 2) , hidden_size, num_layers,batch_first=True)
        self.lstm_low = nn.LSTM((embed_size * 2), hidden_size, num_layers, batch_first=True)

        self.linear = nn.Linear(hidden_size, (embed_size))
        self.embed_size = embed_size

    def forward_upper(self,features,states=None):
        first = True
        prev_embed = torch.zeros([features.shape[0], 1, self.embed_size]).cuda().float()
        # embed()

        for i in range(features.shape[1]):
            embeddings = torch.cat((prev_embed, features[:, i, :].unsqueeze(1)), 2)
            hiddens, _ = self.lstm_upp(embeddings, states)
            y_pred = self.linear(hiddens.squeeze(1))
            prev_embed = y_pred
            prev_embed = prev_embed.unsqueeze(1)
            if first:
                outputs = y_pred
                first = False
            else:
                outputs = torch.cat((outputs, y_pred))

        return outputs

    def forward_lower(self,features,states=None):
        first = True
        prev_embed = torch.zeros([features.shape[0], 1, self.embed_size]).cuda().float()
        # embed()

        for i in range(features.shape[1]):
            embeddings = torch.cat((prev_embed, features[:, i, :].unsqueeze(1)), 2)
            hiddens, _ = self.lstm_low(embeddings, states)
            y_pred = self.linear(hiddens.squeeze(1))
            prev_embed = y_pred
            prev_embed = prev_embed.unsqueeze(1)
            if first:
                outputs = y_pred
                first = False
            else:
                outputs = torch.cat((outputs, y_pred))

        return outputs

    def forward(self, upper_features,lower_features, states=None):
        """ decode image feature vectors and generate pose sequences """
        upper_embed= self.forward_upper(upper_features)
        lower_embed= self.forward_lower(lower_features)
        return upper_embed,lower_embed



class Two_Stream_Res(nn.Module):
    def __init__(self, embed_size, hidden_size, num_layers):
        super(Two_Stream_Res, self).__init__()
        self.two_ego_sp_model = Two_res_spatial_res1()
        self.two_front_sp_model = Two_res_spatial_res2()
        self.two_temporal_net1 = Two_res_temporal_res1()
        self.two_temporal_net2 = Two_res_temporal_res2()
        self.fc8_1 = nn.Sequential(nn.Linear(2000, 500), nn.PReLU(), nn.Linear(500, 64))
        self.fc8_2 = nn.Sequential(nn.Linear(2000, 500), nn.PReLU(), nn.Linear(500, 64))
        self.decoder = DecoderRNN(embed_size, hidden_size, num_layers)


    def forward(self, ego_images, front_images,ego_optical,front_optical):
        ego_images_output = self.two_ego_sp_model(ego_images)
        front_images_output = self.two_front_sp_model(front_images)
        ego_optical_output = self.two_temporal_net1(ego_optical)
        front_optical_output = self.two_temporal_net2(front_optical)
        ego_embed = torch.cat((ego_images_output, ego_optical_output), dim=1)
        front_embed = torch.cat((front_images_output, front_optical_output), dim=1)
        ego_embed=self.fc8_1(ego_embed)
        front_embed = self.fc8_1(front_embed)
        upper_embed,lower_embed = self.decoder(ego_embed, front_embed)
        return upper_embed,lower_embed