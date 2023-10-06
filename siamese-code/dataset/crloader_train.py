import pickle
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import random
import json
from PIL import Image
import os
import argparse
from torch.utils.data import Dataset, DataLoader
import glob
import sys


class MarshDataset(data.Dataset):
    ' Pose custom dataset compatible with torch.utils.data.DataLoader. '
    def __init__(self, path, mode, stacked_opticalflow, directories, keys, load_frame_count, ego_list, listed_activities,train_transform=None, val_transform=None):
        self.dir_path=path
        self.mode = mode
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.stacked_opticalflow = stacked_opticalflow
        self.img_rows = 224
        self.img_cols = 224
        self.directories=directories
        self.keys=keys
        self.frames_count=load_frame_count
        self.ego_list = ego_list
        self.listed_activities=listed_activities
        self.epoch=0

    def get_optical_flow_front(self, activity, front_frame_id):
        self.activity = activity
        name = self.dir_path + "/" + self.activity
        temporal_folder_ego = name + "/features/opticalflow_front"

        flow = torch.FloatTensor(2 * self.stacked_opticalflow, self.img_rows, self.img_cols)

        start_index=front_frame_id - int((self.stacked_opticalflow)/2)
        end_index = front_frame_id + int((self.stacked_opticalflow)/2)
        for j in range(start_index,end_index):
            counter=j-start_index
            idx = str(j)
            frame_idx = 'flow_x_' + idx.zfill(4)
            frame_idy = 'flow_y_' + idx.zfill(4)
            h_image = temporal_folder_ego + '/' + frame_idx + '.jpg'
            v_image = temporal_folder_ego + '/' + frame_idy + '.jpg'

            imgH = (Image.open(h_image))
            imgV = (Image.open(v_image))

            H = self.val_transform(imgH)
            V = self.val_transform(imgV)

            flow[2 * counter, :, :] = H
            flow[2 * counter + 1, :, :] = V
            imgH.close()
            imgV.close()
        return flow

        
    def get_optical_flow_ego(self, activity, ego_frame_id):
        self.activity = activity
        name = self.dir_path + "/" + self.activity
        temporal_folder_ego = name + "/features/opticalflow_ego"

        flow = torch.FloatTensor(2 * self.stacked_opticalflow, self.img_rows, self.img_cols)

        start_index=ego_frame_id - int((self.stacked_opticalflow)/2)
        end_index = ego_frame_id + int((self.stacked_opticalflow)/2)
        for j in range(start_index,end_index):
            counter=j-start_index
            idx = str(j)
            frame_idx = 'flow_x_' + idx.zfill(4)
            frame_idy = 'flow_y_' + idx.zfill(4)
            h_image = temporal_folder_ego + '/' + frame_idx + '.jpg'
            v_image = temporal_folder_ego + '/' + frame_idy + '.jpg'

            imgH = (Image.open(h_image))
            imgV = (Image.open(v_image))

            H = self.val_transform(imgH)
            V = self.val_transform(imgV)

            flow[2 * counter, :, :] = H
            flow[2 * counter + 1, :, :] = V
            imgH.close()
            imgV.close()
        return flow

    def positive_pair(self,index):

        ego_image_path=self.ego_list[index]
        activity = ego_image_path.split('/')[-4]
        ego_img = Image.open(ego_image_path)
        if self.train_transform is not None:
            ego_img = self.train_transform(ego_img)
        ego_image=ego_img

        ego_split = ego_image_path.split('/')
        ego_split[-2] = "frames_front"
        front_image_path = '/'.join(ego_split)
        fro_img = Image.open(front_image_path)
        if self.train_transform is not None:
            fro_img = self.train_transform(fro_img)
        front_image=fro_img

        id=ego_image_path.split('imxx')[-1].split('.')[0]
        idx=int(id)
        opticalflow_ego=self.get_optical_flow_ego(activity,idx)
        opticalflow_front=self.get_optical_flow_front(activity,idx)

        return ego_image, front_image,opticalflow_ego,opticalflow_front

    def soft_negative_pair(self,index):
        ego_image_path = self.ego_list[index]
        full_activity = ego_image_path.split('/')[-4]
        ego_img = Image.open(ego_image_path)
        if self.train_transform is not None:
            ego_img = self.train_transform(ego_img)
        ego_image = ego_img
        id = ego_image_path.split('imxx')[-1].split('.')[0]
        idx = int(id)
        opticalflow_ego = self.get_optical_flow_ego(full_activity, idx)

        full_activity_split = full_activity.split('_')
        name = full_activity_split[0]
        location = full_activity_split[2]
        prep_key = name + "_" + location
        max_act=len(self.listed_activities[prep_key])-1
        move = random.randint(0,max_act)
        new_activity=self.listed_activities[prep_key][move]
        full_new_activity=name+ "_"+new_activity+"_"+location
        move = random.randint(6, self.frames_count[full_new_activity] - 5)
        while move == idx:
            move = random.randint(6, self.frames_count[full_new_activity] - 5)

        front_idx = move
        fro_id = str(front_idx)
        fro_split = ego_image_path.split('/')
        fro_split[-1] = 'imxx' + fro_id.zfill(4) + '.jpg'
        fro_split[-2] = "frames_front"
        fro_split[-4] = full_new_activity
        front_image_path = '/'.join(fro_split)
        fro_img = Image.open(front_image_path)
        if self.train_transform is not None:
            fro_img = self.train_transform(fro_img)
        front_image = fro_img
        opticalflow_front = self.get_optical_flow_front(full_new_activity, front_idx)
        return ego_image, front_image,opticalflow_ego,opticalflow_front

    def hard_negative_pair(self,index):

        ego_image_path = self.ego_list[index]
        activity = ego_image_path.split('/')[-4]
        ego_img = Image.open(ego_image_path)
        if self.train_transform is not None:
            transformed_ego_img = self.train_transform(ego_img)
        ego_image = transformed_ego_img

        ego_id = ego_image_path.split('imxx')[-1].split('.')[0]
        ego_idx = int(ego_id)

        move = random.randint(6, self.frames_count[activity]-5)
        while move == ego_idx:
            move = random.randint(6, self.frames_count[activity]-5)
        front_idx = move
        fro_id=str(front_idx)
        fro_split = ego_image_path.split('/')
        fro_split[-1]='imxx'+ fro_id.zfill(4)+'.jpg'
        fro_split[-2] = "frames_front"
        front_image_path = '/'.join(fro_split)
        fro_img = Image.open(front_image_path)
        if self.train_transform is not None:
            fro_img = self.train_transform(fro_img)
        front_image = fro_img

        opticalflow_ego = self.get_optical_flow_ego(activity, ego_idx)
        opticalflow_front = self.get_optical_flow_front(activity, front_idx)
        return ego_image, front_image,opticalflow_ego,opticalflow_front

    def __getitem__(self, index):
        #print(index)
        if (index<len(self.ego_list)):
            ego_images, front_images,opticalflow_ego,opticalflow_front =self.positive_pair(index)
            return ego_images, front_images,opticalflow_ego,opticalflow_front , 1
        else:
            neg_index=index%len(self.ego_list)
            ego_images, front_images, opticalflow_ego, opticalflow_front =  self.hard_negative_pair(neg_index)
            #ego_images, front_images, opticalflow_ego, opticalflow_front =  self.soft_negative_pair(neg_index)

            return ego_images, front_images, opticalflow_ego, opticalflow_front, 0

    def __len__(self):
        return 2*len(self.ego_list)


def create_ego_list_val(rootdir, key_names, load_frame_count_t):
    ls = []
    listed_ego = sorted(glob.glob(rootdir +'**/synchronized/frames_ego/*.jpg', recursive=True))
    for i in listed_ego:
        listed_ego_frame_class = i
        videoname = i.split('imxx')[-1].split('.')[0]
        # videoname = i.split('_')[-1].split('.')[0]
        named = i.split('/')[-4]
        f = open('/home/adhamanaskar/Research/Siamese_Ameya/dataset/Marsh_val.json', )
        frames_count = json.load(f)
        f.close()
        if (int(videoname) > 5 and (int(videoname) <= int(frames_count[named]) - 5)):
            ls.append(listed_ego_frame_class)
    return ls


def create_ego_list_train(rootdir, key_names, load_frame_count_t):
    ls = []
    listed_ego = sorted(glob.glob(rootdir +'**/synchronized/frames_ego/*.jpg', recursive=True))
    for i in listed_ego:
        listed_ego_frame_class = i
        videoname = i.split('imxx')[-1].split('.')[0]
        # videoname = i.split('_')[-1].split('.')[0]
        named = i.split('/')[-4]
        f = open('/home/adhamanaskar/Research/Siamese_Ameya/dataset/Marsh_train.json', )
        frames_count = json.load(f)
        f.close()
        if (int(videoname) > 5 and (int(videoname) <= int(frames_count[named]) - 5)):
            ls.append(listed_ego_frame_class)
    return ls
def load_frame_count(mode):
    frames_count={}
    if (mode == 'train'):
        f = open('/home/adhamanaskar/Research/Siamese_Ameya/dataset/Marsh_train.json', )
        frames_count = json.load(f)
        f.close()
    else:
        f = open('/home/adhamanaskar/Research/Siamese_Ameya/dataset/Marsh_val.json', )
        frames_count = json.load(f)
        f.close()
    return frames_count
def activity_creator(key_names):
    activities = {}
    # iterating over list of tuples
    for i in key_names:
        key_names_split = i.split('_')
        name = key_names_split[0]
        activity = key_names_split[1]
        location = key_names_split[2]
        prep_key = name + "_" + location
        # dicts[prep_key].append(activity)
        activities.setdefault(prep_key, []).append(activity)
    return activities

def get_loader(train_path,batch_size,shuffle,stacked_opticalflow, num_workers,train_transform, val_transform):
    """ Returns torch.utils.data.DataLoader for custom pose dataset. """
    train_directories=sorted(glob.glob(os.path.join(train_path, "*", "")))
    train_key_names = []
    for line in train_directories:
        videoname = line.split('/')[-2]
        train_key_names.append(videoname)

    load_frame_count_t=load_frame_count("train")
    ego_train_list = create_ego_list_train(train_path, train_key_names, load_frame_count_t)
    train_listed_activities=activity_creator(train_key_names)
    training_set = MarshDataset(path=train_path, mode='train',stacked_opticalflow=stacked_opticalflow,directories=train_directories,keys=train_key_names ,load_frame_count=load_frame_count_t,ego_list=ego_train_list,listed_activities=train_listed_activities,train_transform=train_transform,val_transform=val_transform)
    train_loader = torch.utils.data.DataLoader(dataset=training_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_loader

def main():
    train_rootdir = "/home/ameya/Research/Ameya_Siamese_Dataset/"
    val_rootdir = "/home/ameya/Research/Ameya_Siamese_Dataset_val/"
    y=transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor(),
                                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225])])
    #z=transforms.Compose([ transforms.Resize([224, 224]),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    z = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor(),transforms.Normalize([0.5], [0.5])])
    train_loader,val_loader=get_loader(train_path=train_rootdir,val_path=val_rootdir,batch_size=16,shuffle=True,stacked_opticalflow=10,num_workers= 1, train_transform =y ,val_transform = z)
    print(len(val_loader))
    print(len(train_loader))
    count=0
    for i, (ego_images, front_images, opticalflow_ego, opticalflow_front, target) in enumerate(val_loader):
        print(ego_images.shape)
        print(front_images.shape)
        print(opticalflow_ego.shape)
        print(opticalflow_front.shape)
        break


    print(count)

    sys.exit(0)
if __name__ == '__main__':
    main()
