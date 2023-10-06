import os
import time
import argparse
import shutil
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

from torchvision import transforms
from models.two_stream_multichannel_fc2update import Two_Stream_Res
from loss import ContrastiveLoss
from dataset.marshardloader_train import get_loader

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('tensor2res/fc2_update_hard_pair_resnet')


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.manual_seed(7)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 150 epochs"""

    decay = 0.1 ** (sum(epoch >= np.array(args.lr_steps)))
    lr = args.lr * decay
    print("Current learning rate is %4.6f:" % lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, filename, resume_path):
    cur_path = os.path.join(resume_path, filename)
    torch.save(state, cur_path)
        
def train(train_loader, model, criterion, optimizer, epoch):
    # switch to train mode
    model.train()
    running_loss = 0.0
    total_step = len(train_loader)
    for i, (ego_frame, front_frame, ego_optical, front_optical, target) in enumerate(train_loader):
        ego_frame = ego_frame.float().cuda(async=True)
        front_frame = front_frame.float().cuda(async=True)
        ego_optical = ego_optical.float().cuda(async=True)
        front_optical = front_optical.float().cuda(async=True)
        target = target.cuda(async=True)
        ego_frame_var = torch.autograd.Variable(ego_frame)
        front_frame_var = torch.autograd.Variable(front_frame)
        ego_optical_var = torch.autograd.Variable(ego_optical)
        front_optical_var = torch.autograd.Variable(front_optical)
        target_var = torch.autograd.Variable(target)

        ego_embed, front_embed = model(ego_frame_var, front_frame_var, ego_optical_var, front_optical_var)

        loss = criterion(ego_embed, front_embed, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if i% 1000 ==0 and i is not  0:
            torch.save(model.state_dict(), os.path.join(args.model_path, 'fc2-update-two-stream-twores_hard_multi-{}-iter-{}.ckpt'.format(epoch + 1,str(i))))
        if i % 200 == 0 and i is not  0 :
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'.format(epoch, args.epochs, i,
                                                                                          total_step, loss.item(),
                                                                                          np.exp(loss.item())))
        if i % 100 == 0 and i is not  0 :  # every 1000 mini-batches...

            # ...log the running loss
            writer.add_scalar('TrainingWithFC2 hard pair loss',
                              running_loss / 100,
                              epoch * len(train_loader) + i)
            # print('Test: [{0}/{1}]\t' 'Loss {:.4f})\t'.format(i, len(train_loader), loss=running_loss / 100))

            running_loss = 0.0


def main():
    global args, best_prec1
    args = parser.parse_args()

    # create model
    print("Building model ... ")
    model = Two_Stream_Res()

    model.cuda()

    # criterion = torch.nn.MarginRankingLoss(margin=1.0)
    criterion = ContrastiveLoss(margin=0.9)
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    print("Saving everything to directory %s." % (args.model_path))

    cudnn.benchmark = True

    transform = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    z = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor(),transforms.Normalize([0.5], [0.5])])

    train_loader = get_loader(train_path=args.train_data_path, batch_size=32, shuffle=True, stacked_opticalflow=10, num_workers=1,
                                         train_transform=transform, val_transform=z)

    total_step = len(train_loader)
   
    print("total iter", total_step)
    for epoch in range(0, args.epochs):
        #adjust_learning_rate(optimizer, epoch)
      
        train(train_loader, model, criterion, optimizer, epoch)
        checkpoint_name = "%03d_%s" % (epoch + 1, "hardcheckpoint.pth.tar")
        save_checkpoint({'epoch': epoch + 1,'state_dict': model.state_dict(),'optimizer' : optimizer.state_dict()}, checkpoint_name, args.model_path)
        #torch.save(model.state_dict(), os.path.join(args.model_path, 'fc2-update-two-stream-twores_easy_multi-{}.ckpt'.format(epoch + 1)))

    writer.close()
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Two-Stream')
    parser.add_argument('--train_data_path', metavar='TrainDIR',
                        help='path to dataset')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 50)')
    parser.add_argument('--inchannels', default=20, type=int,
                        metavar='N', help='Temporal Resnet Input for the sequence')
    parser.add_argument('--crop_size', type=int, default=224, help='size for randomly cropping images')
    parser.add_argument('--lr', '--learning-rate', default=0.00001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--lr_steps', default=[100, 200], type=float, nargs="+",
                        metavar='LRSteps', help='epochs to decay learning rate by 10')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--print_freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 50)')
    parser.add_argument('--save_freq', default=25, type=int,
                        metavar='N', help='save frequency (default: 25)')
    parser.add_argument('--model_path', type=str, required=True, help='path for saving trained models')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    args = parser.parse_args()
    print(args)
    main()
