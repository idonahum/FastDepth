import collections
import argparse
import torchvision
from torchvision import transforms
import torch
import numpy as np
import matplotlib.image
import matplotlib.pyplot as plt
import cv2
plt.set_cmap("jet")
def parse_args():
    loss_functions = ['l1','l2','rmsle','l1gn','l2gn','rmslegn']
    backbone_model = ['mobilenet','mobilenetv2']
    parser = argparse.ArgumentParser(description='FastDepth')
    parser.add_argument('-mode', help='High level recipe: write tensors, train, test or evaluate models.')
    parser.add_argument('-backbone',default='mobilenet',type=str,help=f'Which backbone to use, options are: {backbone_model} (default is mobilenet)')
    parser.add_argument('--bsize', default=8, type=int,help='Mini batch size.')
    parser.add_argument('-j','--workers', default=16, type=int,help='Number of workers for data loading.')
    parser.add_argument('-p','--print-freq', default=100, type=int,help='print frequency during training. (in batches).')
    parser.add_argument('-e','--epochs', default=20, type=int,help='Number of epochs, typically passes through the entire dataset.')
    parser.add_argument('-s','--samples', default=None, type=int,help='Maximum number of data samples to write or load.')
    parser.add_argument('-lr','--learning_rate',default = 0.01, type=float,help='Learning rate value ')	
    parser.add_argument('--momentum', default=0.9, type=float,help='Momentum value')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,help='Weights decay (default: 1e-4)')
    parser.add_argument('--pretrained',default =None,type=str, help='pretrained FastDepth model file')
    parser.add_argument('--tensorboard_dir', default='Tensorboard',type=str, help='Directory to write images and plots to.')
    parser.add_argument('--weights_dir',default='Weights',type=str,help='Directory to save and load trained weights.')
    parser.add_argument('--resume', default=None,type=str,help='resume from checkpoint.')
    parser.add_argument('-c', '--criterion', metavar='LOSS', default='l1', choices=loss_functions,help= f'Loss function for training: {loss_functions}. (default: l1)')
    parser.add_argument('--gpu', default=False , type=bool, help="Use gpu or cpu? default is cpu")
    parser.add_argument('--train_set', default=None)
    parser.add_argument('--val_set', default=None)
    args = parser.parse_args()

    print('Arguments are', args)

    return args


def save_best_samples(imgs_dict):
    tensors_list = []
    sorted_imgs = collections.OrderedDict(sorted(imgs_dict.items(),reverse=True))
    for value in sorted_imgs:
        tensors_list.append(sorted_imgs[value])  
    rgb,pred,depth = [],[],[]
    for i in tensors_list:
        rgb.append(i[0])
        pred.append(i[1])
        depth.append(i[2])
    
    rgb = torch.stack(rgb).squeeze(1)
    pred = torch.stack(pred).squeeze(1)
    depth = torch.stack(depth).squeeze(1)
    rgb_grid = torchvision.utils.make_grid(rgb[:8],normalize=True)
    pred_grid = torchvision.utils.make_grid(pred[:8],normalize=True)
    depth_grid = torchvision.utils.make_grid(depth[:8],normalize=True)

    samples = torch.stack((rgb_grid,pred_grid,depth_grid))
    samples_grid = torchvision.utils.make_grid(samples,nrow=1,normalize= True)
    data_transforms = transforms.Compose([transforms.ToPILImage(),transforms.Grayscale(num_output_channels=1)])
    gray_samples = data_transforms(samples_grid.cpu())
    torchvision.utils.save_image(samples_grid,'samples.png')
    matplotlib.image.imsave('samplesjet.png', gray_samples)
