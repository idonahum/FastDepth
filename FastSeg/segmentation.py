
import os
import time,math
import datetime
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
cudnn.benchmark = True
from models import FastSeg,FastSegV2, weights_init
import utils, loss_func
from load_pretrained import load_pretrained_encoder, load_pretrained_fastdepth
from preprocess import create_data_loaders
from preprocess.confusion_matrix import ConfusionMatrixTensorflow
global args, writer

args = utils.parse_args()


def train_seg(model,optimizer,train_loader,val_loader,criteria=loss_func.DepthLoss(),epoch=0,batch=0):
    best_miou, best_epoch = None, None
    batch_count = batch
    if args.gpu and torch.cuda.is_available():
        model.cuda()
        criteria = criteria.cuda()

    print(f'{datetime.datetime.now().time().replace(microsecond=0)} Starting to train..')
    
    while epoch <= args.epochs-1:
        print(f'********{datetime.datetime.now().time().replace(microsecond=0)} Epoch#: {epoch+1} / {args.epochs}')
        model.train()
        interval_loss, total_loss= 0,0
        for i , sample in enumerate(train_loader):
            batch_count += 1
            input, target = sample['image'],sample['label'].cuda()
            target_scales = []
            if args.gpu and torch.cuda.is_available():
                target = target.cuda()
            target_scales.append(target.cuda())
            if args.gpu and torch.cuda.is_available():
                input = input.cuda()
            pred = model(input)
            pred_scales=[pred]
            
            loss = criteria(pred_scales,target_scales)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            interval_loss += loss.item()
            if np.isnan(total_loss):
                raise ValueError('Loss is None')
            if i % args.print_freq==0 and i>0:
                current_loss = interval_loss / args.print_freq
                print(f'********{datetime.datetime.now().time().replace(microsecond=0)} Batch #: {i:5d}/{len(train_loader):0d} Train Loss:{current_loss:.4f}.')
                writer.add_scalars(f'Loss',{'Train': current_loss}, batch_count)
                writer.flush()
                interval_loss = 0
        else:
            print(f'********{datetime.datetime.now().time().replace(microsecond=0)} Finish Epoch #{epoch+1} Total Loss:{total_loss/len(train_loader):.4f}. Saving checkpoint..')
            torch.save({'epoch': epoch,'batch':batch_count,'model_state_dict':model.state_dict(),'train_set':args.train_set,'val_set':args.val_set,'args':args},f'{args.weights_dir}/{args.backbone}/{args.criterion}/FastSeg_{epoch}.pth')
            print(f'********{datetime.datetime.now().time().replace(microsecond=0)} Detour, running validation..')
            miou,_ = evaluate_seg(model,val_loader,criterion=criteria)
            if not best_miou or miou < best_miou:
                best_miou = miou
                best_epoch = epoch+1
                print(f'********{datetime.datetime.now().time().replace(microsecond=0)} Best miou Update! best miou: {best_miou:.4f} , Epoch: {best_epoch}')
            model.train()
            epoch+=1
    torch.save({'epoch': epoch,'batch':batch_count,'model_state_dict': model.state_dict(),'train_set':args.train_set,'val_set':args.val_set,'args':args}, f'{args.weights_dir}/{args.backbone}/{args.criterion}/FastSeg_Final.pth')

                        
def evaluate_seg(model,test_loader,criterion):
    if args.gpu and torch.cuda.is_available():
        model.cuda()
        criterion = criterion.cuda() 
    n_classes = test_loader.dataset.n_classes_without_void
    n_samples,test_loss,timer = 0,0,0
    best_miou = None
    confusion_matrices = dict()
    cameras = test_loader.dataset.cameras
    for camera in cameras:
        confusion_matrices[camera] = dict()
        confusion_matrices[camera] = ConfusionMatrixTensorflow(n_classes)
        n_samples_total = len(test_loader.dataset)
        with test_loader.dataset.filter_camera(camera):
            for i, sample in enumerate(test_loader):
                n_samples += sample['image'].shape[0]
                print(f'\r{n_samples}/{n_samples_total}', end='')

                image = sample['image']
                label_orig = sample['label_orig']
                
               
                if args.gpu and torch.cuda.is_available():
                    image, label_orig = image.cuda(), label_orig.cuda()
                _, image_h, image_w = label_orig.shape

                with torch.no_grad():
                    time1 = time.time()
                    pred= model(image)
                   
                    time1 = time.time() - time1
                    
            
                    timer+= time1
                    pred = F.interpolate(pred, (image_h, image_w),
                                        mode='bilinear',
                                        align_corners=False)
                    pred = torch.argmax(pred, dim=1)

                    # ignore void pixels
                    mask = label_orig > 0
                    if args.gpu and torch.cuda.is_available():
                        mask = mask.cuda()
                    label = torch.masked_select(label_orig, mask)
                    pred = torch.masked_select(pred,mask )

                    # In the label 0 is void but in the prediction 0 is wall.
                    # In order for the label and prediction indices to match
                    # we need to subtract 1 of the label.
                    label -= 1

                    # copy the prediction to cpu as tensorflow's confusion
                    # matrix is faster on cpu
                    pred = pred.cpu()

                    label = label.cpu().numpy()
                    pred = pred.cpu().numpy()

                    confusion_matrices[camera].update_conf_matrix(label, pred)

                print(f'\r{i + 1}/{len(test_loader)}', end='')
    
                

        miou, _ = confusion_matrices[camera].compute_miou()
        print(f'\rCamera: {camera} mIoU: {100*miou:0.2f}')

    confusion_matrices['all'] = ConfusionMatrixTensorflow(n_classes)

    # sum confusion matrices of all cameras
    for camera in cameras:
        confusion_matrices['all'].overall_confusion_matrix += \
                confusion_matrices[camera].overall_confusion_matrix
    miou, _ = confusion_matrices['all'].compute_miou()
    print(f'All Cameras, mIoU: {100*miou:0.2f}')
    return miou, timer/len(test_loader)



                        
                        
args.criterion = 'ce'
if args.backbone == 'mobilenetv2':
    model_name = FastSegV2()
else:
    model_name = FastSeg()
    
if args.mode == 'train':
    writer = SummaryWriter(f'{args.tensorboard_dir}/{args.backbone}/{args.criterion}')
    if args.resume != None:
        resume_path = os.path.join(args.weights_dir,args.resume)
        assert os.path.isfile(resume_path), "No checkpoint found. abort.."
        print('Checkpoint found, loading...')
        checkpoint = torch.load(resume_path)
        args = checkpoint['args']
        epoch = checkpoint['epoch']+1
        batch = checkpoint['batch']
        model_state_dict = checkpoint['model_state_dict']
        model = model_name
        model.load_state_dict(model_state_dict)
        optimizer = optim.SGD(model.parameters(), lr = args.learning_rate , momentum=args.momentum,weight_decay=args.weight_decay)

        
        print('Finished loading')
        train_loader, val_loader = create_data_loaders(args)
        n_classes_without_void = train_loader.dataset.n_classes_without_void
        class_weighting = np.ones(n_classes_without_void)
        criteria = loss_func.CrossEntropyLoss2d(weight=class_weighting, device=args.gpu)
        train_seg(model,optimizer,train_loader,val_loader,criteria)
                  
    else:
        print('Creating new model..')
        train_loader, val_loader = create_data_loaders(args)
        model = model_name
        model.encoder = load_pretrained_encoder(model.encoder,args.weights_dir,args.backbone)
        model.decoder.apply(weights_init)
        optimizer = optim.SGD(model.parameters(), lr = args.learning_rate , momentum=args.momentum,weight_decay=args.weight_decay)
        print('Model created')
        n_classes_without_void = train_loader.dataset.n_classes_without_void
        class_weighting = np.ones(n_classes_without_void)
        criteria = loss_func.CrossEntropyLoss2d(weight=class_weighting, device=args.gpu)
        train_seg(model,optimizer,train_loader,val_loader,criteria)
        
        
elif args.mode == 'eval':
    _ ,val_loader = create_data_loaders(args)
    model = model_name
    model, args.criterion = load_pretrained_fastdepth(model,os.path.join(args.weights_dir,args.pretrained))
    n_classes_without_void = val_loader.dataset.n_classes_without_void
    class_weighting = np.ones(n_classes_without_void)
    criteria = loss_func.CrossEntropyLoss2d(weight=class_weighting, device=args.gpu)
    miou, timer = evaluate_seg(model,val_loader,criteria)
    print('MiOU Accuarcy:',miou)
    print('Time:',timer)

