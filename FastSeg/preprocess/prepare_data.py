import os
import os.path
import pickle
from torch.utils.data import DataLoader

from preprocess import seg_transforms
from preprocess.datasets import NYUSegDataset


def create_data_loaders(args,with_input_orig=False):
        
    train_preprocessor_kwargs = {}
    dataset_kwargs = {'n_classes': 40}
    print('Creating dataset for segemenation... patience.')
    dataset_path = 'datasets'
    if not os.path.isdir(dataset_path):
        raise RuntimeError('Dataset directory not found.')
    depth_mode = 'refined'  
    train_dataset = NYUSegDataset(data_dir=dataset_path,split='train',with_input_orig=with_input_orig,**dataset_kwargs)
        
    train_preprocessor = seg_transforms.get_preprocessor(height=224,width=224,
            depth_mean=train_dataset.depth_mean,depth_std=train_dataset.depth_std,
            depth_mode=depth_mode,phase='train',**train_preprocessor_kwargs)
        
    train_dataset.preprocessor = train_preprocessor
        
    depth_stats = {'mean': train_dataset.depth_mean,'std': train_dataset.depth_std}
            

    valid_preprocessor = seg_transforms.get_preprocessor(
            height=480,width=640,
            depth_mean=depth_stats['mean'],depth_std=depth_stats['std'],
            depth_mode=depth_mode,phase='test')

    val_dataset = NYUSegDataset(data_dir=dataset_path,split='test',with_input_orig=with_input_orig,**dataset_kwargs)

    val_dataset.preprocessor = valid_preprocessor

    train_loader = DataLoader(train_dataset,
                              batch_size=args.bsize,num_workers=args.workers,
                              drop_last=True,shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=1,
                              num_workers=args.workers,shuffle=False)
    print('Finish loading datasets')
        
    return train_loader, val_loader


