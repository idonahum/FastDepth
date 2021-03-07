def parse_args():
    loss_functions = ['l1','l2','DepthLoss']
    backbone_model = ['mobilenet','mobilenetv2']
    import argparse
    parser = argparse.ArgumentParser(description='FastDepth')
    parser.add_argument('-mode', help='train, test or evaluate models. options are: train, eval')
    parser.add_argument('-backbone',default='mobilenet',type=str,help=f'Which backbone to use, options are: {backbone_model} (default is mobilenet)')
    parser.add_argument('--bsize', default=8, type=int,help='Mini batch size.')
    parser.add_argument('-j','--workers', default=16, type=int,help='Number of workers for data loading.')
    parser.add_argument('-p','--print-freq', default=100, type=int,help='print frequency during training. (in batches).')
    parser.add_argument('-e','--epochs', default=500, type=int,help='Number of epochs, typically passes through the entire dataset.')
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
