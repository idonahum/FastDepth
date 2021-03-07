import os
import os.path
import torch

def load_pretrained_encoder(encoder,weights_path,backbone):
    if backbone == 'mobilenetv2':
        state_dict = torch.load(f'{weights_path}/Mobilenetv2_pretrained.pth')
    else:
        checkpoint = torch.load(f'{weights_path}/model_best.pth.tar')
        state_dict = checkpoint['state_dict']
    from collections import OrderedDict
    target_state_dict = OrderedDict()
    for k, v in state_dict.items():
      target_state_dict[k] = v

    new_dict = encoder.state_dict()
    iter_dict = encoder.state_dict()
    if backbone == 'mobilenet':
        for k in iter_dict.keys():
            if 'batches' in k:
                new_dict.pop(k)
    for f, b  in zip(new_dict,target_state_dict):
      new_dict[f] = target_state_dict[b]

    encoder.load_state_dict(new_dict,strict=False)
    return encoder

def load_pretrained_fastdepth(model,weights_path):
        assert os.path.isfile(weights_path), "No pretrained model found. abort.."
        print('Model found, loading...')
        checkpoint = torch.load(weights_path)
        model_state_dict = checkpoint['model_state_dict']
        args = checkpoint['args']
        criterion = args.criterion
        model.load_state_dict(model_state_dict)
        print('Finished loading')
        return model,criterion