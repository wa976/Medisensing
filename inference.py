import os
import torch
import torchaudio
import argparse
import numpy as np
import time

from models import get_backbone_class
from util.icbhi_util import generate_fbank
from torchvision import transforms
import torch.nn as nn
from util.icbhi_diffusion_dataset import ICBHIDataset
from torch.utils.data import DataLoader
from copy import deepcopy
from util.augmentation import SpecAugment



def parse_args():
    parser = argparse.ArgumentParser('argument for inference')
    parser.add_argument('--model', type=str, default='ast')
    parser.add_argument('--n_cls', type=int, default=2)
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--desired_length', type=int, default=4)
    parser.add_argument('--n_mels', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--checkpoint_path', type=str, default="save/best.pth")
    parser.add_argument('--data_folder', type=str, default='./data2/test/real')
    parser.add_argument('--resz', type=float, default=1.0, help='resize the scale of mel-spectrogram')
    parser.add_argument('--specaug_policy', type=str, default='icbhi_ast_sup', 
                    help='policy (argument values) for SpecAugment')
    parser.add_argument('--specaug_mask', type=str, default='mean', 
                        help='specaug mask value', choices=['mean', 'zero'])
    args = parser.parse_args()
    args.h = int(args.desired_length * 100 - 2)
    args.w = 128
    return args


def set_model(args):
    start_time = time.time()
    kwargs = {}
    
    kwargs['input_fdim'] = int(args.h * args.resz)
    kwargs['input_tdim'] = int(args.w * args.resz)
    kwargs['label_dim'] = args.n_cls
    kwargs['imagenet_pretrain'] = False
    kwargs['audioset_pretrain'] = False
    kwargs['mix_beta'] = 1.0  # for Patch-MixCL
    

    model = get_backbone_class(args.model)(**kwargs)
    
    classifier =  deepcopy(model.mlp_head)
    
    
 
    

    ckpt = torch.load(args.checkpoint_path, map_location='cpu')
    state_dict = ckpt['model']

    # HOTFIX: always use dataparallel during SSL pretraining
    new_state_dict = {}
    for k, v in state_dict.items():
        if "module." in k:
            k = k.replace("module.", "")
        if "backbone." in k:
            k = k.replace("backbone.", "")
        if not 'mlp_head' in k: #del mlp_head
            new_state_dict[k] = v
    state_dict = new_state_dict
    model.load_state_dict(state_dict, strict=False)
    
        
       
    classifier.load_state_dict(ckpt['classifier'], strict=True)

    print('pretrained model loaded from: {}'.format(args.checkpoint_path))
        
        
    model.cuda()
    classifier.cuda()

    end_time = time.time()
    print(f'Model setup time: {end_time - start_time:.2f} seconds')
    
   
    return model, classifier

def set_loader(args):
    start_time = time.time()
    
    args.h = int(args.desired_length * 100 - 2)
    args.w = 128
    
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(size=(int(args.h * args.resz), int(args.w * args.resz)), antialias=None)  # explicitly set antialias=None
])
 

    dataset = ICBHIDataset(transform=transform, args=args,print_flag=False)
    

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=4, pin_memory=True)

    
    end_time = time.time()
    print(f'Data loader setup time: {end_time - start_time:.2f} seconds')
    return data_loader, args

def inference(data_loader, model, classifier, args,):

    model.eval()
    classifier.eval()
    
    total_inference_time = 0
    total_samples = 0


    with torch.no_grad():
        for idx, (images, labels) in enumerate(data_loader):
            print(images.shape)
            start_time = time.time()
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            


            with torch.cuda.amp.autocast():
                features = model(images, args=args, training=False)
                output = classifier(features)


            _, preds = torch.max(output, 1)
            
            end_time = time.time()
            batch_time = end_time - start_time
            total_inference_time += batch_time
            total_samples += images.size(0)
            
            print(f'Batch {idx+1} inference time: {batch_time:.2f} seconds')
    avg_inference_time = total_inference_time / total_samples
    print(f'Average inference time per sample: {avg_inference_time:.4f} seconds')

    return preds



def main():
    start_time = time.time()
    args = parse_args()
    

    args.transforms = SpecAugment(args)
    
    data_loader,  args = set_loader(args)


    model, classifier = set_model(args)
    
    inference_start_time = time.time()
    
    preds  = inference(data_loader, model, classifier, args)
    
    inference_end_time = time.time()
    model.eval()  # Set the model to evaluation mode
    
    
    print(f'Total inference time: {inference_end_time - inference_start_time:.2f} seconds')
    print(preds)
    
    end_time = time.time()
    print(f'Total execution time: {end_time - start_time:.2f} seconds')


if __name__ == '__main__':
    main()

    
