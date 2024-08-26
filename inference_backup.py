import os
import torch
import torchaudio
import argparse
import numpy as np

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
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--checkpoint_path', type=str, default='path/to/your/checkpoint.pth')
    parser.add_argument('--data_folder', type=str, default='path/to/your/audio/folder')
    parser.add_argument('--resz', type=float, default=1.0, help='resize the scale of mel-spectrogram')
    parser.add_argument('--specaug_policy', type=str, default='icbhi_ast_sup', 
                    help='policy (argument values) for SpecAugment')
    parser.add_argument('--specaug_mask', type=str, default='mean', 
                        help='specaug mask value', choices=['mean', 'zero'])
    args = parser.parse_args()
    args.h = int(args.desired_length * 100 - 2)
    args.w = 128
    return args

# def set_model(args):
#     kwargs = {}
#     kwargs['input_fdim'] = int(args.h * args.resz)
#     kwargs['input_tdim'] = int(args.w * args.resz)
#     kwargs['label_dim'] = args.n_cls
#     kwargs['imagenet_pretrain'] = False
#     kwargs['audioset_pretrain'] = False

#     model = get_backbone_class(args.model)(**kwargs)
    
    
#     # classifier = nn.Linear(model.final_feat_dim, args.n_cls)
#     classifier = deepcopy(model.mlp_head)
    
    

#     ckpt = torch.load(args.checkpoint_path, map_location='cpu')
#     state_dict = ckpt['model']
    
#     new_state_dict = {}
#     for k, v in state_dict.items():
#         if "module." in k:
#             k = k.replace("module.", "")
#         if "backbone." in k:
#             k = k.replace("backbone.", "")
#         if not 'mlp_head' in k:  # del mlp_head
#             new_state_dict[k] = v
#     state_dict = new_state_dict
#     model.load_state_dict(state_dict, strict=False)
    

#     if ckpt.get('classifier', None) is not None:
#         print("Correct: Classifier found in checkpoint")
#         classifier.load_state_dict(ckpt['classifier'], strict=True)

#     print('Pretrained model loaded from: {}'.format(args.checkpoint_path))


#     return model, classifier

def set_model(args):
    kwargs = {}
    
    kwargs['input_fdim'] = int(args.h * args.resz)
    kwargs['input_tdim'] = int(args.w * args.resz)
    kwargs['label_dim'] = args.n_cls
    kwargs['imagenet_pretrain'] = False
    kwargs['audioset_pretrain'] = False
    kwargs['mix_beta'] = 1.0  # for Patch-MixCL
    

    model = get_backbone_class(args.model)(**kwargs)
    
    classifier =  deepcopy(model.mlp_head)
    projector =  nn.Identity()
    
    
 
    

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
    projector.cuda()
    
    
   
    return model, classifier, projector

def set_loader(args):
    
    args.h = int(args.desired_length * 100 - 2)
    args.w = 128
    
    transform = [transforms.ToTensor(),
                    transforms.Resize(size=(int(args.h * args.resz), int(args.w * args.resz)))]
 
    transform = transforms.Compose(transform)


    dataset = ICBHIDataset(transform=transform, args=args,print_flag=False)
    

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=4, pin_memory=True)

    
    return data_loader, args


def inference(data_loader, model, classifier, args,):

    model.eval()
    classifier.eval()


    with torch.no_grad():
        for idx, (images, labels,file_names) in enumerate(data_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            bsz = labels.shape[0]
            


            with torch.cuda.amp.autocast():
                features = model(images, args=args, training=False)
                output = classifier(features)


            _, preds = torch.max(output, 1)
            
            print(preds)
            
            
            print(f"File: {file_names[0]}")
            print(f"Original class: {labels[0]}")
            print(f"Predicted class: {preds[0]}")
            # print(f"Confidence: {confidence:.4f}")
            print("---")



    return preds


# def inference(args, model, classifier):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)
#     classifier = classifier.to(device)
    

    
#     transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Resize(size=(int(args.h * args.resz), int(args.w * args.resz)), antialias=True)
# ])
    
#     # transform = transforms.Compose(transform)

#     dataset = ICBHIDataset(transform=transform, args=args,print_flag=False)
#     data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
#                                              num_workers=4, pin_memory=True)
    
#     model.eval()
#     classifier.eval()
#     args.transforms = SpecAugment(args)
    
#     with torch.no_grad():
#         for idx, (fbank, labels, file_names) in enumerate(data_loader):
            
#             images =  fbank.cuda(non_blocking=True)
#             labels = labels.cuda(non_blocking=True)
            
            
#             with torch.cuda.amp.autocast():
#                 features = model(args.transforms(images), args=args, training=False)
#                 # features = model(images)
#                 output = classifier(features)
        

#             _, preds = torch.max(output, 1)
                
                
#             print(preds)
            
            
#             print(f"File: {file_names[0]}")
#             print(f"Original class: {labels[0]}")
#             print(f"Predicted class: {preds[0]}")
#             # print(f"Confidence: {confidence:.4f}")
#             print("---")


def main():
    args = parse_args()
    

    args.transforms = SpecAugment(args)
    
    data_loader,  args = set_loader(args)


    model, classifier, projector = set_model(args)
    
    preds  = inference(data_loader, model, classifier, args)
    model.eval()  # Set the model to evaluation mode


if __name__ == '__main__':
    main()

    
