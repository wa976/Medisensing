import time
import os
import torch
import torchaudio
import argparse
import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from watchdog.observers.polling import PollingObserver
import random


from models import get_backbone_class
from util.icbhi_util import generate_fbank
from torchvision import transforms
import torch.nn as nn
from util.icbhi_diffusion_dataset import ICBHIDataset
from torch.utils.data import DataLoader
from copy import deepcopy
from util.augmentation import SpecAugment
import torch.backends.cudnn as cudnn



def parse_args():
    parser = argparse.ArgumentParser('argument for inference')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--model', type=str, default='ast')
    parser.add_argument('--n_cls', type=int, default=2)
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--desired_length', type=int, default=4)
    parser.add_argument('--n_mels', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--checkpoint_path', type=str, default="../backup_icbhi/save/icbhi_ast_ce_jmir_ast_iphone_fold0/best.pth")
    parser.add_argument('--data_folder', type=str, default='./data4')
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

def process_audio_file(file_path, model, classifier, args):
    # 오디오 파일 로드 및 전처리
    waveform, sample_rate = torchaudio.load(file_path)

    

    # 멜 스펙트로그램 생성
    mel_spec = generate_fbank(args, waveform, args.sample_rate, n_mels=args.n_mels)
    
    
    # 크기 조정
    transform = transforms.Compose([transforms.ToTensor(),
        transforms.Resize(size=(int(args.h * args.resz), int(args.w * args.resz)), antialias=None)
    ])
    
    mel_spec = transform(mel_spec)  # (1, 1, H, W)
    
    mel_spec = mel_spec.unsqueeze(0)

    # 인퍼런스
    model.eval()
    classifier.eval()
    with torch.no_grad():
        mel_spec =mel_spec.cuda(non_blocking=True)
        with torch.cuda.amp.autocast():
            features = model(mel_spec, args=args, training=False)
            output = classifier(features)
        
        _, pred = torch.max(output, 1)
    
    return pred.item()

class NewDataHandler(FileSystemEventHandler):
    def __init__(self, args, model, classifier):
        self.args = args
        self.model = model
        self.classifier = classifier
        self.new_files = []
        self.last_file_time = time.time()

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.wav'):
            print(f"New audio file detected: {event.src_path}")
            self.new_files.append(event.src_path)
            self.last_file_time = time.time()

    def process_files(self):
        if len(self.new_files) >= 5:
            print("Processing 5 new files...")
            predictions = []
            for file_path in self.new_files[:5]:
                pred = process_audio_file(file_path, self.model, self.classifier, self.args)
                predictions.append(pred)
                print(f"Inference result for {file_path}: {pred}")
            
            # 최종 결과 계산
            class_0_count = predictions.count(0)
            class_1_count = predictions.count(1)
            
            if class_0_count >= 5:
                final_result = 0
            elif class_1_count >= 2:
                final_result = 1
            else:
                final_result = 2
            
            print(f"Final result for this batch: {final_result}")
            
            self.new_files = self.new_files[5:]
            self.last_file_time = time.time()
            return True
        return False

    def check_timeout(self):
        return time.time() - self.last_file_time > 600  # 10 minutes = 600 seconds
def main():
    args = parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True
    

    args.transforms = SpecAugment(args)
    


    model, classifier = set_model(args)
    
    event_handler = NewDataHandler(args, model, classifier)
    observer = PollingObserver()
    observer.schedule(event_handler, path=args.data_folder, recursive=False)
    observer.start()

    print(f"Watching for new audio files in {args.data_folder}")
    try:
        while True:
            if event_handler.process_files():
                print("Processed 5 files. Waiting for more...")
            elif event_handler.check_timeout():
                print("10 minutes passed without new files. Resetting...")
                event_handler.new_files = []
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
   


if __name__ == '__main__':
    main()

    
