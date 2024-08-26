from curses import meta
import os
import pickle
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob

import torch
from torch.utils.data import Dataset
from copy import deepcopy
from PIL import Image

from .icbhi_util import get_annotations, generate_fbank, get_individual_cycles_torchaudio, cut_pad_sample_torchaudio,generate_spectrogram
from .augmentation import augment_raw_audio
import torchaudio
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt





class ICBHIDataset(Dataset):
    def __init__(self,  transform, args,  print_flag=True):
        test_data_folder = os.path.join(args.data_folder)
        
        
        self.data_folder = test_data_folder
        self.transform = transform
        self.args = args
        
        # parameters for spectrograms
        self.sample_rate = self.args.sample_rate
        self.n_mels = self.args.n_mels        
        
         # Initialize class counting
        self.class_nums = np.zeros(args.n_cls)


        self.data_glob = sorted(glob(self.data_folder+'/*.wav'))
        
        
        print('Total length of dataset is', len(self.data_glob))
                                    
        # ==========================================================================
        """ convert fbank """
        self.audio_images = []
        self.file_names = []

        for index in self.data_glob: #for the training set, 4142
            _, file_id = os.path.split(index)
            self.file_names.append(file_id)  # 파일 이름 저장



            
            audio, sr = torchaudio.load(index)

            
            file_id_tmp = file_id.split('.wav')[0]

            label = file_id_tmp.split('-')[2][0]   
            
            if label == "N":
                label = 0
            elif label == "B":
                label = 1
            elif label == "C":
                label = 1
            elif label == "W":
                label = 1
            else:
                print(index)
                continue
            
            
            
            audio, sr = torchaudio.load(index)
            
         
            image = generate_fbank(self.args, audio, self.sample_rate, n_mels=self.n_mels) 
  
          
            self.audio_images.append((image, int(label)))
        

        if print_flag:
            print('total number of audio data: {}'.format(len(self.data_glob)))
            print('*' * 25)


    def __getitem__(self, index):
        audio_images, label = self.audio_images[index][0], self.audio_images[index][1]

        audio_image = audio_images[0]
        
        

        if self.transform is not None:
            audio_image = self.transform(audio_image)
        

        return audio_image, torch.tensor(label)
        
    def __len__(self):
        return len(self.data_glob)