import json
import cv2
from PIL import Image
import clip

import torch
import torch.utils.data as data
import os
import numpy as np
import random

# Định nghĩa đường dẫn
data4v_root = '/home/ubuntu/shared/hieu.tq/dreamlip_long_captions/cc3m-dreamlip-processed/'
json_name = 'captioner.json'
image_root = '/home/ubuntu/shared/hieu.tq/dreamlip_long_captions/cc3m-dreamlip-processed/'

class share4v_val_dataset(data.Dataset):
    def __init__(self):
        self.data4v_root = data4v_root
        self.json_name = json_name
        self.image_root = image_root
        self.sample_size = 100
        
        with open(data4v_root + json_name, 'r', encoding='utf8') as fp:
            self.json_data = json.load(fp)[:1000]
        
        # Chọn ngẫu nhiên 100 mẫu
        self.json_data = random.sample(self.json_data, self.sample_size)
        
        _, self.preprocess = clip.load("ViT-L/14")
        # _, self.preprocess = clip.load("ViT-B/16")
    
    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, index):
        caption = self.json_data[index]['conversations'][1]['value']
        caption = caption.replace("\n", " ")
        image_name = self.image_root + self.json_data[index]['image']
        image = Image.open(image_name)
        image_tensor = self.preprocess(image)
        return image_tensor, caption


class share4v_train_dataset(data.Dataset):
    def __init__(self):
        self.data4v_root = data4v_root
        self.json_name = json_name
        self.image_root = image_root
        self.sample_size = 100
        
        with open(data4v_root + json_name, 'r', encoding='utf8') as fp:
            self.json_data = json.load(fp)[1000:]
        
        # Chọn ngẫu nhiên 100 mẫu
        self.json_data = random.sample(self.json_data, self.sample_size)
        
        _, self.preprocess = clip.load("ViT-L/14")
        # _, self.preprocess = clip.load("ViT-B/16")
    
    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, index):
        caption = self.json_data[index]['conversations'][1]['value']
        caption = caption.replace("\n", " ")
        
        caption_short = self.json_data[index]['conversations'][0]['value']
        caption_short = caption_short.replace("\n", " ")
        
        image_name = self.image_root + self.json_data[index]['image']
        image = Image.open(image_name)
        image_tensor = self.preprocess(image)
        return image_tensor, caption, caption_short
