# import json
# import cv2
# from PIL import Image
# import clip

# import torch
# import torch.utils.data as data
# import os
# import numpy as np
# import random

# data4v_root = '/home/tachau/docci_data/'
# json_name = 'captioner_docci.json'
# image_root = '/home/tachau/docci_data/'

# class share4v_val_dataset(data.Dataset):
#     def __init__(self):
#         self.data4v_root = data4v_root
#         self.json_name = json_name
#         self.image_root = image_root
#         self.total_len = 1000
#         with open(data4v_root + json_name, 'r',encoding='utf8')as fp:
#             self.json_data = json.load(fp)[:self.total_len]
#         _ , self.preprocess = clip.load("ViT-L/14")
#     def __len__(self):
#         return self.total_len

#     def __getitem__(self, index):
#         caption = self.json_data[index]['conversations'][1]['value']
#         caption = caption.replace("\n", " ")
#         image_name = self.image_root + self.json_data[index]['image']
#         image = Image.open(image_name)
#         image_tensor = self.preprocess(image)
#         return image_tensor, caption


# class share4v_train_dataset(data.Dataset):
#     def __init__(self):
#         self.data4v_root = data4v_root
#         self.json_name = json_name
#         self.image_root = image_root
#         self.total_len = 1000
#         with open(data4v_root + json_name, 'r',encoding='utf8')as fp:
#             self.json_data = json.load(fp)[self.total_len:]
#         _ , self.preprocess = clip.load("ViT-L/14")

#     def __len__(self):
#         return len(self.json_data)

#     def __getitem__(self, index):
#         caption = self.json_data[index]['conversations'][1]['value']
#         caption = caption.replace("\n", " ")
        

#         # caption_short = caption.split(". ")[0]
#         caption_short = self.json_data[index]['conversations'][0]['value']
#         caption_short = caption_short.replace("\n", " ")
        
#         image_name = self.image_root + self.json_data[index]['image']
#         image = Image.open(image_name)
#         image_tensor = self.preprocess(image)
#         return image_tensor, caption, caption_short, image_name

import json
import os
from PIL import Image
import clip
import torch.utils.data as data

# H100 server paths (image field in json = "docci_images/images/xxx.jpg" -> image_root+field OK)
# DOCCI_DATA_ROOT env override lets this run off-server (e.g. local machine) without touching
# the server default -- unset means the server path below is used unchanged.
data4v_root = os.environ.get('DOCCI_DATA_ROOT', '/cm/archive/luongtk/docci/')
json_path   = os.path.join(data4v_root, 'captioner_docci.json')
image_root  = data4v_root

class DocciDataset(data.Dataset):
    def __init__(self, split: str, max_items: int = None, model_name="ViT-B/16"):
        """
        split: one of 'train', 'qual_train', 'test', 'qual_test'
          - 'train'       => chỉ load split == 'train'
          - 'qual_train'  => chỉ load split == 'qual_train'
          - 'test'        => load cả split == 'test' và 'qual_test'
          - 'qual_test'   => chỉ load split == 'qual_test'
        max_items: nếu muốn giới hạn số mẫu
        """
        print(f"json path:{json_path}")
        valid_splits = ('train', 'qual_train', 'test', 'qual_test')
        if split not in valid_splits:
            raise ValueError(f"Unsupported split: {split}. Use one of {valid_splits}.")

        # Xác định danh sách splits sẽ lấy
        if split == 'train':
            target_splits = ['train']
        elif split == 'qual_train':
            target_splits = ['qual_train']
        elif split == 'test':
            target_splits = ['test', 'qual_test']
        else:  # split == 'qual_test'
            target_splits = ['qual_test']

        # 1) Load toàn bộ JSON
        with open(json_path, 'r', encoding='utf8') as fp:
            full_data = json.load(fp)

        # 2) Filter theo target_splits
        self.samples = [d for d in full_data if d.get('split') in target_splits]

        # 3) Nếu cần giới hạn số mẫu
        if max_items is not None:
            self.samples = self.samples[:max_items]

        # 4) Load CLIP preprocess once
        _ , self.preprocess = clip.load(model_name)
        # _ , self.preprocess = clip.load("ViT-L/14")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        # caption đầy đủ
        caption = item['conversations'][1]['value'].replace("\n", " ")
        # caption ngắn
        caption_short = caption.split('.')[0].strip() + '.'

        img_path = os.path.join(image_root, item['image'])
        image = Image.open(img_path).convert('RGB')
        image_tensor = self.preprocess(image)

        # Trả về luôn item['split'] ban đầu để biết nó thuộc loại nào
        return image_tensor, caption, caption_short, img_path, item['split']

# import json
# import cv2
# from PIL import Image
# import clip

# import torch
# import torch.utils.data as data
# import os
# import numpy as np
# import random

# # Định nghĩa đường dẫn
# data4v_root = '/home/ubuntu/shared/hieu.tq/dreamlip_long_captions/cc3m-dreamlip-processed/'
# json_name = 'captioner.json'
# image_root = '/home/ubuntu/shared/hieu.tq/dreamlip_long_captions/cc3m-dreamlip-processed/'

# class share4v_val_dataset(data.Dataset):
#     def __init__(self):
#         self.data4v_root = data4v_root
#         self.json_name = json_name
#         self.image_root = image_root
#         self.sample_size = 100
        
#         with open(data4v_root + json_name, 'r', encoding='utf8') as fp:
#             self.json_data = json.load(fp)[:100000]
        
#         # Chọn ngẫu nhiên 100 mẫu
#         self.json_data = random.sample(self.json_data, self.sample_size)
        
#         _, self.preprocess = clip.load("ViT-L/14")
#         # _, self.preprocess = clip.load("ViT-B/16")
    
#     def __len__(self):
#         return len(self.json_data)

#     def __getitem__(self, index):
#         caption = self.json_data[index]['conversations'][1]['value']
#         caption = caption.replace("\n", " ")
#         image_name = self.image_root + self.json_data[index]['image']
#         image = Image.open(image_name)
#         image_tensor = self.preprocess(image)
#         return image_tensor, caption


# class share4v_train_dataset(data.Dataset):
#     def __init__(self):
#         self.data4v_root = data4v_root
#         self.json_name = json_name
#         self.image_root = image_root
#         self.sample_size = 100 
        
#         with open(data4v_root + json_name, 'r', encoding='utf8') as fp:
#             self.json_data = json.load(fp)[100000:]
        
#         # Chọn ngẫu nhiên 100 mẫu
#         self.json_data = random.sample(self.json_data, self.sample_size)
        
#         _, self.preprocess = clip.load("ViT-L/14")
#         # _, self.preprocess = clip.load("ViT-B/16")
    
#     def __len__(self):
#         return len(self.json_data)

#     def __getitem__(self, index):
#         caption = self.json_data[index]['conversations'][1]['value']
#         caption = caption.replace("\n", " ")
        
#         caption_short = self.json_data[index]['conversations'][0]['value']
#         caption_short = caption_short.replace("\n", " ")
        
#         image_name = self.image_root + self.json_data[index]['image']
#         image = Image.open(image_name)
#         image_tensor = self.preprocess(image)
#         return image_tensor, caption, caption_short
