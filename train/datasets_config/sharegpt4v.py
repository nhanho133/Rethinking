import json
import cv2
from PIL import Image
import clip

import torch
import torch.utils.data as data
import os
import numpy as np
import random
import re

data4v_root = '/media/truongchau/51ee8689-bf17-441a-9814-96ba023bd60a/ShareGPT4V/data/'
# json_name = 'share-captioner_coco_lcs_sam_1246k_1107.json'  # full 1.2M (cần SAM)
json_name = 'sharegpt4v_coco_llava.json'  # 676k (coco + llava, không SAM)
# json_name = 'share-captioner_coco_lcs_sam_1246k_1107_filtered.json'


# /home/ubuntu/shared/ShareGPT4V/data/share-captioner_coco_lcs_sam_1246k_1107.json

# data4v_root = "/home/ubuntu/shared/ShareGPT4V/subsets/"
# json_name = "share-captioner_coco_lcs_sam_1246k_1107.eval1000_train1000_seed42.json"

# json_name = "share-captioner_coco_lcs_sam_1246k_1107.eval1000+train10000_seed42.json"
# json_name="share-captioner_coco_lcs_sam_1246k_1107.eval1000+train100000_seed42.cap_le_13.json"
# json_name="share-captioner_coco_lcs_sam_1246k_1107.eval1000_train1000_seed42.json"
# share-captioner_coco_lcs_sam_1246k_1107.eval1000+train100000_seed42.cap_le_13.json
#   - share-captioner_coco_lcs_sam_1246k_1107.eval1000+train1000_seed42.cap_le_13.json: 2,000 items (first 1000 = eval, all caption<=13)
#   - share-captioner_coco_lcs_sam_1246k_1107.eval1000+train10000_seed42.cap_le_13.json: 11,000 items (first 1000 = eval, all caption<=13)
#   - share-captioner_coco_lcs_sam_1246k_1107.eval1000+train50000_seed42.cap_le_13.json: 51,000 items (first 1000 = eval, all caption<=13)
#   - share-captioner_coco_lcs_sam_1246k_1107.eval1000+train100000_seed42.cap_le_13.json: 101,000 items (first 1000 = eval, all caption<=13)



#  = "share-captioner_coco_lcs_sam_1246k_1107.eval1000+train10000_seed42.json"
# share-captioner_coco_lcs_sam_1246k_1107.eval1000+train100000_seed42.json
# share-captioner_coco_lcs_sam_1246k_1107.eval1000+train10000_seed42.json
# share-captioner_coco_lcs_sam_1246k_1107.eval1000+train50000_seed42.json
# share-captioner_coco_lcs_sam_1246k_1107.eval1000_train1000_seed42.json

print(f"json name:{json_name}")
image_root = '/media/truongchau/51ee8689-bf17-441a-9814-96ba023bd60a/ShareGPT4V/data/'

class share4v_val_dataset(data.Dataset):
    def __init__(self, model_name="ViT-B/16"):
        self.data4v_root = data4v_root
        self.json_name = json_name
        print("train sharegpt4v json file:",json_name)
        self.image_root = image_root
        self.total_len = 1000
        with open(data4v_root + json_name, 'r',encoding='utf8')as fp:
            self.json_data = json.load(fp)[:self.total_len]
        # _ , self.preprocess = clip.load("ViT-L/14")
        _ , self.preprocess = clip.load(model_name)
    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        caption = self.json_data[index]['conversations'][1]['value']
        caption = caption.replace("\n", " ")
        caption_short = re.split(r'(?<=[\.!?])\s+', caption)[0].strip()
        image_name = self.image_root + self.json_data[index]['image']
        image = Image.open(image_name)
        image_tensor = self.preprocess(image)
        return image_tensor, caption, caption_short, image_name, "None"
        # return image_tensor, caption


class share4v_train_dataset(data.Dataset):
    def __init__(self, model_name="ViT-B/16"):
        self.data4v_root = data4v_root
        # print("data4v_root: ", data4v_root)
        self.json_name = json_name
        self.image_root = image_root
        self.total_len = 1000
        with open(data4v_root + json_name, 'r',encoding='utf8')as fp:
            self.json_data = json.load(fp)[self.total_len:]
        # _ , self.preprocess = clip.load("ViT-L/14")
        _ , self.preprocess = clip.load(model_name)

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, index):
        caption = self.json_data[index]['conversations'][1]['value']
        caption = caption.replace("\n", " ")
        

        # caption_short = caption.split(". ")[0]
        caption_short = re.split(r'(?<=[\.!?])\s+', caption)[0].strip()
        
        image_name = self.image_root + self.json_data[index]['image']
        image = Image.open(image_name)
        image_tensor = self.preprocess(image)
        return image_tensor, caption, caption_short, image_name, "None"

# import json
# import os
# from PIL import Image
# import clip
# import torch.utils.data as data

# data4v_root = '/home/ubuntu/shared/hieu.tq/dreamlip_long_captions/docci/'
# json_path   = os.path.join(data4v_root, 'captioner_docci.json')
# image_root  = data4v_root

# class Share4VDataset(data.Dataset):
#     def __init__(self, split: str, max_items: int = None):
#         """
#         split: 'train' hoặc 'test' (hoặc tuỳ giá trị bạn có trong JSON)
#         max_items: nếu bạn vẫn muốn giới hạn số mẫu (ví dụ val chỉ 1000), truyền vào đây
#         """
#         # 1) Load toàn bộ JSON
#         with open(json_path, 'r', encoding='utf8') as fp:
#             full_data = json.load(fp)

#         # 2) Filter theo split
#         self.samples = [d for d in full_data if d.get('split') == split]
#         # 3) Nếu cần giới hạn số mẫu (ví dụ val chỉ lấy 1000)
#         if max_items is not None:
#             self.samples = self.samples[:max_items]

#         # 4) Load CLIP preprocess once
#         _ , self.preprocess = clip.load("ViT-L/14")

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         item = self.samples[idx]

#         # caption đầy đủ
#         caption = item['conversations'][1]['value'].replace("\n", " ")
#         # caption ngắn
#         caption_short = item['conversations'][0]['value'].replace("\n", " ")

#         img_path = os.path.join(image_root, item['image'])
#         image = Image.open(img_path).convert('RGB')
#         image_tensor = self.preprocess(image)

#         return image_tensor, caption, caption_short, img_path, item['split']

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
