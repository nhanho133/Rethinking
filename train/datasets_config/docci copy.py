import json
import os
from PIL import Image
import clip
import torch.utils.data as data

data4v_root = '/home/ubuntu/shared/hieu.tq/dreamlip_long_captions/docci/'
json_path   = os.path.join(data4v_root, 'captioner_docci.json')
# data4v_root = '/home/ubuntu/shared/hieu.tq/dreamlip_long_captions/docci/'
# json_path   = os.path.join(data4v_root, 'captioner_docci.sample1k.train_keep_others.json')
image_root  = data4v_root

class DocciDataset(data.Dataset):
    def __init__(self, split: str, max_items: int = None):
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
        # _ , self.preprocess = clip.load("ViT-B/16")
        _ , self.preprocess = clip.load("ViT-L/14")

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
