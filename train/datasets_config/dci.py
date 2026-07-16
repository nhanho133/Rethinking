import json
import os
from PIL import Image
import clip
import torch
from torch.utils.data import Dataset

import re


class JsonDCIDataset(Dataset):
    """
    Loads a list of {"filename": ..., "caption": ...} entries
    directly from a single JSON file.
    """
    def __init__(self, json_path: str, max_items: int = None):
        super().__init__()
        # 1) Load the flat list of samples
        with open(json_path, 'r', encoding='utf8') as f:
            samples = json.load(f)
        if max_items is not None:
            samples = samples[:max_items]
        self.samples = samples

        # 2) Pull CLIP’s preprocess once (on CPU)
        _, self.preprocess = clip.load("ViT-B/16", device='cpu')
        # _ , self.preprocess = clip.load("ViT-L/14")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample['filename']
        # 3) Load + preprocess the image
        image = Image.open(img_path).convert('RGB')
        image_tensor = self.preprocess(image)

        caption = sample['caption'].strip()
        # If your training loop expects (image, extra_caption, short_caption, path, split),
        # just return the same caption twice (or derive a shorter one if you like).
        short_caption = re.split(r'(?<=[\.!?])\s+', caption)[0].strip()
        return (
            image_tensor,  # [3, H, W]
            caption,       # long caption
            short_caption,       # short caption (here identical)
            img_path,      # for debugging/logging
            "json"         # dummy split name
        )
