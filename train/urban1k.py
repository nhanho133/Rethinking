# import torch
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from tqdm import tqdm
# from dci import JsonDCIDataset
# import clip
# from train import CLIP_Clean_Train

# class Urban1kDataset(Dataset):
#     """
#     Dataset for Urban1k: pairs each image/ caption by filename (without extension).
#     """
#     def __init__(self, root_dir, max_items=None, device='cuda'):
#         self.image_dir   = os.path.join(root_dir, 'image')
#         self.caption_dir = os.path.join(root_dir, 'caption')
#         # all caption files (strip “.txt”)
#         self.ids = sorted([
#             fname[:-4] for fname in os.listdir(self.caption_dir)
#             if fname.endswith('.txt')
#         ])
#         if max_items is not None:
#             self.ids = self.ids[:max_items]

#         # load CLIP preprocess
#         self.device     = torch.device(device)
#         _, self.preprocess = clip.load('ViT-B/16', device=self.device)

#     def __len__(self):
#         return len(self.ids)

#     def __getitem__(self, idx):
#         idx = self.ids[idx]
#         # load caption
#         with open(os.path.join(self.caption_dir, f'{idx}.txt'), 'r', encoding='utf8') as f:
#             caption = f.read().strip().replace('\n', ' ')
#         # load & preprocess image
#         img_path = os.path.join(self.image_dir, f'{idx}.jpg')
#         image    = Image.open(img_path).convert('RGB')
#         image_t  = self.preprocess(image)
#         return image_t, caption, "None", img_path, "None"

import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import clip

class Urban1kDataset(Dataset):
    """
    Dataset for Urban1k: pairs each image/caption by filename (without extension).
    Supports an internal 80/20 split via `split='train'` or `split='val'`.
    """
    def __init__(self,
                 root_dir,
                 split: str = 'train',
                 split_ratio: float = 0.8,
                 seed: int = 42,
                 max_items: int = None,
                 device: str = 'cuda',
                 model_name: str = 'ViT-B/16',
                 use_full_split: bool = False):
        """model_name: CLIP preprocess to use (e.g. 'ViT-L/14@336px' for llm2clip_released
        checkpoints -- was hardcoded to 'ViT-B/16' before, which mismatches that model's
        expected input resolution/normalization).
        use_full_split=True: paper's "Urban1K" benchmark is the FULL 1000 images as a single
        eval set (no 80/20 train/val carve-out) -- set this for zero-shot benchmark eval;
        the default 80/20 behavior is kept for backward compat with any existing callers."""
        assert split in ('train', 'val'), "split phải là 'train' hoặc 'val'"
        self.image_dir   = os.path.join(root_dir, 'image')
        self.caption_dir = os.path.join(root_dir, 'caption')

        # Lấy danh sách id (filename không extension) và shuffle
        ids = sorted([fname[:-4]
                      for fname in os.listdir(self.caption_dir)
                      if fname.endswith('.txt')])
        if max_items is not None:
            ids = ids[:max_items]

        if use_full_split:
            self.ids = ids
        else:
            random.seed(seed)
            random.shuffle(ids)
            # chia 80/20
            split_idx = int(len(ids) * split_ratio)
            if split == 'train':
                self.ids = ids[:split_idx]
            else:
                self.ids = ids[split_idx:]

        # load CLIP preprocess (model không dùng, chỉ lấy transform)
        self.device     = torch.device(device)
        _, self.preprocess = clip.load(model_name, device=self.device)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        uid = self.ids[idx]
        # load caption
        with open(os.path.join(self.caption_dir, f'{uid}.txt'),
                  'r', encoding='utf8') as f:
            caption = f.read().strip().replace('\n', ' ')

        # Tách câu đầu tiên
        sentences = [s.strip() for s in caption.split('.') if s.strip()]
        detail_caption = sentences[0] + '.' if sentences else caption

        # load & preprocess image
        img_path = os.path.join(self.image_dir, f'{uid}.jpg')
        image    = Image.open(img_path).convert('RGB')
        image_t  = self.preprocess(image)

        return image_t, caption, detail_caption, img_path, "None"

