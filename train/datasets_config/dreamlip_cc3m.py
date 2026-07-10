"""
DreamLIP-recaptioned CC3M subset -- the actual long-caption dataset LLM2CLIP's official stage-2
training consumes (see llm2clip/data/download_dataset.sh: it downloads exactly this HF dataset,
qidouxiong619/dreamlip_long_captions). We use a locally-downloaded subset (~15k images,
downloaded via train/download_cc3m_subset.py since the source is web-URL-based CC3M with real
link rot). Same (image, caption, caption_short, img_path, split) interface as DocciDataset so
it's a drop-in replacement for train.py's dataset_mapping.
"""
import json
import os

import clip
import torch.utils.data as data
from PIL import Image

# H100 server: CC3M not preinstalled -> download via download_cc3m_subset.py into this work dir.
MANIFEST_PATH = "/cm/shared/chautvh_second/Nhan_folder/work/cc3m/manifest.json"
IMAGE_ROOT = "/cm/shared/chautvh_second/Nhan_folder/work/cc3m/images"


class DreamLIPCC3MDataset(data.Dataset):
    def __init__(self, split: str, max_items: int = None, model_name="ViT-B/16"):
        valid_splits = ("train", "test")
        if split not in valid_splits:
            raise ValueError(f"Unsupported split: {split}. Use one of {valid_splits}.")

        with open(MANIFEST_PATH, "r", encoding="utf8") as fp:
            full_data = json.load(fp)

        self.samples = [d for d in full_data if d.get("split") == split]
        if max_items is not None:
            self.samples = self.samples[:max_items]

        _, self.preprocess = clip.load(model_name)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        caption = item["caption"].replace("\n", " ")
        caption_short = item["caption_short"].replace("\n", " ")

        img_path = os.path.join(IMAGE_ROOT, item["image"])
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.preprocess(image)

        return image_tensor, caption, caption_short, img_path, item["split"]
