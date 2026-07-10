"""
ShareGPT4V (COCO-train2017-sourced subset). Captions come from ShareCaptioner -- the same
model DreamLIP used for its "SV" columns on CC3M/CC12M/YFCC15M -- but applied to COCO's
curated, more repetitive photo domain instead of noisy diverse web images. Intended as a
harder (less ceiling-effect-prone) in-domain retrieval test than the CC3M subset, at similar
caption style/quality. Same interface as DocciDataset/DreamLIPCC3MDataset.
"""
import json
import os

import clip
import torch.utils.data as data
from PIL import Image

# H100 server: subset manifest built by make_sharegpt4v_subset.py from the FULL 1.246M json
# already on disk; image field in manifest = original "sam/images/..|coco/train2017/..|llava/.."
# relative to the server's data/ dir.
MANIFEST_PATH = "/cm/shared/chautvh_second/Nhan_folder/work/sharegpt4v_subset_manifest.json"
IMAGE_ROOT = "/cm/archive/luongtk/sharegpt4v/data/"


class ShareGPT4VCOCODataset(data.Dataset):
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
