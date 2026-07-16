import random
import re
from pathlib import Path
from transformers import CLIPTokenizer

random.seed(42)

BASE_DATA_PATH = "/ShareGPT4V/data"
IMAGE_PATH = Path(BASE_DATA_PATH)

long_text_1 = """The image features the cover of a book titled "Mastering the Art of French Cooking" by Julia Child.
The cover is predominantly blue, adorned with a pattern of white fleur-de-lis.
The title of the book is prominently displayed in red text, while the authors' names are inscribed in black text.
The book is resting on a black surface, and the background is blurred, drawing focus to the book itself."""

long_text_2 = """The image presents the cover of a book titled "The Murders in the Rue Morgue" by Edgar Allan Poe.
The title is prominently displayed in white text at the top of the cover, while the author's name is written in smaller white text at the bottom.
The background of the cover is a dark blue color, providing a stark contrast to the white text.
In the bottom right corner of the cover, there's a small white text that reads "B&R Samizdat Express", possibly indicating the publisher or the edition of the book."""

raw_examples = [
    {"image": "coco/images/000000000001.jpg", "caption": long_text_1},
    {"image": "coco/images/000000000002.jpg", "caption": long_text_2},
]

def normalize_path(p: str) -> str:
    return p if p.startswith("/") else str(IMAGE_PATH / p)

def segment_caption(text: str):
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sents if s and s.strip()]

def shuffle_words(phrase: str) -> str:
    words = phrase.split()
    random.shuffle(words)
    return " ".join(words)

# 1) preprocess: normalize path + segmented_caption + dataset_part (90/10)
examples = []
for ex in raw_examples:
    ex2 = dict(ex)
    ex2["image"] = normalize_path(ex2["image"])
    ex2["segmented_caption"] = segment_caption(ex2["caption"])
    examples.append(ex2)

split_index = int(len(examples) * 0.90)
for i, ex in enumerate(examples):
    ex["dataset_part"] = "shuffle" if i < split_index else "no_shuffle"

# 2) make neg_details (need at least 2 samples)
if len(examples) < 2:
    raise ValueError("Need at least 2 samples to create negatives like the original code.")

for ex in examples:
    segs = ex["segmented_caption"]
    neg_details = []

    for i in range(len(segs) - 1):
        # pick a different sample
        while True:
            sample = random.choice(examples)
            if sample["image"] != ex["image"]:
                break

        j = min(i, len(sample["segmented_caption"]) - 1)
        candidate = sample["segmented_caption"][j].strip()

        if ex["dataset_part"] == "shuffle":
            candidate = shuffle_words(candidate)

        neg_details.append(candidate)

    ex["neg_details"] = neg_details

# 3) join captions with tokenizer cutoff
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

def build_curriculum(segs, neg_details, max_tokens=248):
    merged = [segs[0].strip()] if segs else []
    neg_caps = []

    for cap, neg in zip(segs[1:], neg_details):
        pos = f"{merged[-1]} {cap.strip()}"
        neg2 = f"{merged[-1]} {neg.strip()}"

        ids = tokenizer([pos, neg2])["input_ids"]
        if len(ids[0]) > max_tokens or len(ids[1]) > max_tokens:
            break

        merged.append(pos)
        neg_caps.append(neg2)

    return merged, neg_caps

final_outputs = []
for ex in examples:
    captions, neg_captions = build_curriculum(ex["segmented_caption"], ex["neg_details"], max_tokens=248)
    if len(neg_captions) == 0:
        continue

    final_outputs.append({
        "image": ex["image"],
        "caption": ex["caption"],
        "captions": captions,
        "neg_captions": neg_captions,
    })

# 4) print nicely
for idx, out in enumerate(final_outputs):
    print("="*80)
    print(f"Sample {idx}")
    print("image:", out["image"])
    print("\nOriginal caption:\n", out["caption"])
    print("\nCAPTIONS (positive curriculum):")
    for i, c in enumerate(out["captions"]):
        print(f"  [{i}] {c}")
    print("\nNEG_CAPTIONS:")
    for i, c in enumerate(out["neg_captions"]):
        print(f"  [{i}] {c}")
