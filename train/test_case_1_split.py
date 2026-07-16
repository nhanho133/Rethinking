"""
Test case 1 (PDF 9/7/2026): verify how a long caption is split into `split_base` (the star-bar
base chunks) and `split_base_longer` (make_base_longer: each base chunk joined with one later
chunk). Reproduces train_epoch's exact splitting logic (num_seed=42, max_num_short_texts=4) on
the George Washington / UT Clock Tower DOCCI caption from the slides, and prints both, so the
output can be checked against the expected example in the PDF.

Run:  python test_case_1_split.py                 # in ket qua roi thoat
      python test_case_1_split.py --pdb            # in ket qua, dung lai pdb de tu soi bien
"""
import argparse
import json
import pdb
import random

from sampling import star_bar_long_text_split

NUM_SEED = 42


def split_into_detail_captions(text_long):
    # verbatim from train.py CLIP_Clean_Train.split_into_detail_captions
    return [
        p.strip()
        for p in text_long.split('.')
        if p.strip()
        and len(p.strip()) >= 18
        and not (len(p.strip()) == 1 and p.strip().isalpha())
    ]


def make_base_longer(chunks, rng):
    # verbatim from train.py CLIP_Clean_Train.make_base_longer
    n = len(chunks)
    if n <= 1:
        return [chunks[0]] if n == 1 else []
    out = []
    for i in range(n):
        partner = rng.randrange(n - 1)
        if partner >= i:
            partner += 1
        a, b = sorted([i, partner])
        out.append((chunks[a].strip() + " " + chunks[b].strip()).strip())
    return out


def build_base_chunks(long_text, max_num_short_texts):
    # verbatim from train_epoch's STAR BAR STRATEGY block
    split_cap = split_into_detail_captions(long_text)
    if len(split_cap) < max_num_short_texts:
        chunks = star_bar_long_text_split(split_cap, len(split_cap), NUM_SEED)
        while len(chunks) < max_num_short_texts:
            new_star = random.randint(1, len(split_cap))
            new_chunks = star_bar_long_text_split(split_cap, new_star, NUM_SEED)
            chunks.append(random.choice(new_chunks))
    else:
        chunks = star_bar_long_text_split(split_cap, max_num_short_texts, NUM_SEED)
    return split_cap, chunks


def find_gw_caption():
    data = json.load(open("/cm/archive/luongtk/docci/captioner_docci.json", encoding="utf8"))
    for d in data:
        cap = d["conversations"][1]["value"]
        if "George Washington" in cap and "Clock Tower" in cap:
            return cap.replace("\n", " ")
    raise RuntimeError("George Washington caption not found in DOCCI")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_num_short_texts", type=int, default=4)
    ap.add_argument("--caption", type=str, default=None, help="override caption (default: GW DOCCI)")
    ap.add_argument("--j", type=int, default=0, help="sample index in batch (affects make_base_longer rng)")
    ap.add_argument("--pdb", action="store_true",
                    help="Drop into pdb after computing chunks/longer, to inspect variables "
                         "yourself (matches PDF's 'Cau lenh test: import pdb; pdb.set_trace()').")
    args = ap.parse_args()

    caption = args.caption or find_gw_caption()
    print("=" * 100)
    print("FULL CAPTION:")
    print(caption)
    print("=" * 100)

    sentences = split_into_detail_captions(caption)
    print(f"\n[split_into_detail_captions] -> {len(sentences)} sentences:")
    for i, s in enumerate(sentences):
        print(f"  ({i}) {s}")

    # train_epoch seeds global random once per batch before the star-bar loop
    random.seed(NUM_SEED)
    split_cap, chunks = build_base_chunks(caption, args.max_num_short_texts)
    print(f"\n[SPLIT BASE]  star_bar_long_text_split(K={args.max_num_short_texts}, seed={NUM_SEED}) "
          f"-> {len(chunks)} base chunks:")
    for i, c in enumerate(chunks):
        print(f"  base[{i}]: {c}\n")

    rng = random.Random(NUM_SEED + 1000 + args.j)  # per-sample rng, exactly as train_epoch
    longer = make_base_longer(chunks, rng)
    print(f"[SPLIT BASE LONGER]  make_base_longer(rng=Random({NUM_SEED}+1000+{args.j})) "
          f"-> {len(longer)} longer chunks (base[a] + ' ' + base[b], a<b):")
    for i, c in enumerate(longer):
        print(f"  base_longer[{i}]: {c}\n")

    if args.pdb:
        print("=" * 100)
        print("Vao pdb. Cac bien co san de soi:")
        print("  sentences      -> list cau chi tiet (split_into_detail_captions)")
        print("  chunks         -> split_base   (list[str], len == max_num_short_texts)")
        print("  longer         -> split_base_longer (list[str], chunks[a]+' '+chunks[b])")
        print("  caption        -> caption day du goc")
        print("Vi du go trong pdb: chunks[0]  |  len(longer[3])  |  p chunks  |  c (tiep tuc) | q (thoat)")
        print("=" * 100)
        pdb.set_trace()


if __name__ == "__main__":
    main()
