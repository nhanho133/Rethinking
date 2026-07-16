"""
Test case 1 (PDF: Test case Rethinking project, 9/7/2026)
Kiểm tra lại cách split text: split_base vs split_base_longer.

split_base        <- sampling.star_bar_long_text_split(sentences, num_partition, seed)
                      (dùng trong train.py::CLIP_Clean_Train, để xây chunks_all)
split_base_longer <- CLIP_Clean_Train.make_base_longer(chunks, rng)
                      (ghép mỗi chunk với 1 chunk khác theo thứ tự index tăng dần)

make_base_longer được copy nguyên văn từ train.py (train/train.py:225) vì hàm đó không
dùng self.* nào khác ngoài tham số truyền vào -> an toàn để tách đứng độc lập, tránh phải
import train.py (cần ftfy / model CLIP mới import được).
"""
import random
import sys
from caption_chunking import split_into_detail_captions
from sampling import star_bar_long_text_split


def make_base_longer(chunks, rng):
    """Copy y hệt CLIP_Clean_Train.make_base_longer (train/train.py:225)."""
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


LONG_TEXT = (
    'An outdoor daytime shot of a sculpted statue of George Washington in the foreground to '
    'the left and the University of Texas Clock Tower in the background to the right. '
    'A blue sky with two clouds floating above is seen behind the tower. '
    'The statue is on a square block platform made of the same metal. '
    'The stone foundation has wording etched to it professionally. '
    'The font reads," GEORGE / WASHINGTON / WHEN / MADE / COMMANDER / IN / CHIEF / OF / THE / '
    'AMERICAN / ARMY / OF / THE / REVOLUTION / ON / JULY / 3 / 1775."'
)

EXPECTED_SPLIT_BASE = (
    "An outdoor daytime shot of a sculpted statue of George Washington in the foreground to "
    "the left and the University of Texas Clock Tower in the background to the right. A blue "
    "sky with two clouds floating above is seen behind the tower."
)

EXPECTED_SPLIT_BASE_LONGER = (
    'An outdoor daytime shot of a sculpted statue of George Washington in the foreground to '
    'the left and the University of Texas Clock Tower in the background to the right. A blue '
    'sky with two clouds floating above is seen behind the tower. The statue is on a square '
    'block platform made of the same metal. The stone foundation has wording etched to it '
    'professionally. The font reads," GEORGE / WASHINGTON / WHEN / MADE / COMMANDER / IN / '
    'CHIEF / OF / THE / AMERICAN / ARMY / OF / THE / REVOLUTION / ON / JULY / 3 / 1775."'
)

NUM_PARTITION = 2  # 2 chunks: chunk0 -> split_base, chunk0+chunk1 -> split_base_longer (full text)
SEED = 1           # seed khớp đúng ranh giới cắt như ví dụ trong PDF (sentence idx 0..1 | 2..4)


def main():
    sentences = split_into_detail_captions(LONG_TEXT)
    print(f"[1] split_into_detail_captions -> {len(sentences)} câu:")
    for i, s in enumerate(sentences):
        print(f"    [{i}] {s}")

    chunks = star_bar_long_text_split(sentences, NUM_PARTITION, SEED)
    print(f"\n[2] split_base (star_bar_long_text_split, num_partition={NUM_PARTITION}, seed={SEED}):")
    for i, c in enumerate(chunks):
        print(f"    chunk[{i}]: {c}")

    rng = random.Random(SEED)
    chunks_longer = make_base_longer(chunks, rng)
    print("\n[3] split_base_longer (make_base_longer):")
    for i, c in enumerate(chunks_longer):
        print(f"    longer[{i}]: {c}")

    print("\n[4] So sánh với ví dụ trong PDF:")
    match_base = chunks[0].strip() == EXPECTED_SPLIT_BASE.strip()
    match_longer = chunks_longer[0].strip() == EXPECTED_SPLIT_BASE_LONGER.strip()
    print(f"    split_base   == PDF example ? {match_base}")
    print(f"    split_base_longer == PDF example ? {match_longer}")
    if not match_base:
        print(f"      got     : {chunks[0]!r}")
        print(f"      expected: {EXPECTED_SPLIT_BASE!r}")
    if not match_longer:
        print(f"      got     : {chunks_longer[0]!r}")
        print(f"      expected: {EXPECTED_SPLIT_BASE_LONGER!r}")

    if "--no-pdb" not in sys.argv:
        import pdb
        pdb.set_trace()  # bạn có thể inspect sentences / chunks / chunks_longer tại đây


if __name__ == "__main__":
    main()
