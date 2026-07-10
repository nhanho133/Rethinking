import random, re
from typing import List, Any, Union

def split_into_detail_captions(long_text):
        sentences = [
            p.strip()
            for p in long_text.split('.')
            if p.strip()
            and len(p.strip()) >= 18
            and not (len(p.strip()) == 1 and p.strip().isalpha())
        ]
        return sentences

def random_composition(
    seq: List[str],
    k: int,
    seed: int = None,
    return_joined: bool = False,
    delim: str = ". "
) -> List[Union[List[str], str]]:
    """
    Bars and Stars.
    - return_joined=False: trả về List[List[str]]
    - return_joined=True : trả về List[str] (mỗi phần là 1 câu ghép)
    """
    while k>len(seq):
        seq.append(random.choice(seq))
    n = len(seq)
    # if not (1 <= k <= n):
        # import pdb
        # pdb.set_trace()
        # raise ValueError("1 <= k <= n   must be satisfied")
    if seed is not None:
        random.seed(seed)

    cuts = sorted(random.sample(range(1, n), k - 1))
    cuts = [0] + cuts + [n]
    parts = [seq[cuts[i]:cuts[i+1]] for i in range(k)]

    if not return_joined:
        return parts

    joined = []
    for group in parts:

        s = delim.join(group)
        s = re.sub(r"\s+", " ", s).strip()
        joined.append(s)
    return joined

def chunking(long_text, max_num_short_texts, seed):
    short_caps = split_into_detail_captions(long_text)
    return random_composition(short_caps, k = max_num_short_texts, seed=seed, return_joined=True)

def main():
    # Ví dụ 1: văn bản dài
    long_text = (
        "A cloudy day view of a row of parked John Deere brand tractors. The tractors are lined up facing to the right, with big wheels in the back. They are the traditional green with yellow markings. In front of the tractors on the right edge are lawn mowing attachments. They are all turned up on their side. The cutting blades are visible hanging down from the wheel they spin on. Behind the tractors are open trailers and more lawn cutters. They are green wagons with slanted openings. The second wagon to the right is red. A light gray cargo container is behind the row of tractors. Across the forefront is a weedy lawn. The background is an open field. Behind the lawn mowing attachments is a small rust-colored coral in the distance. The top half of the frame is an overcast, rainy sky. A tree with several trunks is behind the cargo container and fills the top left corner with leaves."
    )

    num_details = 3
    seed = 42

    chunks = chunking(long_text, num_details, seed)

    print(f"Original text:\n{long_text}\n")
    print(f"Split into {num_details} chunks (seed={seed}):")
    print(f"chunks:{chunks}")
    for i, c in enumerate(chunks, 1):
        print(f"{i}. {c}")

if __name__ == "__main__":
    main()