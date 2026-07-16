import numpy as np

from typing import Optional, List, Tuple

def partition_indices_random(n: int, k: int, seed: Optional[int] = None) -> List[Tuple[int, int]]:
    """
    Chia ngẫu nhiên [0..n) thành k đoạn liên tiếp, mỗi đoạn >= 1 phần tử.
    Trả về k cặp (start, end) theo chuẩn [start, end).
    """
    assert 1 <= k <= n
    rng = np.random.default_rng(seed)
    cuts = sorted(rng.choice(np.arange(1, n), size=k-1, replace=False).tolist())
    bounds = [0] + cuts + [n]
    return [(bounds[i], bounds[i+1]) for i in range(k)]

def merge_by_partitions(texts: List[str], parts: List[Tuple[int, int]], sep: str = " ") -> List[str]:
    """Join texts within each (start, end) segment into a paragraph."""
    def _fix(s: str) -> str:
        s = s.strip()
        if not s:
            return s
        # already ends with a terminal mark (allow quotes after mark)
        if s.endswith(('.', '!', '?', '…')) or s.endswith(('."', '!"', '?"')):
            return s
        return s + '.'

    return [sep.join(_fix(t) for t in texts[s:e]) for (s, e) in parts]

def star_bar_long_text_split(long_text, num_partition, seed):
    parts_rand = partition_indices_random(len(long_text), num_partition, seed=seed)
    result = merge_by_partitions(long_text, parts_rand)
    return result

# long_text = ['The image captures the aftermath of a tragic event – the fatal stabbing of 15-year-old Jermaine Goupall', 
#              'While Jermaine himself is not visible, the presence of terrified boys seeking refuge inside the Costcutter shop and a distressed man standing nearby speaks volumes about the violence that recently unfolded', 
#              'These details, combined with the ominous red sky, paint a picture of fear and desperation.',
#              'The shop, usually a place of routine and normalcy, becomes a temporary sanctuary from danger.',
#              "Jermaine\'s death, a consequence of mistaken identity in a cycle of gang violence, highlights the tragic consequences of such conflict within the community", 
#              'The image serves as a stark reminder of the pervasive nature of knife crime, particularly amongst young individuals, and the urgent need for solutions to address the root causes of this societal issue'
#              ]
# for i in range(len(long_text)):
#     print(f"text {i}:{long_text[i]}") # 6 sentences

# num_partition = 3
# # partition = partition_indices_random(len(long_text),num_partition)

# parts_rand = partition_indices_random(len(long_text), num_partition, seed=42)
# paras_rand = merge_by_partitions(long_text, parts_rand)
# print("random partitions:", parts_rand)
# print("random paragraphs:", paras_rand)

# import pdb
# pdb.set_trace()
