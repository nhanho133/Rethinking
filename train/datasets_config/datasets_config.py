# datasets_config.py — trimmed to the 2 datasets used by the 3 training versions.
def _dreamlip_cc3m(**kwargs):
    from .dreamlip_cc3m import DreamLIPCC3MDataset
    return (DreamLIPCC3MDataset(split='train', max_items=None, **kwargs),
            DreamLIPCC3MDataset(split='test',  max_items=None, **kwargs))

def _sharegpt4v_coco(**kwargs):
    from .sharegpt4v_coco import ShareGPT4VCOCODataset
    return (ShareGPT4VCOCODataset(split='train', max_items=None, **kwargs),
            ShareGPT4VCOCODataset(split='test',  max_items=None, **kwargs))

dataset_mapping = {
    "sharegpt4v_coco": _sharegpt4v_coco,   # task 1 (baseline) + task 2 (ours)
    "dreamlip_cc3m":   _dreamlip_cc3m,     # task 3 (ours on cc3m)
}
