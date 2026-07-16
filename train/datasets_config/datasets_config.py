# datasets_config.py

def _sharedgpt4v(**kwargs):
    from .sharegpt4v import share4v_train_dataset, share4v_val_dataset
    return share4v_train_dataset(**kwargs), share4v_val_dataset(**kwargs)

def _docci(**kwargs):
    from .docci import DocciDataset
    return (
        DocciDataset(split='train', max_items=None, **kwargs),
        DocciDataset(split='test', max_items=None, **kwargs),
    )

def _dci(**kwargs):
    from .dci import JsonDCIDataset
    return (
        JsonDCIDataset(
            json_path="/home/ubuntu/hieu.tq/Git/GOAL/datasets/DCI_train_del_org.json",
            max_items=None,
            **kwargs
        ),
        JsonDCIDataset(
            json_path="/home/ubuntu/hieu.tq/Git/GOAL/datasets/DCI_test.json",
            max_items=None,
            **kwargs
        )
    )

def _openv1(**kwargs):
    from .openEvenV1 import OpenEventV1Dataset
    root = r"/home/ubuntu/shared/OpenEvenv1/train/Train Set"  # lưu ý khoảng trắng trong tên thư mục

    train_set = OpenEventV1Dataset(root_dir=root, split="train", train_ratio=0.8, seed=42, clip_model="ViT-L/14")
    test_set  = OpenEventV1Dataset(root_dir=root, split="test",  train_ratio=0.8, seed=42, clip_model="ViT-L/14")
    # import pdb
    # pdb.set_trace()
    # train_set = OpenEventV1Dataset(root_dir=root, split="train", train_ratio=0.8, seed=42, clip_model="ViT-B/16")
    # test_set  = OpenEventV1Dataset(root_dir=root, split="test",  train_ratio=0.8, seed=42, clip_model="ViT-B/16")
    return train_set, test_set

def _dreamlip_cc3m(**kwargs):
    from .dreamlip_cc3m import DreamLIPCC3MDataset
    return (
        DreamLIPCC3MDataset(split='train', max_items=None, **kwargs),
        DreamLIPCC3MDataset(split='test', max_items=None, **kwargs),
    )

def _sharegpt4v_coco(**kwargs):
    from .sharegpt4v_coco import ShareGPT4VCOCODataset
    return (
        ShareGPT4VCOCODataset(split='train', max_items=None, **kwargs),
        ShareGPT4VCOCODataset(split='test', max_items=None, **kwargs),
    )

def _artpedia(**kwargs):
    from .art import ArtPediaDataset
    # train_set = ArtPediaDataset(split="train")
    # test_set   = ArtPediaDataset(split="test")
    train_set = ArtPediaDataset(split="train")
    test_set   = ArtPediaDataset(split="test")
    return train_set, test_set

dataset_mapping = {
    "sharedgpt4v": _sharedgpt4v,
    "docci": _docci,
    "dci": _dci,
    "dreamlip_cc3m": _dreamlip_cc3m,
    "sharegpt4v_coco": _sharegpt4v_coco,
    "openv1": _openv1,
    "artpedia": _artpedia,
}
