# --- Zero-shot eval: truyền model + preprocess từ cell 1 ---
# import os
# # Ví dụ: chỉ cho phép sử dụng GPU số 1 và 2
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import sys
sys.path.append('../..')

import torch
from types import SimpleNamespace
from tqdm import tqdm

from model import longclip
from classes import imagenet_classes
from data_loader import data_loader, get_label
from templates import imagenet_templates


def zeroshot_classifier(model, classnames, templates, device):
    model.eval()
    with torch.no_grad():
        zs_weights = []
        for classname in tqdm(classnames):
            texts = [tpl.format(classname) for tpl in templates]
            tokens = longclip.tokenize(texts).to(device)
            txt_feats = model.encode_text(tokens)
            txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
            class_feat = txt_feats.mean(dim=0)
            class_feat = class_feat / class_feat.norm()
            zs_weights.append(class_feat)
        zs_weights = torch.stack(zs_weights, dim=1).to(device)
    return zs_weights

def evaluate_zeroshot(model,
                      preprocess,
                      data_dir="/home/ubuntu/shared/hieu.tq/imagenet-s/data/ImageNetS919/validation",
                      num_workers=8,
                      batch_size=256,
                      device=None):
    assert preprocess is not None, "`preprocess` must be provided."

    # ✅ Quan trọng: reset default device về CPU để DataLoader/transform không đòi CUDA generator
    try:
        # Chỉ có từ PyTorch 2.0+, nếu không có thì bỏ qua
        import torch
        if hasattr(torch, "set_default_device"):
            torch.set_default_device("cpu")
    except Exception:
        pass

    # Sau đó chọn device cho model/tensor như cũ
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)
    model = model.to(device).eval()

    from types import SimpleNamespace
    args = SimpleNamespace(
        data_dir=data_dir,
        num_workers=num_workers,
        batch_size=batch_size,
    )

    softmax = torch.nn.Softmax(dim=1)
    loader, _ = data_loader(preprocess, args)

    zs_weights = zeroshot_classifier(model, imagenet_classes, imagenet_templates, device)

    total_num = 0
    true_num = 0

    with torch.no_grad():
        for images, targets, paths in tqdm(loader):
            images = images.to(device, non_blocking=True)

            img_feats = model.encode_image(images)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

            logits = 100.0 * (img_feats @ zs_weights)
            probs = softmax(logits)
            pred = torch.argmax(probs, dim=1)

            for i in range(pred.shape[0]):
                label = get_label(targets[i]).item()
                if pred[i].item() == label:
                    true_num += 1
                total_num += 1

    acc = true_num / total_num if total_num > 0 else 0.0
    print("Kết quả imagenet-sub: ")
    print(f"Zero-shot top-1 accuracy: {acc:.4f}")
    return acc
