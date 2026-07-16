# --- Zero-shot eval (ImageNet-O): truyền model + preprocess từ cell 1 ---
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
            texts = [template.format(classname) for template in templates]  # format with class
            tokens = longclip.tokenize(texts).to(device)                   # tokenize -> device
            class_embs = model.encode_text(tokens)                         # text encode
            class_embs = class_embs / class_embs.norm(dim=-1, keepdim=True)
            class_emb = class_embs.mean(dim=0)
            class_emb = class_emb / class_emb.norm()
            zs_weights.append(class_emb)
        zs_weights = torch.stack(zs_weights, dim=1).to(device)
    return zs_weights


def run_zeroshot(model,
                 preprocess,
                 data_dir="/home/ubuntu/shared/hieu.tq/data/imagenet-o/imagenet-o",
                 num_workers=8,
                 batch_size=256,
                 device=None):
    """
    Truyền `model` và `preprocess` đã có ở cell 1.
    Logic còn lại giữ nguyên như script gốc.
    """
    assert preprocess is not None, "`preprocess` must be provided."

    # ✅ Quan trọng: reset default device về CPU để DataLoader/transform không đòi CUDA generator
    try:
        # Chỉ có từ PyTorch 2.0+, nếu không có thì bỏ qua
        import torch
        if hasattr(torch, "set_default_device"):
            torch.set_default_device("cpu")
    except Exception:
        pass


    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    # Lắp args giống argparse trong ví dụ
    args = SimpleNamespace(
        data_dir=data_dir,
        num_workers=num_workers,
        batch_size=batch_size,
    )

    softmax = torch.nn.Softmax(dim=1)
    loader, dataset = data_loader(preprocess, args)

    # Tạo zero-shot classifier
    zs_weights = zeroshot_classifier(model, imagenet_classes, imagenet_templates, device)

    total_num, true_num = 0, 0

    with torch.no_grad():
        for i, (images, targets, paths) in enumerate(tqdm(loader)):
            images = images.to(device)

            # Predict
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = 100.0 * (image_features @ zs_weights)
            probs = softmax(logits)
            pred = torch.argmax(probs, dim=1)

            total_len = pred.shape[0]
            for j in range(total_len):
                label = get_label(targets[j]).item()
                if pred[j].item() == label:
                    true_num += 1
                total_num += 1

            # # Optionally, save output to file.
            # save_to_file(logits, targets, paths)

    acc = true_num / total_num if total_num > 0 else 0.0
    print("Kết quả imagenet-o: ")
    print(acc)
    return acc