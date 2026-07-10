import numpy as np


def assign_learning_rate(optimizer, new_lr):
   for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr
       


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster


def cosine_lr_pergroup(optimizer, warmup_length, steps):
    """Like cosine_lr but preserves each param group's OWN base lr (captured at construction)
    and scales all groups by the same warmup+cosine factor. Needed when different param groups
    use different learning rates (e.g. a pretrained ViT at 1e-6 alongside a randomly-init
    adapter at 1e-3) -- the single-lr cosine_lr would clobber them to one value."""
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    def _lr_adjuster(step):
        if step < warmup_length:
            factor = (step + 1) / warmup_length
        else:
            e = step - warmup_length
            es = steps - warmup_length
            factor = 0.5 * (1 + np.cos(np.pi * e / es))
        for g, base in zip(optimizer.param_groups, base_lrs):
            g["lr"] = base * factor
        return factor

    return _lr_adjuster
