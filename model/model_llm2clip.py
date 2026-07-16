"""
LLM2CLIP-text-teacher variant of the Rethinking model.

Keeps EVERYTHING about the Rethinking design (the Adaptive Multi-Positive + specificity-hinge
loss, the star-bar sub-caption structure, the retrieval eval) and the visual backbone
(OpenAI CLIP ViT-B/16 -- the same encoder LongCLIP fine-tuning starts from). The ONLY change
vs model_longclip.py is the text side: instead of a stretched CLIP text transformer over token
ids, text is a *frozen* Llama-3-8B-CC (LLM2CLIP's stage-2 caption teacher) whose embeddings are
precomputed offline (see train/precompute_llm2vec_embeddings.py) and looked up from a cache at
train time, then mapped into CLIP's 512-d joint space by a small trainable adapter.

The forward() signature and 7-tuple return exactly match model_longclip.py so train.py's
train_epoch/test_epoch_ver5 need only swap tokenize(...) -> cache.lookup(...). The
_encode_grid / _adaptive_mp_loss / hinge math is copied (not imported) from model_longclip.py,
adapted so "text" inputs are pre-embedded vectors instead of token-id tensors.

Two stages, selected by freeze_visual:
  - Stage 1 (freeze_visual=True):  frozen ViT-B/16 + frozen LLM cache; only the text adapter
    trains -> a fast linear-probe go/no-go check.
  - Stage 2 (freeze_visual=False): ViT-B/16 unfrozen (mirrors LLM2CLIP's own stage-2 recipe:
    freeze the LLM teacher, train the visual tower + adapter against it).
"""
import hashlib
import os

import clip
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


def _dist_info():
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def all_gather_grad(x):
    """all-gather that keeps gradients (torch.distributed.nn), returns concat over ranks.
    On single-GPU / no-dist it is a no-op -> the loss reduces exactly to the local formula."""
    _, world = _dist_info()
    if world == 1:
        return x
    return torch.cat(torch.distributed.nn.all_gather(x), dim=0)


class TextEmbeddingCache:
    """Loads the precomputed sha1(text) -> float16[llm_dim] cache. lookup() fails loudly on a
    miss -- a miss means the offline precompute didn't cover a string, which must never be
    silently papered over by live-encoding the 8B model on this GPU."""

    def __init__(self, cache_path, device="cuda"):
        import glob
        if os.path.isdir(cache_path):
            # Directory of part_NNNNN.pt files written by precompute's part-file mode. Merge
            # them into one in-RAM dict. (Peak RAM = total embeddings; fine when the box has
            # enough RAM. For the full-scale ~20M-string cache this is ~160GB -- check `free -h`.)
            part_files = sorted(glob.glob(os.path.join(cache_path, "part_*.pt")))
            if not part_files:
                raise FileNotFoundError(f"No part_*.pt files under {cache_path}")
            self.cache = {}
            self.dim = None
            for p in part_files:
                blob = torch.load(p, map_location="cpu")
                self.cache.update(blob["cache"])
                self.dim = blob["dim"]
                print(f"[TextEmbeddingCache] merged {p} -> running total {len(self.cache)}")
            print(f"[TextEmbeddingCache] loaded {len(self.cache)} embeddings (dim={self.dim}) "
                  f"from {len(part_files)} part files in {cache_path}")
        else:
            blob = torch.load(cache_path, map_location="cpu")
            self.cache = blob["cache"]
            self.dim = blob["dim"]
            print(f"[TextEmbeddingCache] loaded {len(self.cache)} embeddings (dim={self.dim}) from {cache_path}")
        self.device = device

    @staticmethod
    def _key(text):
        return hashlib.sha1(text.encode("utf8")).hexdigest()

    def lookup(self, texts):
        """texts: str or list[str] -> Tensor [N, dim] on self.device (float32)."""
        if isinstance(texts, str):
            texts = [texts]
        rows = []
        for t in texts:
            k = self._key(t)
            v = self.cache.get(k, None)
            if v is None:
                raise KeyError(
                    f"TextEmbeddingCache miss for string (len={len(t)}): {t[:80]!r}... "
                    f"Re-run precompute_llm2vec_embeddings.py to cover it."
                )
            rows.append(v)
        return torch.stack(rows, 0).to(self.device, dtype=torch.float32)


class LinearBlock(nn.Module):
    """Residual MLP block, copied verbatim from LLM2CLIP eva_clip/model.py::LinearBlock."""

    def __init__(self, dim, expansion_factor=4, dropout=0.):
        super().__init__()
        self.fn = nn.Sequential(
            nn.Linear(dim, int(expansion_factor * dim)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(expansion_factor * dim), dim),
        )
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        return x + self.fn(self.ln(x))


class TextAdapter(nn.Module):
    """Maps frozen LLM (4096-d) embeddings into CLIP's joint dim. Two kinds:
      - 'linear': LayerNorm -> Linear (light, ~2M params; best on small data like DOCCI).
      - 'mlp': faithful LLM2CLIP TextProj (L2-norm -> num_layers residual blocks -> LN ->
        Linear, ~270M params; higher capacity but overfits small data)."""

    def __init__(self, in_dim, out_dim, kind="linear", expansion_factor=2, num_layers=4,
                 dropout=0.0, proj_bias=True):
        super().__init__()
        self.kind = kind
        if kind == "mlp":
            self.text_adaptor = nn.Sequential(
                *[LinearBlock(in_dim, expansion_factor, dropout) for _ in range(num_layers)],
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, out_dim, bias=proj_bias),
            )
        elif kind == "linear":
            self.text_adaptor = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, out_dim, bias=proj_bias),
            )
        else:
            raise ValueError(f"unknown adapter kind: {kind}")

    def forward(self, x):
        if self.kind == "mlp":
            x = F.normalize(x, p=2, dim=-1)   # LLM2CLIP L2-normalizes the LLM embedding first
        return self.text_adaptor(x)


class LLM2CLIPTextTeacher(nn.Module):
    loss_w = (1.0, 1.0, 1.0, 1.0)  # (w_long, w_pos, w_longer, w_hinge), matches model_longclip

    def __init__(self, clip_base="ViT-B/16", llm_dim=4096, embed_dim=512,
                 freeze_visual=False, log_scale=4.6052, device="cpu", adapter_type="linear",
                 loss_w=None):
        super().__init__()
        base, _ = clip.load(clip_base, device=device)
        base = base.float()
        self.visual = base.visual                      # OpenAI CLIP ViT-B/16 visual tower
        self.text_adapter = TextAdapter(llm_dim, embed_dim, kind=adapter_type)
        self.logit_scale = nn.Parameter(torch.ones([]) * log_scale)
        if loss_w is not None:
            self.loss_w = tuple(loss_w)  # (w_long, w_pos, w_longer, w_hinge), overrides class default

        self.freeze_visual = freeze_visual
        if freeze_visual:
            for p in self.visual.parameters():
                p.requires_grad_(False)

    # --- encoders -----------------------------------------------------------------------
    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text_emb):
        """text_emb: [N, llm_dim] cached LLM embeddings (NOT token ids) -> [N, embed_dim]."""
        return self.text_adapter(text_emb.to(self.dtype))

    # --- loss helpers (copied from model_longclip.py, adapted for pre-embedded text) -----
    @staticmethod
    def _sparsemax_1d(z):
        z = z.float()
        k = z.size(0)
        z_sorted, _ = torch.sort(z, descending=True)
        rng = torch.arange(1, k + 1, device=z.device, dtype=z.dtype)
        cssv = torch.cumsum(z_sorted, dim=0) - 1
        support = (z_sorted - cssv / rng) > 0
        k_z = int(support.sum().clamp(min=1).item())
        tau = (torch.cumsum(z_sorted, dim=0)[k_z - 1] - 1) / k_z
        return torch.clamp(z - tau, min=0)

    def _encode_grid(self, emb_list, device):
        """emb_list: list[B] of embedding tensors [K_i, llm_dim] (pre-embedded sub-captions).
        Pad each sample to K=max(batch) by repeating its last row, project as ONE batched
        call. Returns Tf [B, K, embed_dim] (L2-normalized) or None."""
        rows, Ks = [], []
        for t in emb_list:
            if t is None or t.numel() == 0:
                rows.append(None); Ks.append(0)
            else:
                t = t.to(device)
                if t.dim() == 1:
                    t = t.unsqueeze(0)
                rows.append(t); Ks.append(t.size(0))
        K = max(Ks) if Ks else 0
        if K == 0:
            return None
        d_in = next(r for r in rows if r is not None).size(1)
        padded = []
        for r in rows:
            if r is None:
                r = torch.zeros(1, d_in, dtype=torch.float32, device=device)
            if r.size(0) < K:
                r = torch.cat([r, r[-1:].expand(K - r.size(0), d_in)], 0)
            padded.append(r)
        grid = torch.stack(padded, 0)                          # [B, K, d_in]
        B = grid.size(0)
        f = self.encode_text(grid.reshape(B * K, d_in))
        f = f / f.norm(dim=-1, keepdim=True)
        return f.reshape(B, K, -1)                             # [B, K, embed_dim]

    def _adaptive_mp_loss(self, Tf, v, logit_scale, use_sparsemax=False):
        if Tf is None:
            return v.new_zeros(())
        B, K, _ = Tf.shape
        rank, world = _dist_info()
        # Cross-GPU negatives: local queries vs GLOBAL keys (all ranks). On 1 GPU this is a
        # no-op and reduces exactly to the original Tf@v.t() / v@Tf.t() formulation.
        v_all = all_gather_grad(v)                       # [B*world, D]
        Tf_all = all_gather_grad(Tf)                     # [B*world, K, D]
        labels = torch.arange(B, device=v.device) + rank * B
        # weights from DETACHED local diagonal cos(v_i, t_ij) -- per local query, unchanged.
        cos_ii = torch.einsum('bd,bkd->bk', v.detach(), Tf.detach())
        if use_sparsemax:
            W = torch.stack([self._sparsemax_1d(cos_ii[i]) for i in range(B)], 0)
        else:
            W = torch.softmax(cos_ii, dim=1)
        L_t2v = v.new_zeros(())
        L_v2t = v.new_zeros(())
        for j in range(K):
            G_t = logit_scale * (Tf[:, j, :] @ v_all.t())        # [B, B*world] text query -> global img
            G_v = logit_scale * (v @ Tf_all[:, j, :].t())        # [B, B*world] img query  -> global text
            L_t2v = L_t2v + (W[:, j] * F.cross_entropy(G_t, labels, reduction='none')).sum()
            L_v2t = L_v2t + (W[:, j] * F.cross_entropy(G_v, labels, reduction='none')).sum()
        return 0.5 * (L_t2v + L_v2t) / B

    # --- forward: identical signature + 7-tuple as model_longclip.py --------------------
    def forward(self, image, text_long, tokenized_caps=None,
                learnable_mps=False, text_pos=None, text_neg=None,
                text_pos_longer=None, use_hinge=True, use_sparsemax=False):
        device = image.device
        B = image.size(0)
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        tau = 1.0 / logit_scale

        v = self.encode_image(image)
        v = v / v.norm(dim=-1, keepdim=True)                   # [B, D]

        # L_long: image <-> full long caption (symmetric InfoNCE) with cross-GPU negatives.
        # text_long is a [B, llm_dim] cached-embedding tensor, not token ids.
        rank, world = _dist_info()
        tl = self.encode_text(text_long)
        tl = tl / tl.norm(dim=-1, keepdim=True)
        v_all = all_gather_grad(v)                             # [B*world, D]
        tl_all = all_gather_grad(tl)                           # [B*world, D]
        labels = torch.arange(B, device=device) + rank * B
        logits_i = logit_scale * (v @ tl_all.t())             # [B, B*world] img query -> global txt
        logits_t = logit_scale * (tl @ v_all.t())             # [B, B*world] txt query -> global img
        L_long = 0.5 * (F.cross_entropy(logits_i, labels) + F.cross_entropy(logits_t, labels))

        Tf_pos = self._encode_grid(text_pos, device) if text_pos is not None else None
        Tf_lng = self._encode_grid(text_pos_longer, device) if text_pos_longer is not None else None

        L_pos = self._adaptive_mp_loss(Tf_pos, v, logit_scale, use_sparsemax)
        L_longer = self._adaptive_mp_loss(Tf_lng, v, logit_scale, use_sparsemax)

        L_hinge = v.new_zeros(())
        if use_hinge and Tf_pos is not None and Tf_lng is not None and Tf_pos.shape == Tf_lng.shape:
            th_base = torch.einsum('bd,bkd->bk', v, Tf_pos)
            th_long = torch.einsum('bd,bkd->bk', v, Tf_lng)
            d = th_long - th_base
            eps = d.detach().mean().clamp(min=0.0)
            L_hinge = torch.clamp(eps - d, min=0.0).mean()

        w_long, w_pos, w_longer, w_hinge = self.loss_w
        loss = w_long * L_long + w_pos * L_pos + w_longer * L_longer
        if use_hinge:
            loss = loss + w_hinge * L_hinge

        return (loss, L_long.detach(), L_pos.detach(), L_longer.detach(),
                L_hinge.detach(), logit_scale.detach(), tau.detach())


class LLM2CLIPReleasedTeacher(LLM2CLIPTextTeacher):
    """Faithful 'fully LLM2CLIP' platform: wraps Microsoft's RELEASED
    LLM2CLIP-Openai-L-14-336 model (pretrained ViT-L/14-336 visual + their pretrained
    TextProj adapter, 1280-d joint space). The visual tower is FROZEN (training its ~304M
    params + Adam states OOMs a 12GB 3060 regardless of grad-checkpointing); only their text
    adapter is fine-tuned. Reuses LLM2CLIPTextTeacher's forward / _encode_grid /
    _adaptive_mp_loss / hinge unchanged -- only encode_image/encode_text differ. Text inputs
    are the same precomputed Llama-3-8B-CC embeddings (their get_text_features expects exactly
    those), so the existing cache is reused directly.

    This is the platform for the loss ablation the thesis actually needs:
      vanilla LLM2CLIP loss (ClipLoss = L_long only)  vs  ours (L_long + DreamLIP MP + SPECS hinge)
    -- identical model, identical everything, only the loss differs.
    """

    def __init__(self, model_path, log_scale=4.6052, device="cpu", freeze_visual=False,
                 torch_dtype=torch.float32):
        import torch.nn as nn
        from transformers import AutoModel
        nn.Module.__init__(self)
        self.model = AutoModel.from_pretrained(
            model_path, torch_dtype=torch_dtype, trust_remote_code=True
        )
        # On the 3060 the ViT-L visual had to be frozen (OOM); on H100 set freeze_visual=False
        # to train the full visual tower (LLM2CLIP's real stage-2 recipe).
        self.freeze_visual = freeze_visual
        if freeze_visual:
            for p in self.model.vision_model.parameters():
                p.requires_grad_(False)
            for p in self.model.visual_projection.parameters():
                p.requires_grad_(False)
        self.logit_scale = nn.Parameter(torch.ones([]) * log_scale)

    @property
    def dtype(self):
        return torch.float32

    def encode_image(self, image):
        return self.model.get_image_features(pixel_values=image.to(self.dtype))

    def encode_text(self, text_emb):
        # their get_text_features = LLM2CLIP_Adapter (L2-norm -> 4x residual MLP -> LN -> Linear->1280)
        return self.model.get_text_features(text_emb.to(self.dtype))
