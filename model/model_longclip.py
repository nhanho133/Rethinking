from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int, 
                 load_from_clip: bool
                 ):
        super().__init__()

        self.context_length = 248

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)

        if load_from_clip == False:
            self.positional_embedding = nn.Parameter(torch.empty(248, transformer_width))
            self.positional_embedding_res = nn.Parameter(torch.empty(248, transformer_width))

        else:
            self.positional_embedding = nn.Parameter(torch.empty(77, transformer_width))

        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()
        self.mask1 = torch.zeros([248, 1])
        self.mask1[:20, :] = 1
        self.mask2 = torch.zeros([248, 1])
        self.mask2[20:, :] = 1

    
    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text): 
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        
        x = x + (self.positional_embedding.to(x.device) * self.mask1.to(x.device)).type(self.dtype).to(x.device) + (self.positional_embedding_res.to(x.device) * self.mask2.to(x.device)).type(self.dtype).to(x.device) 
        
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def encode_text_full(self, text): 
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        
        x = x + (self.positional_embedding.to(x.device) * self.mask1.to(x.device)).type(self.dtype).to(x.device) + (self.positional_embedding_res.to(x.device) * self.mask2.to(x.device)).type(self.dtype).to(x.device) 
        
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        #x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x
    
    #rewrite PCA to avoid inf
    def PCA(self, input_tensor, PCA_dim):
        # 计算均值
        mean = torch.mean(input_tensor, dim=0)
        # 去均值
        X_centered = input_tensor - mean.unsqueeze(0)
        X_centered = X_centered.float()

        # 使用SVD而不是eig来计算主成分
        U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)
        principal_components = Vt.T[:, :PCA_dim]
        
        # 转换到新的维度
        X_transformed = torch.mm(X_centered, principal_components)
        # 恢复到原始空间
        X_reversed = torch.mm(X_transformed, principal_components.T)
        X_reversed += mean

        return X_reversed
    
    # def PCA(self, input_tensor, PCA_dim):
    #     mean = torch.mean(input_tensor, dim=0)
    #     X_centered = input_tensor - mean.unsqueeze(0)
    #     X_centered = X_centered.float()
    #     cov_matrix = torch.mm(X_centered.T, X_centered)
    #     eigenvalues, eigenvectors = torch.linalg.eig(cov_matrix)
    #     eigenvalues = eigenvalues.float()
    #     eigenvectors = eigenvectors.float()    
    #     sorted_indices = torch.argsort(eigenvalues, descending=True)
    #     eigenvectors = eigenvectors[:, sorted_indices]
    #     principal_components = eigenvectors[:, :PCA_dim]
    #     X_transformed = torch.mm(X_centered, principal_components)
    #     X_reversed = torch.mm(X_transformed, principal_components.T)
    #     X_reversed += mean
    #     return X_reversed


    # def forward(self, image, text):
    #     image_features = self.encode_image(image)
    #     text_features = self.encode_text(text)

    #     # normalized features
    #     image_features = image_features / image_features.norm(dim=1, keepdim=True)
    #     text_features = text_features / text_features.norm(dim=1, keepdim=True)

    #     # cosine similarity as logits
    #     logit_scale = self.logit_scale.exp()
    #     logits_per_image = logit_scale * image_features @ text_features.t()
    #     logits_per_text = logits_per_image.t()

    #     # shape = [global_batch_size, global_batch_size]
    #     return logits_per_image, logits_per_text

    #rewrite forward, fix the bug of no gradient in the original concat_all_gather. Notice that torch.distributed.nn.all_gather has backward function
    # ================= Rethinking / multi-view contrastive forward =================
    # Reconstructed to match train.py interface + slides (Adaptive Multi-Positive,
    # slide 4/6) + SPECS Eq.3 (specificity hinge). Returns:
    #   (loss, L_long, L_pos, L_longer, L_hinge, logit_scale, tau)
    # Loss weights are tunable via self.loss_w = (w_long, w_pos, w_longer, w_hinge).
    loss_w = (1.0, 1.0, 1.0, 1.0)

    @staticmethod
    def _sparsemax_1d(z):
        """1-D sparsemax (Martins & Astudillo, 2016) over a small group."""
        z = z.float()
        k = z.size(0)
        z_sorted, _ = torch.sort(z, descending=True)
        rng = torch.arange(1, k + 1, device=z.device, dtype=z.dtype)
        cssv = torch.cumsum(z_sorted, dim=0) - 1
        support = (z_sorted - cssv / rng) > 0
        k_z = int(support.sum().clamp(min=1).item())
        tau = (torch.cumsum(z_sorted, dim=0)[k_z - 1] - 1) / k_z
        return torch.clamp(z - tau, min=0)

    def _encode_grid(self, text_list, device):
        """text_list: list[B] of token tensors [K_i, ctx]. Pad each sample to
        K=max(batch) by repeating its last sub-caption, encode as a grid in ONE
        batched call. Returns Tf [B, K, D] (L2-normalized) or None.
        (train.py pads sub-captions to max_num_short_texts, so K is effectively
        fixed -> this reproduces the N x K grid the slide-4 loss assumes.)"""
        rows, Ks = [], []
        for t in text_list:
            if t is None or t.numel() == 0:
                rows.append(None); Ks.append(0)
            else:
                t = t.to(device).long()
                if t.dim() == 1:
                    t = t.unsqueeze(0)
                rows.append(t); Ks.append(t.size(0))
        K = max(Ks) if Ks else 0
        if K == 0:
            return None
        ctx = next(r for r in rows if r is not None).size(1)
        padded = []
        for r in rows:
            if r is None:
                r = torch.zeros(1, ctx, dtype=torch.long, device=device)
            if r.size(0) < K:                                  # pad by repeating last row
                r = torch.cat([r, r[-1:].expand(K - r.size(0), ctx)], 0)
            padded.append(r)
        grid = torch.stack(padded, 0)                          # [B, K, ctx]
        B = grid.size(0)
        f = self.encode_text(grid.reshape(B * K, ctx))
        f = f / f.norm(dim=-1, keepdim=True)
        return f.reshape(B, K, -1)                             # [B, K, D]

    def _adaptive_mp_loss(self, Tf, v, logit_scale, use_sparsemax=False):
        """Adaptive Multi-Positive loss (slide 4). For each sub-caption slot j a
        standard NxN contrastive is run in BOTH directions, weighted per image by
        W_{i,j} = softmax_j / sparsemax_j of DETACHED cos(v_i, t_{i,j}):
          t2v: -sum_i sum_j W_{i,j} log( e^{cos(v_i,t_ij)} / sum_n e^{cos(v_n,t_ij)} )
          v2t: -sum_i sum_j W_{i,j} log( e^{cos(v_i,t_ij)} / sum_n e^{cos(v_i,t_nj)} )
        Returns 0.5*(t2v + v2t)."""
        if Tf is None:
            return v.new_zeros(())
        B, K, _ = Tf.shape
        labels = torch.arange(B, device=v.device)
        cos_ii = torch.einsum('bd,bkd->bk', v.detach(), Tf.detach())   # [B,K] cos(v_i,t_{i,j})
        if use_sparsemax:
            W = torch.stack([self._sparsemax_1d(cos_ii[i]) for i in range(B)], 0)  # [B,K]
        else:
            W = torch.softmax(cos_ii, dim=1)                          # [B,K], sums to 1 over j
        L_t2v = v.new_zeros(())
        L_v2t = v.new_zeros(())
        for j in range(K):
            G = logit_scale * (Tf[:, j, :] @ v.t())                   # [B,B] G[i,n]=ls*cos(t_{i,j},v_n)
            L_t2v = L_t2v + (W[:, j] * F.cross_entropy(G,     labels, reduction='none')).sum()
            L_v2t = L_v2t + (W[:, j] * F.cross_entropy(G.t(), labels, reduction='none')).sum()
        return 0.5 * (L_t2v + L_v2t) / B

    def forward(self, image, text_long, tokenized_caps=None,
                learnable_mps=False, text_pos=None, text_neg=None,
                text_pos_longer=None, use_hinge=True, use_sparsemax=False):
        device = image.device
        B = image.size(0)
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        tau = 1.0 / logit_scale

        # image features
        v = self.encode_image(image)
        v = v / v.norm(dim=-1, keepdim=True)                   # [B, D]

        # --- L_long: image <-> full long caption (symmetric InfoNCE) ---
        tl = self.encode_text(text_long.long())
        tl = tl / tl.norm(dim=-1, keepdim=True)
        logits = logit_scale * (v @ tl.t())
        labels = torch.arange(B, device=device)
        L_long = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

        # --- encode sub-caption grids once (base & longer), padded to N x K ---
        Tf_pos = self._encode_grid(text_pos, device) if text_pos is not None else None
        Tf_lng = self._encode_grid(text_pos_longer, device) if text_pos_longer is not None else None

        # --- L_pos / L_longer: Adaptive Multi-Positive (slide 4, both directions) ---
        L_pos = self._adaptive_mp_loss(Tf_pos, v, logit_scale, use_sparsemax)
        L_longer = self._adaptive_mp_loss(Tf_lng, v, logit_scale, use_sparsemax)

        # --- L_hinge: SPECS Eq.3 specificity — longer (more detail) > base, per slot ---
        L_hinge = v.new_zeros(())
        if use_hinge and Tf_pos is not None and Tf_lng is not None and Tf_pos.shape == Tf_lng.shape:
            th_base = torch.einsum('bd,bkd->bk', v, Tf_pos)    # cos(image, base)   [B,K]
            th_long = torch.einsum('bd,bkd->bk', v, Tf_lng)    # cos(image, longer) [B,K]
            d = th_long - th_base                              # want d > 0
            eps = d.detach().mean().clamp(min=0.0)             # dynamic margin (detached, SPECS)
            L_hinge = torch.clamp(eps - d, min=0.0).mean()     # max(0, base - longer + eps)

        # --- total loss (weights tunable via self.loss_w) ---
        w_long, w_pos, w_longer, w_hinge = self.loss_w
        loss = w_long * L_long + w_pos * L_pos + w_longer * L_longer
        if use_hinge:
            loss = loss + w_hinge * L_hinge

        return (loss, L_long.detach(), L_pos.detach(), L_longer.detach(),
                L_hinge.detach(), logit_scale.detach(), tau.detach())
       


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict, load_from_clip: bool):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers, load_from_clip
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()
