import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from utils import get_accuracy_metrics


def init_weights(m):
    # From https://github.com/rstrudel/segmenter/blob/master/segm/model/utils.py
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


# Transformer blocks
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout, out_dim=None):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        if out_dim is None:
            out_dim = dim
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout):
        """
        Args:
            dim (torch.tensor): embedding dimension
            heads (int): number of cross-attention heads
            dropout (float): dropout
        Output:
            tensor of size (B x )
        """
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.attn = None

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)                 # (batch_sz, num_patches + 1, 3 * d_model)
            .reshape(B, N, 3, self.heads, C // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout, drop_path=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        # uncomment if using DropPath
        # if drop_path > 0.0:
        #     self.drop_path = DropPath(drop_path)
        # else:
        #     nn.Identity()
        self.drop_path = nn.Identity()

    def forward(self, x, mask=None, return_attention=False):
        y, attn = self.attn(self.norm1(x), mask)
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim, channels):
        super().__init__()

        self.image_size = image_size
        if image_size[0] % patch_size != 0 or image_size[1] % patch_size != 0:
            raise ValueError(
                "image dimensions must be divisible by the patch size"
            )
        self.grid_size = image_size[0]//patch_size, image_size[1]//patch_size
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, im):
        B, C, H, W = im.shape
        x = self.proj(im).flatten(2).transpose(1, 2)
        return x


# Encoder (ViT)
class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        n_layers,
        d_model,
        d_ff,
        n_heads,
        n_cls,
        dropout=0.1,
        # drop_path_rate=0.0,
        # distilled=False,
        channels=3,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            image_size,
            patch_size,
            d_model,
            channels,
        )
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.n_cls = n_cls

        # cls and pos tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.patch_embed.num_patches + 1, d_model)
        )

        # transformer blocks
        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout) for i in range(n_layers)]
        )

        # output head
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_cls)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        # if self.distilled:
        #     nn.init.trunc_normal_(self.dist_token, std=0.02)

        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token"}

    def forward(self, im, return_features=False):
        B, _, H, W = im.shape

        x = self.patch_embed(im)    # shape = B x N x d_model
        cls_tokens = self.cls_token.expand(B, -1, -1)   # shape = B x 1 x d_model # noqa: E501
        x = torch.cat((cls_tokens, x), dim=1)   # shape = B x (N + 1) x d_model

        # if self.distilled:
        #     dist_tokens = self.dist_token.expand(B, -1, -1)
        #     x = torch.cat((cls_tokens, dist_tokens, x), dim=1)
        # else:
        #     x = torch.cat((cls_tokens, x), dim=1)

        # num_extra_tokens = 1 + self.distilled
        # if x.shape[1] != pos_embed.shape[1]:
        #     pos_embed = resize_pos_embed(
        #         pos_embed,
        #         self.patch_embed.grid_size,
        #         (H // PS, W // PS),
        #         num_extra_tokens,
        #     )
        pos_embed = self.pos_embed      # shape = B x (N + 1) x d_model
        x = x + pos_embed       # point-wise addition
        x = self.dropout(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)    # shape = B x (N + 1) x d_model

        if return_features:
            return x

        # if self.distilled:
        #     x, x_dist = x[:, 0], x[:, 1]
        #     x = self.head(x)
        #     x_dist = self.head_dist(x_dist)
        #     x = (x + x_dist) / 2
        # else:
        #     x = x[:, 0]
        #     x = self.head(x)

        x = x[:, 0]
        x = self.head(x)
        return x

    def get_attention_map(self, im, layer_id):
        if layer_id >= self.n_layers or layer_id < 0:
            raise ValueError(
                f"Provided layer_id: {layer_id} is not valid.",
                f"0 <= {layer_id} < {self.n_layers}."
            )
        B, _, H, W = im.shape
        # PS = self.patch_size

        x = self.patch_embed(im)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        if self.distilled:
            dist_tokens = self.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_tokens, x), dim=1)
        else:
            x = torch.cat((cls_tokens, x), dim=1)

        pos_embed = self.pos_embed
        # num_extra_tokens = 1 + self.distilled
        # if x.shape[1] != pos_embed.shape[1]:
        #     pos_embed = resize_pos_embed(
        #         pos_embed,
        #         self.patch_embed.grid_size,
        #         (H // PS, W // PS),
        #         num_extra_tokens,
        #     )
        x = x + pos_embed

        for i, blk in enumerate(self.blocks):
            if i < layer_id:
                x = blk(x)
            else:
                return blk(x, return_attention=True)


# Decoder (Mask Transformer)
class MaskTransformer(nn.Module):
    def __init__(
        self,
        n_cls,
        patch_size,
        d_encoder,
        n_layers,
        n_heads,
        d_model,
        d_ff,
        # drop_path_rate,
        dropout,
    ):
        super().__init__()
        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.n_cls = n_cls
        self.d_model = d_model
        self.d_ff = d_ff
        self.scale = d_model ** -0.5

        # uncomment if want to use DropPath
        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout) for i in range(n_layers)]
        )

        self.cls_emb = nn.Parameter(torch.randn(1, n_cls, d_model))
        self.proj_dec = nn.Linear(d_encoder, d_model)

        # [d_model x d_model] square matrices initialized randomly
        self.proj_patch = nn.Parameter(
            self.scale * torch.randn(d_model, d_model)
        )
        self.proj_classes = nn.Parameter(
            self.scale * torch.randn(d_model, d_model)
        )

        self.decoder_norm = nn.LayerNorm(d_model)
        self.mask_norm = nn.LayerNorm(n_cls)

        self.apply(init_weights)
        nn.init.trunc_normal_(self.cls_emb, std=0.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_emb"}

    def forward(self, x, im_size):
        """
        Args:
            x (torch.tensor): output of encoder (shape = B x N x embed_dim)
                              where N = number of patches
                              and embed_dim = embedding dimension
            im_size (tuple): (H, W) - dim of input image to generate masks
        Output:
            torch.tensor: shape = B x C x H x W
        """
        H, W = im_size
        GS = H // self.patch_size

        x = self.proj_dec(x)
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), 1)
        for blk in self.blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        patches, cls_seg_feat = x[:, : -self.n_cls], x[:, -self.n_cls:]

        # Projection by multiplying with randomly initialized square matrices
        patches = patches @ self.proj_patch
        cls_seg_feat = cls_seg_feat @ self.proj_classes

        # Normalization
        patches = patches / patches.norm(dim=-1, keepdim=True)
        cls_seg_feat = cls_seg_feat / cls_seg_feat.norm(dim=-1, keepdim=True)

        masks = patches @ cls_seg_feat.transpose(1, 2)
        masks = self.mask_norm(masks)
        masks = rearrange(masks, "b (h w) n -> b n h w", h=int(GS))

        return masks

    def get_attention_map(self, x, layer_id):
        if layer_id >= self.n_layers or layer_id < 0:
            raise ValueError(
                f"Provided layer_id: {layer_id} is not valid.",
                f"0 <= {layer_id} < {self.n_layers}."
            )
        x = self.proj_dec(x)
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), 1)
        for i, blk in enumerate(self.blocks):
            if i < layer_id:
                x = blk(x)
            else:
                return blk(x, return_attention=True)


# Segmenter (Put both encoder and decoder together)
class Segmenter(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        n_cls,
    ):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = encoder.patch_size
        self.encoder = encoder
        self.decoder = decoder

    @torch.jit.ignore
    def no_weight_decay(self):
        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))

        nwd_params = append_prefix_no_weight_decay(
            "encoder.", self.encoder
        ).union(
            append_prefix_no_weight_decay("decoder.", self.decoder)
        )
        return nwd_params

    def forward(self, im, return_logits: bool = False):
        """
        return_logits: return logits if True
        """
        H, W = im.size(2), im.size(3)

        x = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        # num_extra_tokens = 1 + self.encoder.distilled
        # -- not using distillation for now, only remove cls tokens
        num_extra_tokens = 1
        x = x[:, num_extra_tokens:]

        masks = self.decoder(x, (H, W))

        logits = F.interpolate(masks, size=(H, W), mode="bilinear")
        probs = torch.softmax(logits, dim=1)
        if return_logits:
            return logits, probs

        return probs

    def get_attention_map_enc(self, im, layer_id):
        return self.encoder.get_attention_map(im, layer_id)

    def get_attention_map_dec(self, im, layer_id):
        x = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        # num_extra_tokens = 1 + self.encoder.distilled
        # -- not using distillation for now, only remove cls tokens
        num_extra_tokens = 1
        x = x[:, num_extra_tokens:]

        return self.decoder.get_attention_map(x, layer_id)

    def process_one_batch(
        self,
        data: tuple,
        optimizer: torch.optim.Optimizer,
        loss_criterion: torch.nn.Module,
        acc_criterion=None,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        mode: str = "eval",
    ):
        """
        Process one batch of training or validation data

        Args:
            data (torch.Tensor): tuple of (input, target) tensors
            optimizer (torch.optim.Optimizer): optimizer to use
            loss_criterion (torch.nn.Module): loss criterion to use
            acc_criterion (torch.nn.Module, optional): accuracy criterion to use
            device (torch.device): device to use
            mode (str): mode to use (train, eval, or test)

        Returns:
            tuple: (outputs, metrics)
        """
        # Move data to the device
        inputs, targets = data[0].to(device), data[1].to(device).squeeze()

        # Forward pass
        logits, outputs = self.forward(inputs, return_logits=True)

        # Calculate loss
        loss = loss_criterion(logits, targets.long())

        # Backpropagate
        if mode == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Calculate accuracy
        if acc_criterion:
            batch_acc = acc_criterion(outputs, targets)
        else:
            batch_acc = None

        metrics = get_accuracy_metrics(outputs, targets)
        metrics["loss"] = loss.item()
        metrics["accuracy"] = batch_acc.item()

        return outputs, metrics
