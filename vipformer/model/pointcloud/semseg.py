from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange, repeat
from fairscale.nn import checkpoint_wrapper

from timm.models.layers import DropPath

from vipformer.model.pointcloud.utils import divide_patches, Sequential, Group2Emb, PointNetFeaturePropagation


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_q_input_channels: int,
        num_kv_input_channels: int,
        num_latent_channels: int,
        num_output_channels: Optional[int] = None,
        dropout: float = 0.0,
    ):
        """Multi-head attention as described in https://arxiv.org/abs/2107.14795 Appendix E.

        :param num_heads: Number of attention heads.
        :param num_q_input_channels: Number of query input channels.
        :param num_kv_input_channels: Number of key/value input channels.
        :param num_output_channels: Number of output channels attention result channels are projected to.
            Defaults to `num_q_input_channels`
        :param dropout: Dropout probability for attention matrix values. Defaults to `0.0`
        """
        super().__init__()

        if num_output_channels is None:
            num_output_channels = num_q_input_channels

        if num_latent_channels % num_heads != 0:
            raise ValueError("num_latent_channels must be divisible by num_heads")

        num_channels_per_head = num_latent_channels // num_heads

        self.dp_scale = num_channels_per_head ** -0.5
        self.num_heads = num_heads

        self.q_proj = nn.Linear(num_q_input_channels, num_latent_channels, bias=False)
        self.k_proj = nn.Linear(num_kv_input_channels, num_latent_channels, bias=False)
        self.v_proj = nn.Linear(num_kv_input_channels, num_latent_channels, bias=False)
        self.o_proj = nn.Linear(num_latent_channels, num_latent_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_q, x_kv, pad_mask=None, attn_mask=None):
        """
        :param x_q: Query input of shape (B, N, D) where B is the batch size, N the query sequence length
            and D the number of query input channels (= `num_q_input_channels`)
        :param x_kv: Key/value input of shape (B, L, C) where B is the batch size, L the key/value sequence
            length and C are the number of key/value input channels (= `num_kv_input_channels`)
        :param pad_mask: Boolean key padding mask. `True` values indicate padding tokens.
        :param attn_mask: Boolean attention mask. Not needed/supported yet.
        :return: attention result of shape (B, N, F) where B is the batch size, N the query sequence length
            and F the number of output channels (= `num_output_channels`)
        """
        if attn_mask is not None:
            raise NotImplementedError("attention masks not supported yet")

        q = self.q_proj(x_q)
        k = self.k_proj(x_kv)
        v = self.v_proj(x_kv)

        q, k, v = (rearrange(x, "b n (h c) -> (b h) n c", h=self.num_heads) for x in [q, k, v])
        attn = torch.einsum("b i c, b j c -> b i j", q, k) * self.dp_scale

        if pad_mask is not None:
            pad_mask = repeat(pad_mask, "b j -> (b h) () j", h=self.num_heads)
            attn_max_neg = -torch.finfo(attn.dtype).max
            attn.masked_fill_(pad_mask, attn_max_neg)

        attn = attn.softmax(dim=-1)
        # 这里 q,k 做完attention，还要做dropout，想想自己之前是否忽略了这一点
        attn = self.dropout(attn)

        o = torch.einsum("b i j, b j c -> b i c", attn, v)
        o = rearrange(o, "(b h) n c -> b n (h c)", h=self.num_heads)

        return self.o_proj(o)


class CrossAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_q_input_channels: int,
        num_kv_input_channels: int,
        num_latent_channels: int,
        dropout: float = 0.0,
    ):
        """Multi-head cross-attention (see `MultiHeadAttention` for details)."""
        super().__init__()
        self.q_norm = nn.LayerNorm(num_q_input_channels)
        self.kv_norm = nn.LayerNorm(num_kv_input_channels)
        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            num_q_input_channels=num_q_input_channels,
            num_kv_input_channels=num_kv_input_channels,
            num_latent_channels=num_latent_channels,
            num_output_channels=num_latent_channels,
            dropout=dropout,
        )

    def forward(self, x_q, x_kv, pad_mask=None, attn_mask=None):
        """Multi-head attention of query input `x_q` to key/value input (`x_kv`) after (separately) applying layer
        normalization to these inputs."""
        x_q = self.q_norm(x_q)
        x_kv = self.kv_norm(x_kv)
        return self.attention(x_q, x_kv, pad_mask=pad_mask, attn_mask=attn_mask)


class SelfAttention(nn.Module): 
    def __init__(
        self,
        num_heads: int,
        num_latent_channels: int,
        dropout: float = 0.0,
    ):
        """Multi-head self-attention (see `MultiHeadAttention` and for details)."""
        super().__init__()
        self.norm = nn.LayerNorm(num_latent_channels)
        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            num_q_input_channels=num_latent_channels,
            num_kv_input_channels=num_latent_channels,
            num_latent_channels=num_latent_channels,
            num_output_channels=num_latent_channels,
            dropout=dropout,
        )

    def forward(self, x, pad_mask=None, attn_mask=None):
        """Multi-head attention of input `x` to itself after applying layer normalization to the input."""
        x = self.norm(x)
        return self.attention(x, x, pad_mask=pad_mask, attn_mask=attn_mask)


class CrossAttentionLayer(Sequential):
    def __init__(
        self,
        num_heads: int,
        num_q_input_channels: int,
        num_kv_input_channels: int,
        num_latent_channels: int,
        widening_factor: int = 1,
        drop_path_rate: float = 0.0,
        atten_drop: float = 0.0,
        mlp_drop: float = 0.0,
        attention_residual: bool = True,
    ):
        cross_attn = CrossAttention(
            num_heads=num_heads,
            num_q_input_channels=num_q_input_channels,
            num_kv_input_channels=num_kv_input_channels,
            num_latent_channels=num_latent_channels,
            dropout=atten_drop,
        )
        super().__init__(
            Residual(cross_attn, mlp_drop, drop_path_rate) if attention_residual else cross_attn,
            Residual(MLP(num_q_input_channels, widening_factor), mlp_drop, drop_path_rate),
        )


class SelfAttentionLayer(Sequential):
    def __init__(
        self,
        num_heads: int,
        num_latent_channels: int,
        widening_factor: int = 1,
        drop_path_rate: float = 0.0,
        atten_drop: float = 0.0,
        mlp_drop: float = 0.0,
    ):
        self_attn = SelfAttention(
            num_heads=num_heads,
            num_latent_channels=num_latent_channels,
            dropout=atten_drop,
        )
        super().__init__(
            Residual(self_attn, mlp_drop, drop_path_rate),
            Residual(MLP(num_latent_channels, widening_factor), mlp_drop, drop_path_rate),
        )


class MLP(Sequential):
    def __init__(self, num_channels: int, widening_factor: int):
        super().__init__(
            nn.LayerNorm(num_channels),
            nn.Linear(num_channels, widening_factor * num_channels),
            nn.GELU(),
            nn.Linear(widening_factor * num_channels, num_channels),
        )


class Residual(nn.Module):
    def __init__(self, module: nn.Module, dropout: float, drop_path_rate: float):
        super().__init__()
        self.module = module
        self.dropout = nn.Dropout(p=dropout)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > .0 else nn.Identity()

    def forward(self, *args, **kwargs):
        # `x` is the output of Module SelfAttentionLayer/MLP
        x = self.module(*args, **kwargs)
        # apply dropout to `x`, then add with the original input
        x = self.drop_path(self.dropout(x) + args[0])
        return x


class InputAdapter(nn.Module):
    def __init__(self, num_input_channels: int):
        """Transforms and position-encodes task-specific input to generic encoder input.

        :param num_input_channels: Number of channels of the generic encoder input produced by this adapter.
        """
        super().__init__()
        self._num_input_channels = num_input_channels

    @property
    def num_input_channels(self):
        return self._num_input_channels

    def forward(self, x):
        raise NotImplementedError()


class Encoder(nn.Module):
    def __init__(
        self,
        num_latent_channels: int,
        num_cross_attention_layers: int = 1,
        num_cross_attention_heads: int = 4,
        cross_attention_widening_factor: int = 1,
        first_cross_attention_layer_shared: bool = False,
        num_self_attention_layers: int = 6,
        num_self_attention_heads: int = 4,
        self_attention_widening_factor: int = 1,
        dpr_list: list=[],
        atten_drop: float = 0.0,
        mlp_drop: float = 0.0,
        activation_checkpointing: bool = False,
    ):
        """Generic Perceiver IO encoder.

        :param input_adapter: Transforms and position-encodes task-specific input to generic encoder input
            of shape (B, M, C) where B is the batch size, M the input sequence length and C the number of
            key/value input channels. C is determined by the `num_input_channels` property of the
            `input_adapter`.
        :param num_latent_channels: Number of latent channels (D).
        :param num_cross_attention_heads: Number of cross-attention heads.
        :param num_cross_attention_qk_channels: Number of query and key channels for cross-attention
            (see `MultiHeadAttention.num_qk_channels` for details).
        :param num_cross_attention_v_channels: Number of value channels for cross-attention
            (see `MultiHeadAttention.num_v_channels` for details).
        :param num_cross_attention_layers: Number of cross-attention layers (alternating with self-attention blocks).
        :param first_cross_attention_layer_shared: Whether the first cross-attention layer should share its weights
            with subsequent cross-attention layers (if any).
        :param num_self_attention_heads: Number of self-attention heads.
        :param num_self_attention_qk_channels: Number of query and key channels for self-attention
            (see `MultiHeadAttention.num_qk_channels` for details).
        :param num_self_attention_v_channels: Number of value channels for self-attention
            (see `MultiHeadAttention.num_v_channels` for details).
        :param num_self_attention_layers: Number of self-attention layers per self-attention block.
        :param num_self_attention_blocks: Number of self-attention blocks sharing weights between corresponding
            self-attention layers.
        :param dropout: Dropout probability for self- and cross-attention layers and residuals.
        :param activation_checkpointing: If True, implements an activation checkpoint for each self-attention
            layer and cross-attention layer.
        """
        super().__init__()

        if num_cross_attention_layers <= 0:
            raise ValueError("num_cross_attention_layers must be > 0")
        self.num_cross_attention_layers = num_cross_attention_layers

        def cross_attn():
            layer = CrossAttentionLayer(
                num_heads=num_cross_attention_heads,
                num_q_input_channels=num_latent_channels,
                num_kv_input_channels=num_latent_channels,
                num_latent_channels=num_latent_channels,
                widening_factor=cross_attention_widening_factor,
                atten_drop=atten_drop,
                mlp_drop=mlp_drop
            )
            return checkpoint_wrapper(layer) if activation_checkpointing else layer

        self.cross_attn_n = cross_attn()

        if first_cross_attention_layer_shared or num_cross_attention_layers == 1:
            self.cross_attn_1 = self.cross_attn_n
        else:
            self.cross_attn_1 = cross_attn()

        self.sa_layers = nn.ModuleList()
        for i in range(num_self_attention_layers):
            self.sa_layers.append(SelfAttentionLayer(
                                num_heads=num_self_attention_heads,
                                num_latent_channels=num_latent_channels,
                                widening_factor=self_attention_widening_factor,
                                drop_path_rate=dpr_list[i],
                                atten_drop=atten_drop,
                                mlp_drop=mlp_drop))

    def forward(self, group_embs, pos_embs, pts, layer_idx, pad_mask=None):
        ''' The overall idea is conducting CrossAttention, followed by SelfAttention
                - e.g. 1 CrossAttentionLayers + 11 SelfAttentionLayers = 12 layers in total
            Args:
                group_embs: [batch, num_groups, num_latent_channels], here `num_groups` == `num_latents`
                pos_embs: [batch, num_groups, num_latent_channels]
                pts_embs: [batch, num_points, num_latent_channels]
                layer_idx: a list, specifies the layers whose features will be extracted
            Return:
                x_latent: [batch, num_groups, num_latent_channels]
        '''
        # ------ 每次经过AttentionLayer，都要加上位置编码
        x_latent = self.cross_attn_1(group_embs+pos_embs, pts, pad_mask)

        layer_feats = []
        idx = layer_idx

        for i, sa_layer in enumerate(self.sa_layers):
            # bypass `cross_attn_1` because it has been executed before the loop
            if i+1 < self.num_cross_attention_layers:
                x_latent = self.cross_attn_n(x_latent+pos_embs, pts, pad_mask)
            x_latent = sa_layer(x_latent+pos_embs)
            if i+1 in idx:
                layer_feats.append(x_latent)

        return layer_feats


class CrossFormer_semseg(nn.Module):
    def __init__(self, 
        input_adapter = None,
        point_channels=6,
        num_latents=128,
        num_latent_channels=384,
        group_size=32,
        num_cross_attention_layers=1,
        num_cross_attention_heads=6,
        num_self_attention_layers=12,
        num_self_attention_heads=6,
        mlp_widen_factor=4,
        max_dpr=0.1,
        atten_drop=.0,
        mlp_drop=.0,
        layer_idx=[],
        num_obj_classes=13):
        super().__init__()

        self.num_groups = num_latents
        self.group_size = group_size
        # for semantic segmentation, point_channels=6, that has x,y,z,r,g,b
        self.group2emb = Group2Emb(num_latent_channels, point_channels=point_channels)

        self.position_emb = nn.Sequential(
            nn.Linear(point_channels, 128),
            nn.GELU(),
            nn.Linear(128, num_latent_channels))

        self.input_adapter = input_adapter

        dpr_list = [x.item() for x in torch.linspace(0, max_dpr, num_self_attention_layers)]
        self.encoder = Encoder(
            num_latent_channels=num_latent_channels,
            num_cross_attention_layers=num_cross_attention_layers,
            num_cross_attention_heads=num_cross_attention_heads,
            cross_attention_widening_factor=mlp_widen_factor,
            num_self_attention_layers=num_self_attention_layers,
            num_self_attention_heads=num_self_attention_heads,
            self_attention_widening_factor=mlp_widen_factor,
            dpr_list=dpr_list,
            atten_drop=atten_drop,
            mlp_drop=mlp_drop)
        self.layer_idx = layer_idx

        self.norm = nn.LayerNorm(num_latent_channels)

        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(0.2))

        self.num_layer_idx = len(layer_idx)
        self.propagation = PointNetFeaturePropagation(in_channel=self.num_layer_idx*num_latent_channels+3, \
                                                    mlp=[mlp_widen_factor*num_latent_channels, 1024])

        self.conv1 = nn.Conv1d(2*self.num_layer_idx*num_latent_channels+1024, 512, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, num_obj_classes, 1)
        self.relu = nn.ReLU()

    def forward(self, pts):
        '''
            Args:
                pts: [batch, num_points, point_channels=6]
            Return:
                sem_pred: [batch, num_points, num_obj_classes]
        '''
        B, N, _ = pts.shape
        # encode each input point -> pts_embs: [batch, num_points, num_latent_channels]
        pts_embs = self.input_adapter(pts)

        # ------ `divide_patches` uses FPS to downsample a point cloud
        # neighbors: [batch, num_groups, group_size, point_channels]    centers: [batch, num_groups, point_channels]
        neighbors, centers = divide_patches(pts, self.num_groups, self.group_size)
        # group_embs: [batch, num_groups, num_latent_channels]
        group_embs = self.group2emb(neighbors)
        # pos_embs: [batch, num_groups, num_latent_channels]
        pos_embs = self.position_emb(centers)

        # feature_list: [3, batch, num_groups, dim_model]
        feature_list = self.encoder(group_embs, pos_embs, pts_embs, self.layer_idx)
        # feature_list: [3, batch, dim_model, num_groups]
        feature_list = [self.norm(x).transpose(-1, -2).contiguous() for x in feature_list]
        if self.num_layer_idx == 3:
            # x: [batch, 3*dim_model, num_groups]
            x = torch.cat((feature_list[0],feature_list[1],feature_list[2]), dim=1)
        if self.num_layer_idx == 4:
            # x: [batch, 4*dim_model, num_groups]
            x = torch.cat((feature_list[0],feature_list[1],feature_list[2],feature_list[3]), dim=1)
        # x_max: [batch, num_layer_idx*dim_model]
        x_max = torch.max(x,2)[0]
        # x_avg: [batch, num_layer_idx*dim_model]
        x_avg = torch.mean(x,2)

        # x_max_feature: [batch, num_layer_idx*dim_model, num_points]
        x_max_feature = x_max.view(B, -1).unsqueeze(-1).repeat(1, 1, N)
        # x_avg_feature: [batch, num_layer_idx*dim_model, num_points]
        x_avg_feature = x_avg.view(B, -1).unsqueeze(-1).repeat(1, 1, N)
        # x_global_feature: [batch, num_layer_idx*dim_model + num_layer_idx*dim_model, num_points]
        x_global_feature = torch.cat((x_max_feature, x_avg_feature), 1)
        # ------ when UpSampling using propagation, only pass the `point coordinates` into the function, without color information
        # pts[:, :, :3].transpose(-1, -2) -> [batch, 3, num_points]
        # centers[:, :, :3].transpose(-1, -2)  -> [batch, 3, num_groups]
        # x: [batch, 3*dim_model, num_groups]
        # f_level_0: [batch, 1024, num_points]
        f_level_0 = self.propagation(pts[:, :, :3].transpose(-1, -2), centers[:, :, :3].transpose(-1, -2), pts[:, :, :3].transpose(-1, -2), x)

        # x: [batch, 2*num_layer_idx*num_latent_channels+1024, num_points]
        x = torch.cat((f_level_0, x_global_feature), 1)
        # x: [batch, 512, num_points]
        x = self.relu(self.bn1(self.conv1(x)))
        # x: [batch, 512, num_points]
        x = self.dp1(x)
        # x: [batch, 512, num_points]
        x = self.relu(self.bn2(self.conv2(x)))
        # x: [batch, num_obj_classes, num_points]
        x = self.conv3(x)
        # x: [batch, num_points, num_obj_classes]
        x = x.permute(0, 2, 1)

        return x
