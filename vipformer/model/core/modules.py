from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange, repeat
from fairscale.nn import checkpoint_wrapper
from torch import Tensor

from vipformer.model.core.utils import Sequential
from timm.models.layers import DropPath


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_q_input_channels: int,
        num_kv_input_channels: int,
        num_qk_channels: Optional[int] = None,
        num_v_channels: Optional[int] = None,
        num_output_channels: Optional[int] = None,
        dropout: float = 0.0,
    ):
        """Multi-head attention as described in https://arxiv.org/abs/2107.14795 Appendix E.

        :param num_heads: Number of attention heads.
        :param num_q_input_channels: Number of query input channels.
        :param num_kv_input_channels: Number of key/value input channels.
        :param num_qk_channels: Number of channels query and key input channels are projected to,
            for computing the attention matrix. Defaults to number `num_q_input_channels`
        :param num_v_channels: Number of channels value input channels are projected to.
            Defaults to `num_qk_channels`.
        :param num_output_channels: Number of output channels attention result channels are projected to.
            Defaults to `num_q_input_channels`
        :param dropout: Dropout probability for attention matrix values. Defaults to `0.0`
        """
        super().__init__()

        if num_qk_channels is None:
            num_qk_channels = num_q_input_channels

        if num_v_channels is None:
            num_v_channels = num_qk_channels

        if num_output_channels is None:
            num_output_channels = num_q_input_channels

        if num_qk_channels % num_heads != 0:
            raise ValueError("num_qk_channels must be divisible by num_heads")

        if num_v_channels % num_heads != 0:
            raise ValueError("num_v_channels must be divisible by num_heads")

        num_qk_channels_per_head = num_qk_channels // num_heads

        self.dp_scale = num_qk_channels_per_head ** -0.5
        self.num_heads = num_heads

        self.q_proj = nn.Linear(num_q_input_channels, num_qk_channels, bias=False)
        self.k_proj = nn.Linear(num_kv_input_channels, num_qk_channels, bias=False)
        self.v_proj = nn.Linear(num_kv_input_channels, num_v_channels, bias=False)
        self.o_proj = nn.Linear(num_v_channels, num_output_channels)
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
        # NOTE: after softmax normalization, there is a Dropout to atten values
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
        num_qk_channels: Optional[int] = None,
        num_v_channels: Optional[int] = None,
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
            num_qk_channels=num_qk_channels,
            num_v_channels=num_v_channels,
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
        num_channels: int,
        num_qk_channels: Optional[int] = None,
        num_v_channels: Optional[int] = None,
        dropout: float = 0.0,
    ):
        """Multi-head self-attention (see `MultiHeadAttention` and for details)."""
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)
        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            num_q_input_channels=num_channels,
            num_kv_input_channels=num_channels,
            num_qk_channels=num_qk_channels,
            num_v_channels=num_v_channels,
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
        num_qk_channels: Optional[int] = None,
        num_v_channels: Optional[int] = None,
        widening_factor: int = 1,
        atten_drop: float = 0.1,
        mlp_drop: float = 0.5,
        attention_residual: bool = True,
    ):
        cross_attn = CrossAttention(
            num_heads=num_heads,
            num_q_input_channels=num_q_input_channels,
            num_kv_input_channels=num_kv_input_channels,
            num_qk_channels=num_qk_channels,
            num_v_channels=num_v_channels,
            dropout=atten_drop,
        )
        super().__init__(
            # In CrossAttentionLayer, set `drop_path_rate=0`
            Residual(cross_attn, atten_drop, drop_path_rate=0) if attention_residual else cross_attn,
            Residual(MLP(num_q_input_channels, widening_factor), mlp_drop, drop_path_rate=0),
        )
    # The forward() method of Sequential accepts any input and forwards it to the first module it contains. 
    # It then “chains” outputs to inputs sequentially for each subsequent module, finally returning the output of the last module.


class SelfAttentionLayer(Sequential):
    def __init__(
        self,
        num_heads: int,
        num_channels: int,
        num_qk_channels: Optional[int] = None,
        num_v_channels: Optional[int] = None,
        widening_factor: int = 1,
        drop_path_rate: float = 0.0,
        atten_drop: float = 0.1,
        mlp_drop: float = 0.5,
    ):
        self_attn = SelfAttention(
            num_heads=num_heads,
            num_channels=num_channels,
            num_qk_channels=num_qk_channels,
            num_v_channels=num_v_channels,
            dropout=atten_drop,
        )
        super().__init__(
            Residual(self_attn, atten_drop, drop_path_rate),
            Residual(MLP(num_channels, widening_factor), mlp_drop, drop_path_rate),
        )


class SelfAttentionBlock(Sequential):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        num_channels: int,
        num_qk_channels: Optional[int] = None,
        num_v_channels: Optional[int] = None,
        widening_factor: int = 1,
        max_dpr: float = 0.0,
        atten_drop: float = 0.1,
        mlp_drop: float = 0.5,
        activation_checkpointing: bool = False,
    ):
        dpr_list = [dpr.item() for dpr in torch.linspace(0, max_dpr, num_layers)]
        layers = [
            SelfAttentionLayer(
                num_heads=num_heads,
                num_channels=num_channels,
                num_qk_channels=num_qk_channels,
                num_v_channels=num_v_channels,
                widening_factor=widening_factor,
                drop_path_rate=dpr_list[i],
                atten_drop=atten_drop,
                mlp_drop=mlp_drop
            )
            for i in range(num_layers)
        ]

        if activation_checkpointing:
            layers = [checkpoint_wrapper(layer) for layer in layers]

        super().__init__(*layers)


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
        return self.drop_path(self.dropout(x) + args[0])


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


class OutputAdapter(nn.Module):
    def __init__(self, output_query: Tensor):
        """Transforms generic decoder cross-attention output to task-specific output.

        :param output_query: output query prototype (does not include batch dimension) used as query input to
            generic decoder cross-attention.
        """
        super().__init__()
        self._output_query = nn.Parameter(output_query)
        self._init_parameters()

    def _init_parameters(self):
        with torch.no_grad():   # 这里不会影响 self._output_query 本身是带梯度的
            # 把数值范围限定在 (-2.0, 2.0)
            self._output_query.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

    @property
    def num_output_query_channels(self):
        return self._output_query.shape[-1]

    def output_query(self, x):
        return repeat(self._output_query, "... -> b ...", b=x.shape[0])

    def forward(self, x):
        raise NotImplementedError()


class ClassificationOutputAdapter(OutputAdapter):
    def __init__(self, num_classes: int, num_output_queries: int = 1, num_output_query_channels: Optional[int] = None):

        if num_output_query_channels is None:
            num_output_query_channels = num_classes

        super().__init__(output_query=torch.empty(num_output_queries, num_output_query_channels))
        self.linear = nn.Linear(num_output_query_channels, num_classes)

    def forward(self, x):
        return self.linear(x).squeeze(dim=1)


class PerceiverEncoder(nn.Module):
    def __init__(
        self,
        input_adapter: InputAdapter,
        num_latents: int,
        num_latent_channels: int,
        num_cross_attention_heads: int = 4,
        num_cross_attention_qk_channels: Optional[int] = None,
        num_cross_attention_v_channels: Optional[int] = None,
        num_cross_attention_layers: int = 1,
        first_cross_attention_layer_shared: bool = False,
        cross_attention_widening_factor: int = 1,
        num_self_attention_heads: int = 4,
        num_self_attention_qk_channels: Optional[int] = None,
        num_self_attention_v_channels: Optional[int] = None,
        num_self_attention_layers_per_block: int = 6,
        num_self_attention_blocks: int = 1,
        first_self_attention_block_shared: bool = True,
        self_attention_widening_factor: int = 1,
        max_dpr: float = 0.0,
        atten_drop: float = 0.1,
        mlp_drop: float = 0.5,
        activation_checkpointing: bool = False
    ):
        """Generic Perceiver IO encoder.

        :param input_adapter: Transforms and position-encodes task-specific input to generic encoder input
            of shape (B, M, C) where B is the batch size, M the input sequence length and C the number of
            key/value input channels. C is determined by the `num_input_channels` property of the
            `input_adapter`.
        :param num_latents: Number of latent variables (N).
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
        :param num_self_attention_layers_per_block: Number of self-attention layers per self-attention block.
        :param num_self_attention_blocks: Number of self-attention blocks sharing weights between corresponding
            self-attention layers.
        :param first_self_attention_block_shared: Whether the first self-attention block should share its weights
            with subsequent self-attention blocks (if any).
        :param dropout: Dropout probability for self- and cross-attention layers and residuals.
        :param activation_checkpointing: If True, implements an activation checkpoint for each self-attention
            layer and cross-attention layer.
        """
        super().__init__()

        self.input_adapter = input_adapter

        if num_cross_attention_layers <= 0:
            raise ValueError("num_cross_attention_layers must be > 0")

        if num_self_attention_blocks <= 0:
            raise ValueError("num_self_attention_blocks must be > 0")

        if num_cross_attention_layers > num_self_attention_blocks:
            raise ValueError("num_cross_attention_layers must be <= num_self_attention_blocks")

        self.num_cross_attention_layers = num_cross_attention_layers
        self.num_self_attention_blocks = num_self_attention_blocks

        self.first_cross_attention_layer_shared = first_cross_attention_layer_shared
        self.first_self_attention_block_shared = first_self_attention_block_shared

        def cross_attn():
            layer = CrossAttentionLayer(
                num_heads=num_cross_attention_heads,
                num_q_input_channels=num_latent_channels,
                num_kv_input_channels=input_adapter.num_input_channels,
                num_qk_channels=num_cross_attention_qk_channels,
                num_v_channels=num_cross_attention_v_channels,
                widening_factor=cross_attention_widening_factor,
                atten_drop=atten_drop,
                mlp_drop=mlp_drop
            )
            return checkpoint_wrapper(layer) if activation_checkpointing else layer

        def self_attn():
            return SelfAttentionBlock(
                num_layers=num_self_attention_layers_per_block,
                num_heads=num_self_attention_heads,
                num_channels=num_latent_channels,
                num_qk_channels=num_self_attention_qk_channels,
                num_v_channels=num_self_attention_v_channels,
                widening_factor=self_attention_widening_factor,
                max_dpr=max_dpr,
                atten_drop=atten_drop,
                mlp_drop=mlp_drop,
                activation_checkpointing=activation_checkpointing
            )

        self.cross_attn_n = cross_attn()
        self.self_attn_n = self_attn()

        if self.first_cross_attention_layer_shared or num_cross_attention_layers == 1:
            self.cross_attn_1 = self.cross_attn_n
        else:
            self.cross_attn_1 = cross_attn()

        if self.first_self_attention_block_shared or num_self_attention_blocks == 1:
            self.self_attn_1 = self.self_attn_n
        else:
            self.self_attn_1 = self_attn()

        # learnable initial latent vectors
        self.latent = nn.Parameter(torch.empty(num_latents, num_latent_channels))
        self._init_parameters()

    def _init_parameters(self):
        with torch.no_grad():
            self.latent.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

    def forward(self, x, pad_mask=None):
        b, *_ = x.shape

        # encode task-specific input
        x = self.input_adapter(x)

        # repeat initial latent vector along batch dimension
        x_latent = repeat(self.latent, "... -> b ...", b=b)

        x_latent = self.cross_attn_1(x_latent, x, pad_mask)
        x_latent = self.self_attn_1(x_latent)

        for i in range(1, self.num_self_attention_blocks):
            if i < self.num_cross_attention_layers:
                x_latent = self.cross_attn_n(x_latent, x, pad_mask)
            x_latent = self.self_attn_n(x_latent)

        return x_latent


class PerceiverEncoder_feats_head(PerceiverEncoder):
    def __init__(self, 
        input_adapter: InputAdapter, 
        num_latents: int, 
        num_latent_channels: int, 
        num_cross_attention_heads: int = 4, 
        num_cross_attention_qk_channels: Optional[int] = None, 
        num_cross_attention_v_channels: Optional[int] = None, 
        num_cross_attention_layers: int = 1, 
        first_cross_attention_layer_shared: bool = False, 
        cross_attention_widening_factor: int = 1, 
        num_self_attention_heads: int = 4, 
        num_self_attention_qk_channels: Optional[int] = None, 
        num_self_attention_v_channels: Optional[int] = None, 
        num_self_attention_layers_per_block: int = 6, 
        num_self_attention_blocks: int = 1, 
        first_self_attention_block_shared: bool = True, 
        self_attention_widening_factor: int = 1, 
        max_dpr = 0.0,
        atten_drop: float = 0.1,
        mlp_drop: float = 0.5,
        activation_checkpointing: bool = False):
        super().__init__(input_adapter, num_latents, num_latent_channels, num_cross_attention_heads, num_cross_attention_qk_channels, num_cross_attention_v_channels, num_cross_attention_layers, first_cross_attention_layer_shared, cross_attention_widening_factor, num_self_attention_heads, num_self_attention_qk_channels, num_self_attention_v_channels, num_self_attention_layers_per_block, num_self_attention_blocks, first_self_attention_block_shared, self_attention_widening_factor, max_dpr, atten_drop, mlp_drop, activation_checkpointing)

        self.latent_head = nn.Sequential(
            nn.BatchNorm1d(2*num_latent_channels), 
            nn.ReLU(), 
            nn.Linear(2*num_latent_channels, num_latent_channels, bias = False),
            nn.BatchNorm1d(num_latent_channels), 
            nn.ReLU(), 
            nn.Linear(num_latent_channels, num_latent_channels, bias = False))

    def forward(self, x, pad_mask=None):
        b, *_ = x.shape

        # encode task-specific input
        x = self.input_adapter(x)

        # repeat initial latent vector along batch dimension
        x_latent = repeat(self.latent, "... -> b ...", b=b)

        x_latent = self.cross_attn_1(x_latent, x, pad_mask)
        x_latent = self.self_attn_1(x_latent)

        for i in range(1, self.num_self_attention_blocks):
            if i < self.num_cross_attention_layers:
                x_latent = self.cross_attn_n(x_latent, x, pad_mask)
            x_latent = self.self_attn_n(x_latent)

        backbone_feats = torch.cat([x_latent.max(1)[0], x_latent.mean(1)], dim=1)
        x_latent_feats = self.latent_head(backbone_feats)

        return x_latent_feats, backbone_feats


class PerceiverDecoder(nn.Module):
    def __init__(
        self,
        output_adapter: OutputAdapter,
        num_latent_channels: int,
        num_cross_attention_heads: int = 4,
        num_cross_attention_qk_channels: Optional[int] = None,
        num_cross_attention_v_channels: Optional[int] = None,
        cross_attention_widening_factor: int = 1,
        num_self_attention_heads: int = 4,
        num_self_attention_qk_channels: Optional[int] = None,
        num_self_attention_v_channels: Optional[int] = None,
        num_self_attention_layers_per_block: int = 2,
        self_attention_widening_factor: int = 1,
        atten_drop: float = 0.0,
        mlp_drop: float = 0.0,
        activation_checkpointing: bool = False,
    ):
        """Generic Perceiver IO decoder.

        :param output_adapter: Transforms generic decoder cross-attention output of shape (B, O, F) to task-specific
            output. B is the batch size, O the output sequence length and F the number of cross-attention output
            channels. F is determined by the `num_output_query_channels` property of the `output_adapter`.
        :param num_latent_channels: Number of latent channels (C_latent) as produced by a Perceiver IO encoder.
        :param num_cross_attention_heads: Number of cross-attention heads.
        :param num_cross_attention_qk_channels: Number of query and key channels for cross-attention
            (see `MultiHeadAttention.num_qk_channels` for details).
        :param num_cross_attention_v_channels: Number of value channels for cross-attention
            (see `MultiHeadAttention.num_v_channels` for details).
        :param dropout: Dropout probability for cross-attention layers and residuals.
        :param activation_checkpointing: If True, implements an activation checkpoint for the decoder's
            cross-attention layer.
        """
        super().__init__()

        cross_attn = CrossAttentionLayer(
            num_heads=num_cross_attention_heads,
            num_q_input_channels=output_adapter.num_output_query_channels,
            num_kv_input_channels=num_latent_channels,
            num_qk_channels=num_cross_attention_qk_channels,
            num_v_channels=num_cross_attention_v_channels,
            widening_factor=cross_attention_widening_factor,
            atten_drop=atten_drop,
            mlp_drop=mlp_drop,
        )

        self_attn = SelfAttentionBlock(
                num_layers=num_self_attention_layers_per_block,
                num_heads=num_self_attention_heads,
                num_channels=num_latent_channels,
                num_qk_channels=num_self_attention_qk_channels,
                num_v_channels=num_self_attention_v_channels,
                widening_factor=self_attention_widening_factor,
                max_dpr=0,  # In decoder's SA block, don't use DropPath
                atten_drop=atten_drop,
                mlp_drop=mlp_drop,
                activation_checkpointing=activation_checkpointing
            )

        if activation_checkpointing:
            cross_attn = checkpoint_wrapper(cross_attn)

        self.cross_attn = cross_attn
        self.self_attn = self_attn
        self.output_adapter = output_adapter

    def forward(self, x):
        ''' forward function in Decoder come from 3 parts
                1. output query array
                2. latent key array
                3. latent value array
            NOTE: 2. and 3. are outputs of Encoder
        '''
        # output_query: [batch_size, num_output_queries=1, num_output_query_channels=num_latent_channels]
        output_query = self.output_adapter.output_query(x)
        # output: [batch_size, num_output_queries=1, num_output_query_channels]
        x_latent = self.cross_attn(output_query, x)
        output = self.self_attn(x_latent)
        # output: [batch_size, num_classes]
        return self.output_adapter(output)


class PerceiverDecoder_var(nn.Module):
    def __init__(
        self,
        num_latent_channels: int,
        num_classes: int = 40, 
        mlp_drop: float = 0.0,
    ):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.BatchNorm1d(2*num_latent_channels),
            nn.ReLU(),
            nn.Linear(2*num_latent_channels, num_latent_channels),
            nn.BatchNorm1d(num_latent_channels),
            nn.ReLU(),
            nn.Linear(num_latent_channels, num_latent_channels//2),
            nn.BatchNorm1d(num_latent_channels//2),
            nn.ReLU(),
            nn.Linear(num_latent_channels//2, num_classes))

    def forward(self, x):
        '''
            Args:
                x: [batch, num_latents, num_latent_channels]

        '''
        # output_query: [batch_size, num_output_queries=1, num_output_query_channels=num_latent_channels]
        backbone_feats = torch.cat([x.max(1)[0], x.mean(1)], dim=1)
        # output: [batch_size, num_classes]
        output = self.decoder(backbone_feats)

        return output
        

class PerceiverIO(Sequential):
    ''' inherit Module Sequential
    '''
    def __init__(self, encoder: PerceiverEncoder, decoder: PerceiverDecoder):
        super().__init__(encoder, decoder)

    @property
    def encoder(self):
        return self[0]

    @property
    def decoder(self):
        return self[1]
