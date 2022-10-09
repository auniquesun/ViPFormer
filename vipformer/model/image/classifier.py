import math

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
from einops import rearrange, repeat
from torch import Tensor

from vipformer.model.core import (
    ClassificationDecoderConfig,
    ClassificationOutputAdapter,
    EncoderConfig,
    InputAdapter,
    PerceiverConfig,
    PerceiverDecoder,
    PerceiverEncoder,
    PerceiverIO,
)


@dataclass
class ImageEncoderConfig(EncoderConfig):
    image_shape: Tuple[int, int, int] = (224, 224, 3)
    num_frequency_bands: int = 32


class ImageInputAdapter(InputAdapter):
    def __init__(self, image_shape: Tuple[int, ...], num_frequency_bands: int):
        # spatial_shape: [224,224]   num_image_channels: [3]
        *self.spatial_shape, num_image_channels = image_shape
        self.image_shape = image_shape
        # num_frequency_bands 是什么意思，还没搞懂
        self.num_frequency_bands = num_frequency_bands

        super().__init__(num_input_channels=num_image_channels + self._num_position_encoding_channels())

        # create encodings for single example
        pos = self._positions()
        enc = self._position_encodings(pos)

        # flatten encodings along spatial dimensions
        enc = rearrange(enc, "... c -> (...) c")

        # position encoding prototype
        self.register_buffer("position_encoding", enc)

    def _positions(self, v_min=-1.0, v_max=1.0):
        """Create evenly spaced position coordinates for self.spatial_shape with values in [v_min, v_max].

        :param v_min: minimum coordinate value per dimension.
        :param v_max: maximum coordinate value per dimension.
        :return: position coordinates tensor of shape (*shape, len(shape)).
        """
        # coords: [2, 224]
        coords = [torch.linspace(v_min, v_max, steps=s) for s in self.spatial_shape]
        # [224, 224, 2]
        return torch.stack(torch.meshgrid(*coords), dim=len(self.spatial_shape))

    def _position_encodings(
        self, p: Tensor, max_frequencies: Optional[Tuple[int, ...]] = None, include_positions: bool = True
    ) -> Tensor:
        """Fourier-encode positions p using self.num_bands frequency bands.

        :param p: positions of shape (*d, c) where c = len(d).
        :param max_frequencies: maximum frequency for each dimension (1-tuple for sequences,
               2-tuple for images, ...). If `None` values are derived from shape of p.
        :param include_positions: whether to include input positions p in returned encodings tensor.
        :returns: position encodings tensor of shape (*d, c * (2 * num_bands + include_positions)).
        """
        encodings = []

        if max_frequencies is None:
            # p: [224, 224, 2]
            # max_frequencies: [224, 224]
            max_frequencies = p.shape[:-1]

        # frequencies: [2, 64]
        frequencies = [
            torch.linspace(1.0, max_freq / 2.0, self.num_frequency_bands, device=p.device)
            for max_freq in max_frequencies
        ]
        frequency_grids = []

        for i, frequencies_i in enumerate(frequencies):
            # var1 = p[..., i : i + 1]   -> [224, 224, 1]
            # var2 = frequencies_i[None, ...] -> [1, 64]
            # var1 * var2 -> [224, 224, 64]
            frequency_grids.append(p[..., i : i + 1] * frequencies_i[None, ...])
        # 出了循环
        # frequency_grids: [2, 224, 224, 64]

        if include_positions:
            # p: [224, 224, 2]
            # p 加入 encodings
            encodings.append(p)

        # torch.sin(math.pi * frequency_grid) -> [224, 224, 64]
        # 2个 [224, 224, 64] 的tensor加入encodings
        encodings.extend([torch.sin(math.pi * frequency_grid) for frequency_grid in frequency_grids])
        # torch.cos(math.pi * frequency_grid) -> [224, 224, 64]
        # 2个 [224, 224, 64] 的tensor加入encodings
        encodings.extend([torch.cos(math.pi * frequency_grid) for frequency_grid in frequency_grids])

        # [224,224,2+64+64]
        return torch.cat(encodings, dim=-1)

    def _num_position_encoding_channels(self, include_positions: bool = True) -> int:
        # len(self.spatial_shape) -> 2
        # self.num_frequency_bands -> 64
        # 为什么还要乘 len(spatial_shape)，是在干什么？
        return len(self.spatial_shape) * (2 * self.num_frequency_bands + include_positions)

    def forward(self, x):
        b, *d = x.shape

        if tuple(d) != self.image_shape:
            raise ValueError(f"Input image shape {tuple(d)} different from required shape {self.image_shape}")

        # x_enc: [b, ]
        x_enc = repeat(self.position_encoding, "... -> b ...", b=b)
        # x: [b, 224*224, 3]
        x = rearrange(x, "b ... c -> b (...) c")
        # 这里相当于把 rgb channel 和 position channel 连接起来，但问题是 
        # position channel 和输入channel是相加的关系呀
        # perceiver 是这么搞的呀，没有采用两个编码相加的方式，而是channel拼接
        return torch.cat([x, x_enc], dim=-1)


class ImageClassifier(PerceiverIO):
    def __init__(self, config: PerceiverConfig[ImageEncoderConfig, ClassificationDecoderConfig]):
        input_adapter = ImageInputAdapter(
            image_shape=config.encoder.image_shape, num_frequency_bands=config.encoder.num_frequency_bands
        )

        encoder_kwargs = config.encoder.base_kwargs()
        if encoder_kwargs["num_cross_attention_qk_channels"] is None:
            encoder_kwargs["num_cross_attention_qk_channels"] = input_adapter.num_input_channels

        encoder = PerceiverEncoder(
            input_adapter=input_adapter,
            num_latents=config.num_latents,
            num_latent_channels=config.num_latent_channels,
            activation_checkpointing=config.activation_checkpointing,
            **encoder_kwargs,
        )
        output_adapter = ClassificationOutputAdapter(
            num_classes=config.decoder.num_classes,
            num_output_queries=config.decoder.num_output_queries,
            num_output_query_channels=config.decoder.num_output_query_channels,
        )
        decoder = PerceiverDecoder(
            output_adapter=output_adapter,
            num_latent_channels=config.num_latent_channels,
            activation_checkpointing=config.activation_checkpointing,
            **config.decoder.base_kwargs(),
        )
        super().__init__(encoder, decoder)
