from dataclasses import dataclass
from typing import Any, Optional, Tuple

from torch import nn


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
class PointCloudEncoderConfig(EncoderConfig):
    pointcloud_shape: Tuple[int, int, int] = (2048, 3)
    num_frequency_bands: int = 32


class PointCloudInputAdapter(InputAdapter):
    def __init__(self, pointcloud_shape: Tuple[int, ...], num_input_channels: int):
        super().__init__(num_input_channels=num_input_channels)

        _, self.point_channels = pointcloud_shape

        self.point_mlp = nn.Sequential(
            nn.Linear(self.point_channels, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, num_input_channels)
        )

    def forward(self, x):
        '''
        Args
            x: [batch_size, num_points, point_channels]
        Return
            point_feats: [batch_size, num_points, num_input_channels]
        '''
        # if (n, c) != (self.num_points, self.point_channels):
            # raise ValueError(f"Input pointcloud shape {(n, c)} different from required shape {(self.num_points, self.point_channels)}")

        point_feats = self.point_mlp(x)

        return point_feats


class PointCloudClassifier(PerceiverIO):
    def __init__(self, config: PerceiverConfig[PointCloudEncoderConfig, ClassificationDecoderConfig]):
        input_adapter = PointCloudInputAdapter(
            pointcloud_shape=config.encoder.pointcloud_shape, num_frequency_bands=config.encoder.num_frequency_bands
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
