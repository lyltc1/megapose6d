from typing import List, Optional, Callable
import torch
import torch.nn as nn

from .embedding import TimeStepEmbedding, PoseEmbedding

class Denoiser(nn.Module):
    def __init__(self, cfg):
        super(Denoiser, self).__init__()
        self.cfg = cfg
        self.target_dim = cfg.target_dim
        self.time_embed = TimeStepEmbedding()
        self.pose_embed = PoseEmbedding(target_dim=self.target_dim)
        self.flatten = nn.Flatten()
        self.mlp = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, cfg.z_dim),
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        first_dim = (
            self.time_embed.out_dim
            + self.pose_embed.out_dim
            + cfg.z_dim
        )

        self._first_xy = nn.Linear(first_dim, cfg.d_model)
        self._trunk_xy = TransformerEncoderWrapper()
        self._last_xy = MLP(
            cfg.d_model, [cfg.mlp_hidden_dim, 2], norm_layer=nn.LayerNorm
        )

        self._first_z = nn.Linear(first_dim, cfg.d_model)
        self._trunk_z = TransformerEncoderWrapper()
        self._last_z = MLP(
            cfg.d_model, [cfg.mlp_hidden_dim, 1], norm_layer=nn.LayerNorm
        )

        self._first_r = nn.Linear(first_dim, cfg.d_model)
        self._trunk_r = TransformerEncoderWrapper()
        self._last_r = MLP(
            cfg.d_model, [cfg.mlp_hidden_dim, 4], norm_layer=nn.LayerNorm
        )


    def forward(self, x: torch.Tensor, t: torch.Tensor, z: torch.Tensor):
        # # B x N x dim  # B  # B x N x dim_z
        # B, _ = x.shape
        # t_emb = self.time_embed(t)
        # # expand t from B x C to B x N x C
        # x_emb = self.pose_embed(x)
        # #z = z.permute(0, 2, 1)
        # z = self.avg_pool(z)
        # z = z.squeeze(-1)
        # feed_feats = torch.cat([x_emb, t_emb, z], dim=-1)
        B, N, _ = z.shape
        z = self.mlp(z)

        t_emb = self.time_embed(t)
        # expand t from B x C to B x N x C
        t_emb = t_emb.view(B, 1, t_emb.shape[-1]).expand(-1, N, -1)

        x_emb = self.pose_embed(x).view(B, 1, -1).expand(-1, N, -1)
        feed_feats = torch.cat([x_emb, t_emb, z], dim=-1)

        output_xy = self._last_xy(self._trunk_xy(self._first_xy(feed_feats)))
        output_z = self._last_z(self._trunk_z(self._first_z(feed_feats)))
        output_r = self._last_r(self._trunk_r(self._first_r(feed_feats)))
        return torch.cat([output_xy, output_z, output_r], dim=-1)[:, 0, :]

def TransformerEncoderWrapper():
    encoder_layer = torch.nn.TransformerEncoderLayer(
        d_model=512,
        nhead=4,
        dim_feedforward=1024,
        dropout=0.1,
        batch_first=True,
        norm_first=True,
    )
    _trunk = torch.nn.TransformerEncoder(encoder_layer, 8)
    return _trunk


class MLP(torch.nn.Sequential):
    """This block implements the multi-layer perceptron (MLP) module.

    Args:
        in_channels (int): Number of channels of the input
        hidden_channels (List[int]): List of the hidden channel dimensions
        norm_layer (Callable[..., torch.nn.Module], optional):
            Norm layer that will be stacked on top of the convolution layer.
            If ``None`` this layer wont be used. Default: ``None``
        activation_layer (Callable[..., torch.nn.Module], optional):
            Activation function which will be stacked on top of the
            normalization layer (if not None), otherwise on top of the
            conv layer. If ``None`` this layer wont be used.
            Default: ``torch.nn.ReLU``
        inplace (bool): Parameter for the activation layer, which can
            optionally do the operation in-place. Default ``True``
        bias (bool): Whether to use bias in the linear layer. Default ``True``
        dropout (float): The probability for the dropout layer. Default: 0.0
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        inplace: Optional[bool] = True,
        bias: bool = True,
        norm_first: bool = False,
        dropout: float = 0.0,
    ):
        # The addition of `norm_layer` is inspired from
        # the implementation of TorchMultimodal:
        # https://github.com/facebookresearch/multimodal/blob/5dec8a/torchmultimodal/modules/layers/mlp.py
        params = {} if inplace is None else {"inplace": inplace}

        layers = []
        in_dim = in_channels

        for hidden_dim in hidden_channels[:-1]:
            if norm_first and norm_layer is not None:
                layers.append(norm_layer(in_dim))

            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))

            if not norm_first and norm_layer is not None:
                layers.append(norm_layer(hidden_dim))

            layers.append(activation_layer(**params))

            if dropout > 0:
                layers.append(torch.nn.Dropout(dropout, **params))

            in_dim = hidden_dim

        if norm_first and norm_layer is not None:
            layers.append(norm_layer(in_dim))

        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias))
        if dropout > 0:
            layers.append(torch.nn.Dropout(dropout, **params))

        super().__init__(*layers)
