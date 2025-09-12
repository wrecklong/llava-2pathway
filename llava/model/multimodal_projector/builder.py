# Based on https://github.com/haotian-liu/LLaVA.

import torch
import torch.nn as nn
import re

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, fpn_input_dim=[], **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')
    # if getattr(config, 'mm_use_4_vision_tokens', False):
    #     mm_hidden_size = config.mm_hidden_size * 4
    # else:
    mm_hidden_size = config.mm_hidden_size
    if projector_type == 'linear':
        return nn.Linear(mm_hidden_size, config.hidden_size)
    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)
    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')
