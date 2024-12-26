import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import UNetModel

class ConditionalUNet(UNetModel):
    def __init__(self, in_channels, model_channels, out_channels, num_res_blocks, attention_resolutions, 
                 num_heads, num_heads_upsample, dropout, channel_mult,
                 use_checkpoint, use_scale_shift_norm, num_classes=None):
        #print(num_classes)
        super(ConditionalUNet, self).__init__(
            in_channels=in_channels + 2,  # +1 for mask channel +1 for conditioning channel
            model_channels=model_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            num_heads=num_heads,
            num_heads_upsample=num_heads_upsample,
            dropout=dropout,
            channel_mult=channel_mult,
            num_classes=num_classes,
            use_checkpoint=use_checkpoint,
            use_scale_shift_norm=use_scale_shift_norm,
            
        )
        self.num_classes=num_classes
        self.lesion_load_fc = nn.Linear(1, model_channels)
        self.num_lesions_fc = nn.Linear(1, model_channels)
        self.healthy_fc = nn.Linear(1, model_channels)

        self.combination_net = nn.Sequential(
            nn.Linear(model_channels*3, model_channels),
            nn.ReLU(),
            nn.Linear(model_channels, 1)
        )

    def forward(self, x, timesteps, mask, lesion_load, num_lesions, lesion):
        lesion_load = lesion_load.view(-1, 1)  # Reshape to (batch_size, 1)
        num_lesions = num_lesions.view(-1, 1)  # Reshape to (batch_size, 1)
        lesion = lesion.view(-1, 1)  # Reshape to (batch_size, 1)
        lesion_load_emb = self.lesion_load_fc(lesion_load)
        num_lesions_emb = self.num_lesions_fc(num_lesions)
        lesion_emb = self.healthy_fc(lesion.float())

        combined_emb = self.combination_net(torch.cat([lesion_load_emb, 
                                                       num_lesions_emb,
                                                       lesion_emb], dim=1))
        combined_channel = combined_emb.unsqueeze(2).unsqueeze(3).expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat([x, mask, combined_channel], dim=1)
        if self.num_classes is not None:
            y = lesion.clone().view(-1).long()
            return super(ConditionalUNet, self).forward(x, timesteps, y=y)
        else:
            return super(ConditionalUNet, self).forward(x, timesteps)
