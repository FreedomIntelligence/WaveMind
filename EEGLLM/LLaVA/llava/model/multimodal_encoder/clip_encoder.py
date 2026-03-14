import requests
import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig,CLIPVisionModelWithProjection

import sys
import os
from PIL import Image
import io



def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        if self.select_feature == 'cls':
            self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        elif self.select_feature == 'cls_linear':
            self.vision_tower = CLIPVisionModelWithProjection.from_pretrained(self.vision_tower_name)
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')


        self.vision_tower.to(device=self.device, dtype=self.dtype)
        self.vision_tower.requires_grad_(False)
        self.vision_tower.enable_input_require_grads()

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        if self.select_feature == 'patch' or self.select_feature == 'cls_patch':
            image_features = image_forward_outs.hidden_states[self.select_layer]
            if self.select_feature == 'patch':
                image_features = image_features[:, 1:]
                # shape: (batch_size, num_patches, hidden_size)
            elif self.select_feature == 'cls_patch':
                image_features = image_features[:, :]
                # shape: (batch_size, num_patches+1, hidden_size)
        elif self.select_feature == 'cls':
            image_features = image_forward_outs.pooler_output.unsqueeze(1)
            # shape: (batch_size, 1,hidden_size)
        elif self.select_feature == 'cls_linear':
            image_features = image_forward_outs.image_embeds.unsqueeze(1)
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):

        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)   
            # image_forward_outs = self.vision_tower(**inputs, output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        if self.select_feature == 'cls' or self.select_feature == 'cls_linear':
            assert image_features.shape[1] == 1
            if self.select_feature == 'cls':
                assert image_features.shape[2] == 1024
            elif self.select_feature == 'cls_linear':
                assert image_features.shape[2] == 768
        elif self.select_feature == 'patch' or self.select_feature == 'cls_patch':
            assert image_features.shape[1] >1
            assert image_features.shape[2] >= 768
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        # assert image_features.shape[2] == 768
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size
    
    @property
    def mm_hidden_size(self):
        return self.config.mm_hidden_size
    

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2



class CLIPVisionTowerS2(CLIPVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__(vision_tower, args, delay_load)

        self.s2_scales = getattr(args, 's2_scales', '336,672,1008')
        self.s2_scales = list(map(int, self.s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError:
            raise ImportError('Package s2wrapper not found! Please install by running: \npip install git+https://github.com/bfshi/scaling_on_scales.git')
        self.multiscale_forward = multiscale_forward

        # change resize/crop size in preprocessing to the largest image size in s2_scale
        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.image_processor.size['shortest_edge'] = self.s2_image_size
            self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.image_processor.size['shortest_edge'] = self.s2_image_size
        self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

        self.is_loaded = True

    @torch.no_grad()
    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.multiscale_forward(self.forward_feature, image.unsqueeze(0), img_sizes=self.s2_scales, max_split_size=self.s2_split_size)
                image_features.append(image_feature)
        else:
            image_features = self.multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales, max_split_size=self.s2_split_size)

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)
    @property
    def mm_hidden_size(self):
        return self.config.mm_hidden_size
