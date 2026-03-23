#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.nn as nn
import random
import warnings



from .multimodal_encoder.builder import build_neuro_tower, build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)
        self.config=config
        if hasattr(config, "mm_vision_tower"):
            self.mm_projector = build_vision_projector(config)
            if hasattr(config, "mm_vision_tower"):
                self.vision_tower = build_vision_tower(config, delay_load=True)
                self.neuro_tower = build_neuro_tower(config)
            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )
            print("Generation mode")
        else:
            print("Training Mode")



    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def get_neuro_tower(self):
        neuro_tower = getattr(self, 'neuro_tower', None)
        if type(neuro_tower) is list:
            neuro_tower = neuro_tower[0]
        return neuro_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type
        freeze_neuro_tower = model_args.freeze_neuro_tower
        random_neuro_tower = model_args.random_neuro_tower

        self.config.mm_vision_tower = vision_tower


        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
            vision_tower.load_model()
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        if self.get_neuro_tower() is None:
            neuro_tower = build_neuro_tower(model_args)
            if fsdp is not None and len(fsdp) > 0:
                self.neuro_tower = [neuro_tower]
            else:
                self.neuro_tower= neuro_tower
            neuro_tower.load_model()

        else:
            if fsdp is not None and len(fsdp) > 0:
                neuro_tower = self.neuro_tower[0]
            else:
                neuro_tower= self.neuro_tower
            neuro_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = model_args.mm_hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True


        if pretrain_mm_mlp_adapter is not None:
            import os
            if not os.path.exists(pretrain_mm_mlp_adapter):
                checkpoints = sorted(list(Path(os.path.dirname(pretrain_mm_mlp_adapter)).glob("checkpoint-*")), key=lambda x: int(x.name.split('-')[-1]))
                if len(checkpoints) > 0:
                    pretrain_mm_mlp_adapter = os.path.join(os.path.dirname(pretrain_mm_mlp_adapter), checkpoints[-1],'non_lora_trainables.bin')
                else:
                    raise FileNotFoundError(f"pretrain_mm_mlp_adapter not found in {pretrain_mm_mlp_adapter}, please check the path.")
            
            
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            print(f"Loading pretrained mm projector weights from {pretrain_mm_mlp_adapter}")
            
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))
        else:
            print("pretrained mm projector weights provided, using random initialization.")

        if freeze_neuro_tower:
            print("Freezing neuro tower parameters")
            for param in self.neuro_tower.parameters():
                param.requires_grad = False
        else:
            print("Unfreezing neuro tower parameters")
            for param in self.neuro_tower.parameters():
                param.requires_grad = True
                
        if random_neuro_tower:
            print("Randomly initialize NeuroTower")
            neuro_para= self.neuro_tower.parameters()
            for p in neuro_para:
                if p.dim() > 1:
                    torch.nn.init.uniform_(p, -0.1, 0.1)


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor

import torch.nn.functional as F
from data.Utils import get_wavemind_root
class LlavaMetaForCausalLM(ABC):
    def __init__(self,*args, **kwargs):
        super().__init__()
        from llava.model.multimodal_encoder.eeg_encoder import DBsearch
        import os
        #############################
        
        if os.environ.get('WaveMind_ROOT_PATH_') is None or os.environ.get('WaveMind_ROOT_PATH_') == "":
            root = get_wavemind_root()
        else:
            root = os.environ['WaveMind_ROOT_PATH_']
        CLIP_path=os.path.join(root,"data/Total/CLIP_groundTruth") 
        
        self.DBtool=DBsearch(CLIP_path=CLIP_path)
        ##########################

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        model=self.get_model()
        assert model is not None, "Model is not initialized"
        vision_tower=model.get_vision_tower()
        assert vision_tower is not None, "Vision tower is not initialized"
        return vision_tower

    def get_neuro_tower(self):
        model=self.get_model()
        assert model is not None, "Model is not initialized"
        neuro_tower=model.get_neuro_tower()
        assert neuro_tower is not None, "Neuro tower is not initialized"
        return neuro_tower

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images).contiguous()
        mm_projector = self.get_model().mm_projector
        mm_device = next(mm_projector.parameters()).device
        mm_dtype= next(mm_projector.parameters()).dtype
        image_features=image_features.to(mm_device,dtype=mm_dtype)
        image_features=F.normalize(image_features, p=2, dim=-1)
        image_features = self.get_model().mm_projector(image_features).contiguous()
        return image_features

    def encode_eegs(self, images):
        eeg_features = self.get_model().get_neuro_tower()(images).contiguous()
        mm_projector = self.get_model().mm_projector
        mm_device = next(mm_projector.parameters()).device
        mm_dtype= next(mm_projector.parameters()).dtype
        eeg_features=eeg_features.to(mm_device,dtype=mm_dtype)

        # Validate EEG features are not zero vectors before normalization
        eeg_norms = torch.norm(eeg_features, p=2, dim=-1)
        if (eeg_norms < 1e-8).any():
            raise ValueError("EEG features contain zero or near-zero vectors (L2 norm < 1e-8), cannot normalize. This may indicate issues with the EEG encoder output.")

        eeg_features=F.normalize(eeg_features, p=2, dim=-1)
        eeg_features = self.get_model().mm_projector(eeg_features).contiguous()
        return eeg_features



    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                print("Loading pretrained MLP adapter...")
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
            else:
                raise ValueError("Please set --pretrain-mm-mlp-adapter to the path of the pretrained MLP adapter.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        modilitys,modility_types
    ):
        vision_tower = self.get_vision_tower()
        neuro_tower = self.get_neuro_tower()

        if type(modilitys) is list:
            assert len(modilitys)==len(modility_types)
            for i in range(len(modilitys)):
                if (modilitys[i] is None and modility_types[i] is not None) or (modilitys[i] is not None and modility_types[i] is None):
                    raise ValueError("The modilitys and modility_types should be consistent.")

        if vision_tower is None or neuro_tower is None or modilitys is None or len(modilitys)==0 or all(x is None for x in modilitys):
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        assert type(modilitys) == list and type(modility_types) == list

        assert len(modilitys)==len(modility_types)

        if len(modilitys)>0 and len(modilitys)!=input_ids.shape[0]:
            raise ValueError("The number of modilitys should be equal to the number of input_ids.")


        eegs = [modilitys[i] for i, t in enumerate(modility_types) if t == 'eeg']
        images = [modilitys[i] for i, t in enumerate(modility_types) if t == 'image']

        if type(images) is list and len(images) > 0:
            images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat(images, dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [img.shape[0] for img in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            image_features = []

        if type(eegs) is list and len(eegs) > 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                eegs=torch.tensor(eegs) if not isinstance(eegs[0], torch.Tensor) else eegs
                eegs = [x.unsqueeze(0) if x.ndim == 3 else x for x in eegs]
            concat_eegs = torch.stack(eegs, dim=0).to(dtype=torch.float32)
            eeg_features = [x.unsqueeze(0) if x.ndim == 1 else x for x in self.encode_eegs(concat_eegs)]

        else:
            eeg_features = []

        total_features = []
        img_idx = 0
        eeg_idx = 0
        for t in modility_types:
            if t == 'image':
                total_features.append(image_features[img_idx])
                img_idx += 1
            elif t == 'eeg':
                total_features.append(eeg_features[eeg_idx])
                eeg_idx += 1
            elif t is None:
                total_features.append(None)
            else:
                raise ValueError(f"Unknown modality type: {t}. Supported types are 'image', 'eeg', and None.")


        total_features = [x for x in total_features]


        image_features=total_features

        # print("------------------------")
        # print(input_ids)

        if type(input_ids) is list:
            assert len(image_features) == len(modility_types)==1, "batch_size should be equal to modility_types"
        elif type(input_ids) is torch.Tensor and len(input_ids.shape)==1:
            assert len(image_features) == len(modility_types)==1, "batch_size should be equal to modility_types"
        elif type(input_ids) is torch.Tensor and len(input_ids.shape) > 1:
            assert len(image_features) == len(modility_types)==input_ids.shape[0], "batch_size should be equal to modility_types"

        if type(input_ids) is list or type(input_ids) is torch.Tensor and len(input_ids.shape) > 1:
            for batch_idx, cur_input_ids in enumerate(input_ids):
                num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
                assert num_images<=1
                if num_images == 0:
                    assert image_features[batch_idx] is None
                else:
                    assert image_features[batch_idx] is not None
            


        # print("image_features",len(image_features))
        # print("modility_types",modility_types)
        # print("eeg_features",image_features[0].shape)
        



        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]


        # for batch_idx, cur_input_ids in enumerate(input_ids):
        #     print("sample idx",batch_idx)
        #     print("cur_input_ids",cur_input_ids)
        #     print(len(cur_input_ids))

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            # print("sample idx",batch_idx)
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            assert num_images<=1
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                assert cur_image_features is None, "The image features should be None if there is no image token in the input ids"
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                # cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds_1)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                # print("Text:cur_input_embeds_1",cur_input_embeds_1.shape)
                # print("Text:labels",labels[batch_idx].shape)
                # print(labels[batch_idx])
                
                # print("Text:cur_image_idx",cur_image_idx)
                # print("----------------")
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            # print("image_token_indices",image_token_indices)
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]


            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            # print("-----------------------------")
            # print("multi before")
            # print("cur_input_ids",len(cur_input_ids))
            # print(cur_input_ids)
            # print("cur_labels",len(cur_labels))
            # print(cur_labels)
            # print("-----------------------------")


            use_image=False
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                # print("----------------------------")
                # print("cur_new_input_embeds",cur_input_embeds_no_im[i].shape)
                # print("cur_new_labels",cur_labels_noim[i].shape)
                # print(cur_labels_noim[i])
                # print("----------------------------")
                if i < num_images:
                    # print("插入图片")
                    use_image=True
                    cur_image_features = image_features[cur_image_idx]
                    assert cur_image_features is not None, "The image features should not be None if there is an image token in the input ids"

                    #############################
                    # cur_image_features = cur_image_features.repeat(576, 1)
                    #############################


                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    # print("----------------------------")
                    # print("cur_image_features",cur_image_features.shape)
                    # print("cur_new_labels",torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype).shape)
                    # print("----------------------------")
                    
            



            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]


            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            # print("----------------------------")
            # print("multi after")
            # print("cur_new_input_embeds_in",cur_new_input_embeds.shape)
            # print("cur_new_labels_in",cur_new_labels.shape)
            # print(cur_new_labels)
            # print("cur_image_idx",cur_image_idx)
            # print("----------------------------")

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
            assert use_image==True,"image embed should insert"
        # print("-----------------------------")
        # print("cur_image_idx",cur_image_idx)
        # print("len(image_features)",len([x for x in image_features]))
        assert cur_image_idx==len([x for x in image_features]),"the image token should be same as the image features"
        
        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            # 理论上每个iter cur_new_embed和cur_new_labels不等长
            # print("-------------------------------------------")
            # print("cur_new_embed_66666",cur_new_embed.shape)
            # print("cur_new_labels_66666",cur_new_labels.shape)
            # print("-------------------------------------------")



            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                raise NotImplementedError("Left padding not implemented for EEG processing")
                # new_input_embeds_padded.append(torch.cat((
                #     torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                #     cur_new_embed
                # ), dim=0))
                # if cur_len > 0:
                #     new_labels_padded[i, -cur_len:] = cur_new_labels
                #     attention_mask[i, -cur_len:] = True
                #     position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)



        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        # print(66666666666)
        # for id,att_mask in enumerate(attention_mask):
        #     print(att_mask)
        #     print(new_labels[id])
        # sys.exit()



        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels





    # def prepare_inputs_labels_for_multimodal(
    #     self, input_ids, position_ids, attention_mask, past_key_values, labels,
    #     images, image_sizes=None,eegs=None
    # ):
    #     vision_tower = self.get_vision_tower()
    #     neuro_tower = self.get_neuro_tower()
    #     if vision_tower is None or images is None or input_ids.shape[1] == 1:
    #         return input_ids, position_ids, attention_mask, past_key_values, None, labels

    #     if type(images) is list or images.ndim == 5:
    #         if type(images) is list:
    #             images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
    #         concat_images = torch.cat([image for image in images], dim=0)

    #         image_features = self.encode_images(concat_images)
    #         split_sizes = [image.shape[0] for image in images]
    #         image_features = torch.split(image_features, split_sizes, dim=0)
    #         mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
    #         image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
    #         if mm_patch_merge_type == 'flat':
    #             image_features = [x.flatten(0, 1) for x in image_features]
    #         elif mm_patch_merge_type.startswith('spatial'):
    #             new_image_features = []
    #             for image_idx, image_feature in enumerate(image_features):
    #                 if image_feature.shape[0] > 1:
    #                     base_image_feature = image_feature[0]
    #                     image_feature = image_feature[1:]
    #                     height = width = self.get_vision_tower().num_patches_per_side
    #                     assert height * width == base_image_feature.shape[0]
    #                     if image_aspect_ratio == 'anyres':
    #                         num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
    #                         image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
    #                     else:
    #                         raise NotImplementedError
    #                     if 'unpad' in mm_patch_merge_type:
    #                         image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
    #                         image_feature = image_feature.flatten(1, 2).flatten(2, 3)
    #                         image_feature = unpad_image(image_feature, image_sizes[image_idx])
    #                         image_feature = torch.cat((
    #                             image_feature,
    #                             self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
    #                         ), dim=-1)
    #                         image_feature = image_feature.flatten(1, 2).transpose(0, 1)
    #                     else:
    #                         image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
    #                         image_feature = image_feature.flatten(0, 3)
    #                     image_feature = torch.cat((base_image_feature, image_feature), dim=0)
    #                 else:
    #                     image_feature = image_feature[0]
    #                     if 'unpad' in mm_patch_merge_type:
    #                         image_feature = torch.cat((
    #                             image_feature,
    #                             self.model.image_newline[None].to(image_feature.device)
    #                         ), dim=0)
    #                 new_image_features.append(image_feature)
    #             image_features = new_image_features
    #         else:
    #             raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
    #     else:
    #         image_features = self.encode_images(images)

    #     # print(f"image_features: {image_features.shape}")

    #     # TODO: image start / end is not implemented here to support pretraining.
    #     if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
    #         raise NotImplementedError

    #     # Let's just add dummy tensors if they do not exist,
    #     # it is a headache to deal with None all the time.
    #     # But it is not ideal, and if you have a better idea,
    #     # please open an issue / submit a PR, thanks.
    #     _labels = labels
    #     _position_ids = position_ids
    #     _attention_mask = attention_mask
    #     if attention_mask is None:
    #         attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    #     else:
    #         attention_mask = attention_mask.bool()
    #     if position_ids is None:
    #         position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
    #     if labels is None:
    #         labels = torch.full_like(input_ids, IGNORE_INDEX)

    #     # remove the padding using attention_mask -- FIXME
    #     _input_ids = input_ids
    #     input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
    #     labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

    #     new_input_embeds = []
    #     new_labels = []
    #     cur_image_idx = 0
    #     for batch_idx, cur_input_ids in enumerate(input_ids):
    #         num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
    #         if num_images == 0:
    #             cur_image_features = image_features[cur_image_idx]
    #             cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
    #             cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
    #             new_input_embeds.append(cur_input_embeds)
    #             new_labels.append(labels[batch_idx])
    #             cur_image_idx += 1
    #             continue

    #         image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
    #         cur_input_ids_noim = []
    #         cur_labels = labels[batch_idx]
    #         cur_labels_noim = []
    #         for i in range(len(image_token_indices) - 1):
    #             cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
    #             cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
    #         split_sizes = [x.shape[0] for x in cur_labels_noim]
    #         cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
    #         cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
    #         cur_new_input_embeds = []
    #         cur_new_labels = []

    #         for i in range(num_images + 1):
    #             cur_new_input_embeds.append(cur_input_embeds_no_im[i])
    #             cur_new_labels.append(cur_labels_noim[i])
    #             if i < num_images:
    #                 cur_image_features = image_features[cur_image_idx]
    #                 cur_image_idx += 1
    #                 cur_new_input_embeds.append(cur_image_features)
    #                 cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

    #         cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

    #         cur_new_input_embeds = torch.cat(cur_new_input_embeds)
    #         cur_new_labels = torch.cat(cur_new_labels)

    #         new_input_embeds.append(cur_new_input_embeds)
    #         new_labels.append(cur_new_labels)

    #     # Truncate sequences to max length as image embeddings can make the sequence longer
    #     tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
    #     if tokenizer_model_max_length is not None:
    #         new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
    #         new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

    #     # Combine them
    #     max_len = max(x.shape[0] for x in new_input_embeds)
    #     batch_size = len(new_input_embeds)

    #     new_input_embeds_padded = []
    #     new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
    #     attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
    #     position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

    #     for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
    #         cur_len = cur_new_embed.shape[0]
    #         if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
    #             new_input_embeds_padded.append(torch.cat((
    #                 torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
    #                 cur_new_embed
    #             ), dim=0))
    #             if cur_len > 0:
    #                 new_labels_padded[i, -cur_len:] = cur_new_labels
    #                 attention_mask[i, -cur_len:] = True
    #                 position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
    #         else:
    #             new_input_embeds_padded.append(torch.cat((
    #                 cur_new_embed,
    #                 torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
    #             ), dim=0))
    #             if cur_len > 0:
    #                 new_labels_padded[i, :cur_len] = cur_new_labels
    #                 attention_mask[i, :cur_len] = True
    #                 position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

    #     new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

    #     if _labels is None:
    #         new_labels = None
    #     else:
    #         new_labels = new_labels_padded

    #     if _attention_mask is None:
    #         attention_mask = None
    #     else:
    #         attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

    #     if _position_ids is None:
    #         position_ids = None

    #     return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
