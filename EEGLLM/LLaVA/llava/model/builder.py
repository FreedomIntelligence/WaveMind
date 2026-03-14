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


import os
import sys
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch
from llava.model import *
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", attn_implementation="sdpa", **kwargs):
  
    kwargs = {"device_map": device_map, **kwargs}
  
    if device != "cuda":
        kwargs['device_map'] = {"": device}


    if load_8bit:
        kwargs['load_in_8bit'] = True
    
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        from transformers import BitsAndBytesConfig
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
        kwargs.pop('load_in_4bit')
    else:
        kwargs['torch_dtype'] = torch.float16
        


    if 'timestep' in kwargs and kwargs['timestep'] is not None:
        model_path= os.path.join(model_path, f"checkpoint-{kwargs['timestep']}")
        kwargs.pop('timestep')
        
    if 'llava' in model_name.lower() or 'mela' in model_name.lower():
        # Load LLaVA model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')

        # if ('lora' in model_name.lower()) and model_base is not None:
        if model_base is not None:
            print('Loading LLaVA from base model...')
            
            ############################################ 
            if "mixtral" in model_base.lower():
                print('Loading Mixtral as backbone...')
                from llava.model.language_model.llava_mixtral import LlavaMixtralConfig
                lora_cfg_pretrained = LlavaMixtralConfig.from_pretrained(model_path)
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                model = LlavaMixtralForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, attn_implementation=attn_implementation, **kwargs)
            elif "mistral" in model_base.lower():
                print('Loading Mistral as backbone...')
                from llava.model.language_model.llava_mistral import LlavaMistralConfig
                lora_cfg_pretrained = LlavaMistralConfig.from_pretrained(model_path)
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                model = LlavaMistralForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, attn_implementation=attn_implementation, **kwargs)
            elif "gemma" in model_base.lower():
                print('Loading Gemma as backbone...')
                from llava.model.language_model.llava_gemma import LlavaGemmaConfig
                lora_cfg_pretrained = LlavaGemmaConfig.from_pretrained(model_path)
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                model = LlavaGemmaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, attn_implementation=attn_implementation, **kwargs)
            elif "vicuna" in model_base.lower() or "llama" in model_base.lower():
                print('Loading Vicuna or LLaMA as backbone...')
                from llava.model.language_model.llava_llama import LlavaConfig
                from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
                lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, attn_implementation=attn_implementation, **kwargs)
            elif "qwen" in model_base.lower() or "quyen" in model_base.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_base)
                from llava.model.language_model.llava_qwen import LlavaQwenConfig
                lora_cfg_pretrained = LlavaQwenConfig.from_pretrained(model_path)
                if "moe" in model_base.lower() or "A14B" in model_base.lower():
                    from llava.model.language_model.llava_qwen_moe import LlavaQwenMoeConfig
                    model = LlavaQwenMoeForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, attn_implementation=attn_implementation, config=lora_cfg_pretrained, **kwargs)
                else:
                    from llava.model.language_model.llava_qwen import LlavaQwenConfig
                    model = LlavaQwenForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, attn_implementation=attn_implementation, config=lora_cfg_pretrained, **kwargs)
            else:
                raise ValueError(f"Invalid model name {model_base}. Please check the model name.")
            ############################################

            
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
            # 




            print(f'Loading additional LLaVA weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                print(f"Loading at path {os.path.join(model_path, 'non_lora_trainables.bin')}...")
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                try:
                    print(f'Loading from HF Hub {model_path}')
                    # this is probably from HF Hub
                    from huggingface_hub import hf_hub_download
                    def load_from_hf(repo_id, filename, subfolder=None):
                        cache_file = hf_hub_download(
                            repo_id=repo_id,
                            filename=filename,
                            subfolder=subfolder)
                        return torch.load(cache_file, map_location='cpu')
                    non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
                except Exception as e:
                    print(f"Failed to load non_lora_trainables.bin from {model_path}. Error: {e}")
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
                
                
            if any(k.startswith('model.neuro_tower.') for k in non_lora_trainables):
                model.get_neuro_tower().load_model(load_checkpoint=True)
                model.get_neuro_tower().is_loaded = True
            
            
            missing_keys, unexpected_keys=model.load_state_dict(non_lora_trainables, strict=False)
            assert unexpected_keys == [], f"Unexpected keys found in non-LoRA trainables: {unexpected_keys}"
            
            if ('lora' in model_name.lower()):
                from peft import PeftModel
                print(f'Loading LoRA weights,from {model_path}')
                model = PeftModel.from_pretrained(model, model_path)
                print('Merging LoRA weights...')
                model = model.merge_and_unload()
                print('Model is loaded...')

            
        # elif ('pretrain' in model_name.lower()) and model_base is not None:
        #     # this may be mm projector only
        #     print('Loading pretrain LLaVA projector...')
        #     if 'mpt' in model_name.lower():
        #         if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
        #             shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
        #         tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
        #         cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        #         model = LlavaMptForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
        #     else:
        #         from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM 
        #         print('Loading LLaVA from base model...')
        #         tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
        #         cfg_pretrained = AutoConfig.from_pretrained(model_path)
        #         model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
        #     print(f'Loading MLP weight...')
        #     mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
        #     mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
        #     model.load_state_dict(mm_projector_weights, strict= False)
        #     print('MLP is loaded...')
        # else:
        #     raise ValueError(f"Invalid model name {model_name} or model_base {model_base}. Please check the model name and base model.")

    else:
        # Load language model
        print("Pure language model loading...")
        if os.path.exists(model_path+'/adapter_model.safetensors'):
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    # if tokenizer.pad_token is None:
    #     print("Setting pad_token to eos_token")
    #     tokenizer.pad_token = tokenizer.eos_token


    modility_processor = None
    if 'llava' in model_name.lower() or 'mela' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        neuro_tower = model.get_neuro_tower()
        
        if neuro_tower is not None or not neuro_tower.is_loaded:
            neuro_tower.load_model()
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=device_map)
        if device_map != 'auto':
            vision_tower.to(device=device_map, dtype=torch.float16)

    modility_processor = vision_tower.image_processor if 'llava' in model_name.lower() else neuro_tower.eegProcessor
    # modility_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048
        
    return tokenizer, model, modility_processor, context_len
