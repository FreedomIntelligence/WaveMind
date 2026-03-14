# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
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


import glob
import sys

import deepspeed

from llava.utils import rank0_print
from llava.dataLoader import make_supervised_data_module
import llava
from peft import LoraConfig, get_peft_model
from peft import PeftModel

import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

import torch

import transformers
import tokenizers


from llava.train.llava_trainer import LLaVATrainer

from llava import conversation as conversation_lib
from llava.model import *

from llava.utils import *



local_rank = int(os.environ.get("LOCAL_RANK", 0))










def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    excluded_keywords = ['vision_tower', 'mm_projector', 'vision_resampler','neuro_tower']

    for name, module in model.named_modules():
        if any(kw in name for kw in excluded_keywords):
            continue

        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[-1])

    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
        print(f"Saving model checkpoint to {output_dir}")


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg






def train(attn_implementation=None):
    # import multiprocessing
    # multiprocessing.set_start_method('spawn')

    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))



    bnb_model_from_pretrained_args = {}




    if model_args.mm_vision_select_feature!= 'patch':
        print("mm_vision_select_feature is not patch, set mm_hidden_size to 768")
        bnb_model_from_pretrained_args.update(dict(
            mm_vision_select_feature=model_args.mm_vision_select_feature,
            mm_hidden_size=model_args.mm_hidden_size
        ))


    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))
        
    customized_kwargs = dict()
    customized_kwargs.update(bnb_model_from_pretrained_args)
    from transformers import AutoConfig
    print(model_args.model_name_or_path)
    cfg_pretrained = AutoConfig.from_pretrained(model_args.model_name_or_path)
    customized_kwargs["config"] = cfg_pretrained
        
    if model_args.vision_tower is not None:
        if "mixtral" in model_args.model_name_or_path.lower():
            rank0_print("Loading Mixtral series model")
            model = LlavaMixtralForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=training_args.attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                low_cpu_mem_usage=False,
                **customized_kwargs,
            )
            from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

            deepspeed.utils.set_z3_leaf_modules(model, [MixtralSparseMoeBlock])
        elif "mistral" in model_args.model_name_or_path.lower() or "zephyr" in model_args.model_name_or_path.lower():
            rank0_print("Loading Mistral series model")
            
        
            
            
            model = LlavaMistralForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=training_args.attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                low_cpu_mem_usage=False,
                **customized_kwargs,
            )
        elif (
            "wizardlm-2" in model_args.model_name_or_path.lower()
            or "vicuna" in model_args.model_name_or_path.lower()
            or "llama" in model_args.model_name_or_path.lower()
            or "yi" in model_args.model_name_or_path.lower()
            or "nous-hermes" in model_args.model_name_or_path.lower()
            and "wizard-2" in model_args.model_name_or_path.lower()
        ):
            rank0_print("Loading LLAMA series model")
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=training_args.attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                low_cpu_mem_usage=False,
                **customized_kwargs,
            )
        elif "qwen" in model_args.model_name_or_path.lower():
            
            if "moe" in model_args.model_name_or_path.lower() or "A14B" in model_args.model_name_or_path:
                rank0_print("Loading Qwen MOE series model")
                model = LlavaQwenMoeForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    attn_implementation=training_args.attn_implementation,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    low_cpu_mem_usage=False,
                    **customized_kwargs,
                )
                from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock

                deepspeed.utils.set_z3_leaf_modules(model, [Qwen2MoeSparseMoeBlock])
            else:
                rank0_print("Loading Qwen 2.5 series model")
                model = LlavaQwenForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    attn_implementation=attn_implementation,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    low_cpu_mem_usage=False,
                    **customized_kwargs,
                )
        elif "gemma" in model_args.model_name_or_path.lower():
            rank0_print("Loading Gemma series model")
            model = LlavaGemmaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=training_args.attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                low_cpu_mem_usage=False,
                **customized_kwargs,
            )
        else:
            raise ValueError(f"Unknown model class {model_args}")
            
            
    else:
        rank0_print("Loading Llama series model, no vision tower")
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **customized_kwargs
        )
    
        
 

    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
    if training_args.gradient_checkpointing:
        try:
            model.enable_input_require_grads()
        except:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)




    origin_load_path=training_args.origin_load_path if training_args.origin_load_path is not None else training_args.output_dir

    if training_args.lora_enable:
        print("lora_enable")
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")


        if os.path.exists(os.path.join(origin_load_path,'adapter_model.safetensors')) or os.path.exists(os.path.join(origin_load_path,'adapter_model.bin')):
            print("Found LORA (final checkpoint) adapter, loading from there, trainable is True")
            model = PeftModel.from_pretrained(model, origin_load_path, device_map="auto", is_trainable=True)
        elif list(pathlib.Path(origin_load_path).glob("checkpoint-*")):
            last_checkpoing_dir=list(pathlib.Path(origin_load_path).glob("checkpoint-*"))
            last_checkpoing_dir=sorted(last_checkpoing_dir)[-1]
            if os.path.exists(os.path.join(last_checkpoing_dir,'adapter_model.safetensors')) or os.path.exists(os.path.join(last_checkpoing_dir,'adapter_model.bin')):
                print("Found LORA adapter (middle checkpoint) in origin_load_path, loading adapters from there,trainable is True")
                model = PeftModel.from_pretrained(model, last_checkpoing_dir, device_map="auto",is_trainable=True)
            else:
                print("No LORA adapter (middle checkpoint) found, initializing a new one")
                model = get_peft_model(model, lora_config)
        else:
            print("init a new LORA adapter")
            model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        # want to load checkpoint but keep lora off(possible tune MLP only)
        if os.path.exists(os.path.join(origin_load_path,'adapter_model.safetensors')) or os.path.exists(os.path.join(origin_load_path,'adapter_model.bin')):
            print("Found LORA (final checkpoint) adapter, loading from there, trainable is False")
            model = PeftModel.from_pretrained(model, origin_load_path, device_map="auto", is_trainable=False)
        if list(pathlib.Path(origin_load_path).glob("checkpoint-*")):
            rank0_print("Adding LoRA adapters...")
            last_checkpoing_dir=list(pathlib.Path(origin_load_path).glob("checkpoint-*"))
            last_checkpoing_dir=sorted(last_checkpoing_dir)[-1]
            print("Found checkpoint in origin_load_path, loading adapters from there,trainable is False")
            model = PeftModel.from_pretrained(model, last_checkpoing_dir, device_map="auto",is_trainable=False)
        else:
            print("No LORA adapter,probabaly alignment stage")


    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
        )

    
    if 'vicuna' in model_args.model_name_or_path.lower():
        model_name = 'vicuna_v1'
    elif 'qwen2.5' in model_args.model_name_or_path.lower():
        model_name = 'qwen_2_5'
    elif 'llama-3.1' in model_args.model_name_or_path.lower():
        model_name = 'llava_llama_3'
    elif 'llama-2' in model_args.model_name_or_path.lower():
        model_name = 'llama_2'
        tokenizer.pad_token = tokenizer.eos_token
    elif 'mistral' in model_args.model_name_or_path.lower():
        model_name='mistral_instruct'
        tokenizer.pad_token = tokenizer.eos_token
    
    
        
    else:
        model_name = model_args.model_name_or_path.split('/')[-1]
    if model_name in conversation_lib.conv_templates:
        print(f'select conv_templates of {model_name}')
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_name]
    else:
        raise ValueError(f"Unknown model name {model_name}")



    
    
    ####################################### load MLP weight
    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )

        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length
        ####################################### Tune only MLP
        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            print("Tune only MLP")
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True
        else:
            print("Tune MLP and Backbone")

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            print("freeze MLP")
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)
        ####################################### Tune only MLP

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.config.neuro_tower = model_args.neuro_tower
        model.config.freeze_neuro_tower=model_args.freeze_neuro_tower
        model.config.random_neuro_tower=model_args.random_neuro_tower
        
        
        
        
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)




    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)


    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args,model=model)




    for param in model.parameters(): param.data = param.data.contiguous()

    trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)



    
    
    
    # Check if the model is trained and ready for RAG
    if data_args.train_data_mode =='mela':

        tmp_path = os.environ['WaveMind_ROOT_PATH_'] + "/data/Total/test_FeatureExtractor.npz"
        assert os.path.exists(tmp_path), f"{tmp_path} does not exist, please check your WaveMind_ROOT_PATH_ env variable"
        check_neuroTower_pass_test(model.get_neuro_tower(),tmp_path)
        model.DBtool.check_model_ok_for_RAG(model.get_neuro_tower())

    
    
    final_check_model(model)
    model.config.save_pretrained(training_args.output_dir)
    # Find all checkpoint directories and resueme from the last one
    checkpoints = sorted(list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")), key=lambda x: int(x.name.split('-')[-1]))
    if checkpoints:
        print("Find checkpoint, resuming training from the last checkpoint")
        last_checkpoint = checkpoints[-1]
        model_path = str(last_checkpoint)
        
        if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
            print(f"Loading non-lora trainables from last checkpoint at {os.path.join(model_path, 'non_lora_trainables.bin')}...")
            non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)
        else:
            print(f"No non-lora trainables found in last checkpoint at {model_path}, still resuming training...")
        trainer.train(resume_from_checkpoint=True)
    else:
        print("No checkpoint found in output_dir, clean start")
        trainer.train()
        
        
    trainer.save_state()

    model.config.use_cache = True

    # if training_args.lora_enable or isinstance(model, PeftModel):
    #     state_dict = get_peft_state_maybe_zero_3(
    #         model.named_parameters(), training_args.lora_bias
    #     )
    #     non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
    #         model.named_parameters()
    #     )
    #     if training_args.local_rank == 0 or training_args.local_rank == -1:
    #         model.config.save_pretrained(training_args.output_dir)
    #         model.save_pretrained(training_args.output_dir, state_dict=state_dict)
    #         torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    # else:
    #     safe_save_model_for_hf_trainer(trainer=trainer,
    #                                    output_dir=training_args.output_dir)
    
    

    non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
        model.named_parameters()
    )
    if (training_args.local_rank == 0 or training_args.local_rank == -1) and (training_args.lora_enable or isinstance(model, PeftModel)):
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        model.save_pretrained(training_args.output_dir, state_dict=state_dict)
    torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    print(f"sucessfully save to {os.path.join(training_args.output_dir, 'non_lora_trainables.bin') }")

    assert any('mm_projector' in item for item in non_lora_state_dict.keys()), "mm_projector should be in non_lora_trainables.bin"


    remove_global_step(training_args.output_dir)


if __name__ == "__main__":
    train()
