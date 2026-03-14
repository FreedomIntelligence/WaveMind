#!/bin/bash
#General Settings
project_root_path=${WaveMind_ROOT_PATH_}
if [ -z "$WaveMind_ROOT_PATH_" ]; then
    echo "WaveMind_ROOT_PATH_ is not set"
    exit 1
fi
prompt_root_folder=$project_root_path/Data_Engineering
cd $project_root_path || exit



# I/O
output_dir=$project_root_path/EEGLLM/LLaVA/checkpoints/LLAVA/LLAVA_LLAMA/llava_LLAMA_7b_pretrain


# Backbone
model_name_or_path=lmsys/vicuna-7b-v1.5



# Data
# data_path=$project_root_path/EEGLLM/LLaVA/playground/data/llava_v1_5_mix665k.json
data_path=$project_root_path/EEGLLM/LLaVA/playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json
image_folder=$project_root_path/EEGLLM/LLaVA/playground/data/LLaVA-Pretrain



# Neuro Tower
neuro_tower=ATMSmodify
neuro_tower_checkpoint_dir_path=$project_root_path/EEG_Encoder/Resource/Checkpoint/ALL
freeze_neuro_tower=True
random_neuro_tower=False

# adapter
tune_mm_mlp_adapter=True




deepspeed --master_port=$((RANDOM % 20 + 29500))  $project_root_path/EEGLLM/LLaVA/llava/train/train_mem.py \
    --lora_enable False --lora_r 8 --lora_alpha 128 --mm_projector_lr 5e-5 \
    --deepspeed $project_root_path/EEGLLM/LLaVA/scripts/zero2.json \
    --model_name_or_path $model_name_or_path \
    --version v1 \
    --data_path $data_path \
    --train_data_mode llava \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --freeze_neuro_tower $freeze_neuro_tower \
    --neuro_tower $neuro_tower \
    --neuro_tower_checkpoint_dir_path $neuro_tower_checkpoint_dir_path \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir $output_dir \
    --num_train_epochs 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --eval_step=50 \
    --eval_strategy "no" \
    --save_steps 150 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine_with_restarts" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to none \
    --mm_vision_select_feature cls_linear \
    --mm_hidden_size=768 \
    --per_device_train_batch_size 4 \
    --dataloader_num_workers 0 \
    --tune_mm_mlp_adapter $tune_mm_mlp_adapter \
    --random_neuro_tower $random_neuro_tower \
    --image_folder $image_folder 


