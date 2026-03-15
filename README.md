# WaveMind: Towards a Conversational EEG Foundation Model Aligned to Textual and Visual Modalities 
<div align="center">

</div>



<!-- ##  Note
We are still cleaning code and update data. Stay Tune!

✔ Init Repo

✔ WaveMind Instruct

✔ WaveMind Bench

✔ Preprocess-code

✔ Phase 1 Code

✔ WaveMind-Pipeline -->




## ⚡ Introduction
Hello! Welcome to the repository for WaveMind!
<!-- [WaveMind](https://arxiv.org/pdf/2412.18925) -->

Electroencephalography (EEG) interpretation using multimodal large language models (MLLMs) offers a novel approach to analyzing brain signals. However, the inherent complexity of brain activity, encompassing both cognitive functions representing subjective consciousness and non-cognitive processes associated with homeostasis, creates distinct supervisory modalities during training. This divergence hinders the generalization capability of existing EEG-MLLM models across tasks and impedes fluent natural language interaction. To address these limitations, we introduce WaveMind, the first LLM framework specifically designed to interpret EEG data by projecting diverse neural signals into a shared semantic space. We synthesize the WaveMind-Instruct dataset, comprising 362k instructions, with GPT assistance. WaveMind achieves remarkable performance on four downstream classification tasks and supports fluent, open-ended dialogue about brain activity. Ablation studies underscore significant synergies between supervision modility and across tasks, demonstrating the importance of comprehensive modeling of brain signals for developing general-purpose EEG interpretation systems.





<div align=center>
<img src="asset/Architecture.svg"  width = "90%" alt="Architecture" align=center/>
</div>



We open-sourced our models, data, and code here.

<!-- ## 👨‍⚕️ Model Deployment
- **Model Access**

|                      | Backbone     | Supported Languages | Link                                                                  |
| -------------------- | ------------ | ----- | --------------------------------------------------------------------- |
| **WaveMind-Vicuna**  | Vicuna-1.5 7B  | English    | [HF Link](https://huggingface.co/FreedomIntelligence/HuatuoGPT-o1-8B) |
| **WaveMind-Qwen** | Qwen2.5-Insturct 7B | English    | [HF Link](https://huggingface.co/FreedomIntelligence/HuatuoGPT-o1-70B) | -->



- **Configuration Environment**

```bash
conda activate <your-local_virtual_env>
bash Setup_Env.sh <your_project_absolute_path>
```
Example:
```bash
conda activate WaveMind
bash Setup_Env.sh /xxxx/xxxx/WaveMind
```
This bash script will auto-install the required packages and dependencies for the project. And write 'WaveMind_ROOT_PATH_' in ~/.bashrc which can MUST be used in the project.

- **Demostration**

1. We encourage the use of our pre-processing pipeline below, but you can also try using your own EEG data.
2. Below is the example code. We also provide [Predict Script](/WaveMind/EEGLLM/Predict.py) for inference.

```python
import os
from EEGLLM.Predict import WaveMind_inference, select_data_for_test,get_model_base_from_path,get_model_name_from_path
from llava.model.builder import load_pretrained_model



model_path = "your local model path"
question='Hi doctor, please help me to see any event inside my EEG.'

model_base=get_model_base_from_path(model_path)
tokenizer, model, modility_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=model_base,
    model_name=get_model_name_from_path(model_path),
)
# Here we use the preprocessed data after pipeline, you may change to any EEG clip as long as it is 32 channel, 512 data-point.

# Option 1: Use preprocessed data (Recommand)
# Dataset can be one of TUEV, TUAB, ImageNet-EEG,THING-EEG, SEED
ds=select_data_for_test(dataset='TUEV')
sample=ds[0]
category,eeg_tensor=sample['text'].decode(),sample['eeg_data']

# Option 2: Use your own data
# We did not test on other datasets, therefore use with caution
# better finetune in your dataset, use our provided train script for your reference

raw=mne.io.read_raw('xxx.edf', preload=True)
electrode_list=raw.ch_names
eeg_tensor=modility_processor.preprocess(raw.get_data(),fs=raw.info['sfreq'],l_freq=None,h_freq=None,electrode_list=electrode_list,verbose=True,confirm_cut_1s='random')

# Perform Inference
WaveMind_inference(model=model,tokenizer=tokenizer,question=question,processed_eeg=eeg_tensor,RAG=True,verbose=True,model_base=model_base)
```


## 📚 Data Engineering

Unfortunately, due to privacy and licensing reasons, we are unable to publicly disclose the preprocessed dataset. However we provide a preprocessing process and code, which can be used to preprocess the data yourself.

- **Raw Data Access**

For some datasets that are difficult to download, we provide convenient download scripts.

| Data               | Description                                                                                   | Download Script | Folder Name | Link                                                                                           |
|--------------------|-----------------------------------------------------------------------------------------------|-----------------|-----------|------------------------------------------------------------------------------------------------|
| TUAB               | A corpus of EEGs that have been annotated as normal or abnormal.                              |  [Link](data/ImageNetEEG/download.py)               |      edf     | [Link](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-verifiable-problem)      |
| TUEV               | The subset of TUEG that contains annotations of EEG segments as one of six classes.           |    [Link](data/TUEV/download.py)             |   edf        | [Link](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT)           |
| ImageNet-EEG       | This dataset includes EEG data from 6 subjects when looking 40 categories image from ImageNet.|     [Link](data/ImageNetEEG/download.py)            |  Refer to  File Tree       | [Link](https://tinyurl.com/eeg-visual-classification)                                         |
| THING-EEG          | The dataset includes the EEG data of 10 subjects when viewed corresponding images.            |       N.A          |   Data        | [Link](https://huggingface.co/datasets/LidongYang/EEG_Image_decode)                            |
| SEED               | The dataset which provides EEG data and corresponding emotional states.                       |           N.A     |   Preprocessed_EEG        | [Link](https://bcmi.sjtu.edu.cn/home/seed/seed.html)                                          |

- **Data Preprocessing**

Please refer to [here](data/README.md) for details.

### Automated Preprocessing Script

We provide an automated bash script to process all EEG datasets sequentially:

```bash
# Make the script executable
chmod +x data/preprocess_wavemind_dataset.sh

# Process all datasets
./data/preprocess_wavemind_dataset.sh

# Process specific dataset only
./data/preprocess_wavemind_dataset.sh SEED

# Show available datasets
./data/preprocess_wavemind_dataset.sh --list

# Skip dependency checks
./data/preprocess_wavemind_dataset.sh --skip-check
```







## 🚀 Training





 **Stage 1: Dual-Representation Alignment**

**IMPORTANT**: The `--config-name` parameter is now mandatory.

**Available config presets** (in [EEG_Encoder/examples/](EEG_Encoder/examples/)):
- `train_atms.yaml`: Quick-start for ATMSmodify training (recommended)
- `eval_sd.yaml`: Subject-Dependent evaluation
- `eval_si.yaml`: Subject-Independent evaluation
- `advanced_shm.yaml`: Advanced shared memory configuration

**Basic training with ATMSmodify**:
```bash
python EEG_Encoder/run_CLIPtraining.py --config-name=train_atms
```

**Training with custom overrides**:
```bash
python EEG_Encoder/run_CLIPtraining.py --config-name=base \
    experiment.models=[ATMSmodify] \
    experiment.gpu_number=[0] \
    training.DEFAULT_EPOCHS=30 \
    experiment.datasets=[ImageNetEEG]
```

**Evaluation (Subject Dependent)**:
```bash
python EEG_Encoder/run_CLIPtraining.py --config-name=eval_sd \
    advanced.model_checkpoint_name=/path/to/checkpoint.pth
```

**Evaluation (Subject Independent)**:
```bash
python EEG_Encoder/run_CLIPtraining.py --config-name=eval_si \
    advanced.model_checkpoint_name=/path/to/checkpoint.pth
```

**Available Models**: MLP, ATMS, ShallowFBCSPNet, channelNet, NICE, ATMSmodify (primary), EEGITNet, CBraMod, NeuroLM-B, NeuroLM-L

**Encoder Checkpoint**: The trained ATMM(ATMSmodify) checkpoint can be found in [EEG_Encoder/Resource/Checkpoint/ALL/](EEG_Encoder/Resource/Checkpoint/ALL/)

**Full configuration options**: See [EEG_Encoder/examples/base.yaml](EEG_Encoder/examples/base.yaml) for all available parameters.


**Stage 2: Cold Start Training**

We use LLAVA_pretain to enable EEG-MLLM to recognize CLIP space, before EEG Instruction tuning.

1. Download from [LLaVA-Pretrain](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain), unfold and put it under EEGLLM/LLaVA/playground/data/LLaVA-Pretrain.
2. Please change the corresponding options and start training on the script below.



```bash
bash ./EEGLLM/examples/stage2_pretrain/pretrain.sh
```
**Stage 3: EEG Instruction Tuning**

Please change the corresponding options in the script to start training.

<div class="alert alert-info" style="padding: 15px; border-left: 5px solid #17a2b8; background: #f8f9fa; border-radius: 4px;">
⚠️ WaveMind_Instruct Not Ready Yet. Stay Tune! You may use your SFT data instead.<br>
</div>
<br>

```bash
bash ./EEGLLM/examples/stage3_finetune/finetune_lora_eeg.sh
```

## 📖 WaveMind Bench Construction


<div class="alert alert-info" style="padding: 15px; border-left: 5px solid #17a2b8; background: #f8f9fa; border-radius: 4px;">
⚠️ Make sure you have completed the preprocessing process with generated data_label.h5<br>
</div>

Run:

```bash
bash ./Data_Engineering/Script/Test_data/construct_WaveMind.sh
```








## 🧐 WaveMind Bench Evaluation


Run evaluation script to evaluate the WaveMind over WaveMind Bench.
```bash
CUDA_VISIBLE_DEVICES=0 python ./EEGLLM/Evaluation/Evaluation_Classification.py --model_path /path/to/model
```
Please refer to script for more setting details.

## 📂 File Tree
```bash
/WaveMind_ROOT_PATH_
├── data
│   ├── ImageNetEEG
│   │   ├── eeg_signals_raw_with_mean_std.pth  -> raw_file need to be download
│   │   ├── Image -> raw_file need to be download
│   │   └── ....
│   ├── preUtils.py
│   ├── SEED
│   │   ├── Preprocessed_EEG  -> raw_file need to be download
│   │   └── ....
│   ├── THING-EEG
│   │   ├── Data   -> raw_file need to be download
│   │   ├── data_config.json
│   │   ├── download.py
│   │   └── ....
│   ├── Total
│   │   ├── CLIP_groundTruth   -> generated_file
│   │   ├── data_label.h5      -> generated_file
│   │   ├── dataset_weights.pth  -> auto_generated_file when training
│   │   └── ....
│   ├── TUAB
│   │   ├── download.exp  -> raw_file need to be download
│   │   ├── edf   -> raw_file need to be download
│   │   ├── save  -> cache dir
│   │   └── ....
│   ├── TUEV
│   │   ├── download.exp  
│   │   ├── edf    -> raw_file need to be download
│   │   ├── eegs.npz  -> cache file
│   │   └── ....
│   └── ....
├── EEG_Encoder
│   ├── Resource
│   │   ├── Checkpoint   -> EEG Encoder checkpoint
│   │   └── ....
│   ├── run_CLIPtraining.py -> Script to Train EEG Encoder
│   └── ....
├── EEGLLM
│   ├── Evaluation   -> Evaluation on WaveMind_Bench
│   └── ....
├── Data_Engineering
│   ├── data
│   │   ├── EEG_data    ->  WaveMind_Bench EEG data location
│   │   ├── Test_data   ->  WaveMind_Bench MCQ data location
│   │   └── ...
│   ├── Script
│   │   └── Test_data   -> Scriptd to WaveMind_Bench data generation
│   └── ....
└── ....
```







<!-- ## 📖 Citation
```
@misc{chen2024huatuogpto1medicalcomplexreasoning,
      title={HuatuoGPT-o1, Towards Medical Complex Reasoning with LLMs}, 
      author={Junying Chen and Zhenyang Cai and Ke Ji and Xidong Wang and Wanlong Liu and Rongsheng Wang and Jianye Hou and Benyou Wang},
      year={2024},
      eprint={2412.18925},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.18925}, 
}
``` -->

## ✨ Acknowledgement
1. Thanks to L. Dong et al. for their contribution in EEG feature alignment (in cognitive activity alignment), we refer to their work: [EEG_Image_decode](https://github.com/dongyangli-del/EEG_Image_decode)
2. Thanks to [torcheeg](https://github.com/torcheeg/torcheeg) and [Pyhealth](https://github.com/sunlabuiuc/PyHealth) for providing the preprocessing tool.
3. This README Template comes from [HuatuoGPT-o1](https://github.com/FreedomIntelligence/HuatuoGPT-o1/blob/main/README.md).
4. Thanks to myself and co-authors effort.


## 📄 Licenses
This project is released under the [Apache License 2.0](./LICENSE). We welcome the use and extension of our work by the community.




## More...
<div style="text-align: center;">
Thank you for your attention to our project, any question feel free to open the issue!

</div>




<!-- ## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=FreedomIntelligence/HuatuoGPT-o1&type=Date)](https://star-history.com/#FreedomIntelligence/HuatuoGPT-o1&Date) -->
