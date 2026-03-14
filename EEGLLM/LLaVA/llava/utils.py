import copy
from dataclasses import dataclass, field
import datetime
import logging
import logging.handlers
import warnings
import transformers
import os
import sys
from llava.mm_utils import tokenizer_image_token
import requests
import torch
from llava import conversation as conversation_lib
from typing import Dict, Optional, Sequence, List
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.constants import LOGDIR
from packaging import version
import tokenizers
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')
import random,json
import numpy as np
from llava.conversation import conv_templates, SeparatorStyle
from transformers import StoppingCriteria, TextStreamer


server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
moderation_msg = "YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES. PLEASE TRY AGAIN."

handler = None

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    root_folder:Optional[str] = field(default=None)
    train_data_mode: Optional[str] = field(default="llava")
    train_pure_text:bool = field(default=False)
    image_aspect_ratio: str = 'square'
    filter_keywords: Optional[str] = field(default=None)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None),
    neuro_tower: str = 'ATMSmodify',
    neuro_tower_checkpoint_dir_path: str = field(default=None),
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="cls_linear")
    mm_hidden_size: Optional[int] = field(default=768)
    freeze_neuro_tower: bool = field(default=True)
    random_neuro_tower: bool = field(default=False)
    
    

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    origin_load_path: str = None
    attn_implementation: str = field(default="sdpa")









def build_logger(logger_name, logger_filename):
    global handler

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the format of root handlers
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Add a file handler for all loggers
    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(
            filename, when='D', utc=True, encoding='UTF-8')
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ''
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == '\n':
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != '':
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ''


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def violates_moderation(text):
    """
    Check whether the text violates OpenAI moderation API.
    """
    url = "https://api.openai.com/v1/moderations"
    headers = {"Content-Type": "application/json",
               "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]}
    text = text.replace("\n", "")
    data = "{" + '"input": ' + f'"{text}"' + "}"
    data = data.encode("utf-8")
    try:
        ret = requests.post(url, headers=headers, data=data, timeout=5)
        flagged = ret.json()["results"][0]["flagged"]
    except requests.exceptions.RequestException as e:
        flagged = False
    except KeyError as e:
        flagged = False

    return flagged


def pretty_print_semaphore(semaphore):
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"



def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = ((DEFAULT_IMAGE_TOKEN + '\n') + sentence['value']) if random.random() > 0.5 else (sentence['value']+ (DEFAULT_IMAGE_TOKEN + '\n'))
                sentence['value'] = sentence['value'].replace('\n\n', '\n').strip(' ')
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources





def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    
    def find_subsequence(sequence, subseq):
        n = len(sequence)
        m = len(subseq)
        if m == 0:
            return None
        
        for i in range(n - m + 1):
            match = True
            for j in range(m):
                if sequence[i + j] != subseq[j]:
                    match = False
                    break
            if match:
                return i
        return None



    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # conversations=['<s>[INST]<image>Hello[/INST]Nice</s>']
    # conversations=['<s>[INST]<image>Hello?[/INST]Hello</s>[INST]Helo![/INST]Helo!</s>']
    # conversations=['[INST]<image>Hello?[/INST]Hello</s>[INST]Hello![/INST]Hello!</s>[INST]Hello![/INST]Hello!</s></s></s></s></s>']


    # print(f"conversation: {conversations}")


    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids


    
    if input_ids.ndim!=2:
        raise ValueError("Not support")


    if input_ids[0][0] == input_ids[0][1]: 
        input_ids = input_ids[:, 1:] 



    targets = input_ids.clone()
    
    

    
    
    # print(f"conversation{conversations}")
    
    
    

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2


    beg ='[INST]'
    sep = "[/INST]"
    beg_ids = tokenizer.convert_tokens_to_ids(beg)

    
    is_llama2_or_mistral=True if beg_ids==0 else False



    eos_id= tokenizer.eos_token_id
    
    
    
    targets=targets[0]
    input_ids=input_ids[0]
    
    
    targets[0]=-100
    
    
    # --------------------------
    
    if is_llama2_or_mistral:
        dtype_data=targets.dtype
        input_ids,target= input_ids.tolist(), targets.tolist()
        
        inst_tokens = tokenizer.encode("[INST]", add_special_tokens=False)
        end_inst_tokens = tokenizer.encode("[/INST]", add_special_tokens=False)
        end_inst_tokens.pop(0)
        
        # print(f"inst{inst_tokens}")
        # print(f"end_inst{end_inst_tokens}")
        
        idx = 0
        while idx < len(input_ids):
            # print(f"{input_ids[idx:idx+len(inst_tokens)]} {inst_tokens}")
            if input_ids[idx:idx+len(inst_tokens)] == inst_tokens:
                # print('inside')
                start_pos = idx
                # print(f"start_pos: {start_pos}")
                search_space = input_ids[idx+len(inst_tokens):]
                # print(search_space)
                end_pos = find_subsequence(search_space, end_inst_tokens)
                # print(start_pos, end_pos)
                if end_pos is not None:
                    end_pos = idx + len(inst_tokens) + end_pos
                    end_index = end_pos + len(end_inst_tokens)
                    targets[start_pos:end_index] = -100
                    idx = end_index 
                else:
                    idx += 1 
            else:
                idx += 1
        input_ids,target= torch.tensor(input_ids, dtype=dtype_data), torch.tensor(target, dtype=dtype_data)
    else:
        beg = '[INST]'
        sep = "[/INST]"
        beg_id = tokenizer.convert_tokens_to_ids(beg)
        sep_id = tokenizer.convert_tokens_to_ids(sep)
        
        idx = 0
        while idx < len(input_ids):
            if input_ids[idx] == beg_id:
                start_pos = idx
                end_pos = None
                for j in range(idx + 1, len(input_ids)):
                    if input_ids[j] == sep_id:
                        end_pos = j
                        break
                if end_pos is not None:
                    targets[start_pos:end_pos+1] = -100
                    idx = end_pos
            idx += 1
            
    
    
    
    
    # --------------------------
    
    
    # while idx < len(input_ids):
    #     if input_ids[idx] == beg_id:
    #         start_pos = idx           
    #         end_pos = None
    #         for j in range(idx + 1, len(input_ids)):
    #             if input_ids[j] == sep_id:
    #                 end_pos = j
    #                 break
        
    #         if end_pos is not None:
    #             targets[start_pos:end_pos+1] = -100
    #             idx = end_pos

    #     idx += 1

    for idx in range(len(input_ids)-1,-1,-1):
            
        if input_ids[idx] != eos_id:
            break
        else:
            targets[idx] = -100

    targets[idx+1]=input_ids[idx+1]
    
    
    
    targets=targets.unsqueeze(0)
    input_ids= input_ids.unsqueeze(0)
    
    
    
    


    # print(input_ids)
    # print(targets)
    # print(tokenizer.decode([token for token in targets.flatten().tolist() if token != -100]))
    
    # sys.exit()
    

    
    return dict(
        input_ids=input_ids,
        labels=targets,
    )
    
    






def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    
    

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())


    # print(f"conversation{conversations}")

    
    
    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO
    
    

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
                
    # print(tokenizer.decode([token for token in target.tolist() if token != -100 else 0]))

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)



def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    # roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}
    roles = {"human": "user", "gpt": "assistant"}

    # Add image tokens to tokenizer as a special tokens
    # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
    tokenizer = copy.deepcopy(tokenizer)
    # When there is actually an image, we add the image tokens as a special token
    if has_image:
        tokenizer.add_tokens(["<image>"], special_tokens=True)

    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    # additional_special_tokens_ids= tokenizer.additional_special_tokens_ids
    
    
    # print(tokenizer.add_special_tokens)
    # sys.exit()
    
    unmask_tokens_idx =  []
    # unmask_tokens_idx.extend(additional_special_tokens_ids)
    # nl_tokens = tokenizer("\n").input_ids

    # Reset Qwen chat templates so that it won't include system message every time we apply
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    # _system = tokenizer("system").input_ids + nl_tokens
    # _user = tokenizer("user").input_ids + nl_tokens
    # _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            # Make sure llava data can load
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)
            
            conv = [{"role" : role, "content" : content}]
            encode_id = tokenizer.apply_chat_template(conv,tokenize=False)
            encode_id = tokenizer.encode(encode_id)
            
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id
        

                    
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
        input_ids.append(input_id)
        targets.append(target)


    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    
    # print(input_ids)
    # print(targets)
    # print("---------------")
    # sys.exit(0)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )

def preprocess_llama3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    max_len=2048,
    system_message: str = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.",
) -> Dict:
    # roles = {"human": "<|start_header_id|>user<|end_header_id|>", "gpt": "<|start_header_id|>assistant<|end_header_id|>"}
    roles = {"human": "user", "gpt": "assistant"}

    # Add image tokens to tokenizer as a special tokens
    # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
    tokenizer = copy.deepcopy(tokenizer)
    # When there is actually an image, we add the image tokens as a special token
    if has_image:
        tokenizer.add_tokens(["<image>"], special_tokens=True)
    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    bos_token_id = tokenizer.convert_tokens_to_ids("<|begin_of_text|>")
    start_header_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    end_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

    unmask_tokens = ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "\n\n"]
    unmask_tokens_idx = [tokenizer.convert_tokens_to_ids(tok) for tok in unmask_tokens]

    # After update, calling tokenizer of llama3 will
    # auto add bos id for the tokens. ヽ(｀⌒´)ﾉ
    def safe_tokenizer_llama3(text):
        input_ids = tokenizer(text).input_ids
        if input_ids[0] == bos_token_id:
            input_ids = input_ids[1:]
        return input_ids

    nl_tokens = tokenizer.convert_tokens_to_ids("\n\n")
    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            # Make sure llava data can load
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)
            
            conv = [{"role" : role, "content" : content}]
            # First is bos token we don't need here
            encode_id = tokenizer.apply_chat_template(conv)[1:]
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id
        

                    
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )
    
    
def preprocess_gemma(sources: List[List[Dict[str, str]]], tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv: conversation_lib.Conversation = conversation_lib.default_conversation.copy()
    roles: Dict[str, str] = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations: List[str] = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source: List[Dict[str, str]] = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role: str = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids: torch.Tensor = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    else:
        input_ids: torch.Tensor = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets: torch.Tensor = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.GEMMA

    # Mask target
    sep: str = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len: int = int(target.ne(tokenizer.pad_token_id).sum())

        rounds: List[str] = conversation.split(conv.sep)
        re_rounds = []
        for conv_idx in range(0, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx : conv_idx + 2]))

        cur_len = 1  # Ignore <bos>
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep  # Re-append sep because split on this
            # Now "".join(parts)==rou

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer)) - 1  # Ignore <bos>
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1  # Ignore <bos>
            else:
                round_len = len(tokenizer(rou).input_ids) - 1  # Ignore <bos>
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1  # Ignore <bos>

            round_len += 2  # sep: <end_of_turn>\n takes 2 tokens
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len

        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"warning: tokenization mismatch: {cur_len} vs. {total_len}." f" (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    
    
    
    
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    # Vicuna
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "qwen":
        return preprocess_qwen(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "gemma":
        return preprocess_gemma(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "llama_v3":
        return preprocess_llama3(sources, tokenizer, has_image=has_image)
    
    
    raise ValueError(
        f"No support"
    )
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]


    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


def rank0_print(*args):
    local_rank=int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print(*args)



def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation




def final_check_model(model):
    from peft import PeftModel
    # rank0_print(model)

    if isinstance(model, PeftModel):
        print("PeftModel in LORA")
        base_model=model.base_model.model.model.layers
        if hasattr(model.base_model.model.model, "mm_projector"):
            vision_tower=model.base_model.model.model.vision_tower
            neuro_tower = model.base_model.model.model.neuro_tower
            mm_projector = model.base_model.model.model.mm_projector

    else:
        print("Full training")
        base_model=model.model.layers
        if hasattr(model.model, "mm_projector"):
            vision_tower = model.model.vision_tower
            mm_projector = model.model.mm_projector
            neuro_tower = model.model.neuro_tower
        else:
            warnings.warn("Fail to get model")

    assert neuro_tower is not None, "Neuro tower should not be None"


    base_params = list(base_model.parameters())
    base_total = sum(p.numel() for p in base_params)
    base_trainable = sum(p.numel() for p in base_params if p.requires_grad)


    # Check MM Projector parameters
    if mm_projector is not None:
        mm_params = list(mm_projector.parameters())
        mm_total = sum(p.numel() for p in mm_params)
        mm_active = sum(p.numel() for p in mm_params if p.requires_grad)
        mm_frozen = mm_total - mm_active
        if mm_total == 0:
            raise RuntimeError("MM Projector has no parameters to check")
        if mm_active != mm_total:
            raise RuntimeError(f"MM Projector has {mm_frozen} frozen parameters (expected all {mm_total} active)")
        vision_params = list(vision_tower.parameters())
        vision_total = sum(p.numel() for p in vision_params)
        vision_trainable = sum(p.numel() for p in vision_params if p.requires_grad)
        if vision_trainable > 0:
            print(f"Warning: Vision Tower has {vision_trainable} trainable parameters, but all should be frozen")
            if vision_trainable > 200:
                raise RuntimeError(f"Vision Tower has {vision_trainable} trainable parameters, but all should be frozen")
        neuro_params = list(neuro_tower.parameters())
        neuro_total = sum(p.numel() for p in neuro_params)
        neuro_trainable = sum(p.numel() for p in neuro_params if p.requires_grad)
    else:
        warnings.warn("Can not get mm_projector")


    # Check LoRA configuration
    lora_total = 0
    lora_active = 0
    target_modules_count = 0
    if isinstance(model, PeftModel):
        if model.peft_type != 'LORA':
            raise ValueError(f"Model type should be LORA, but got {model.peft_type}")

        target_modules = model.peft_config['default'].target_modules
        target_modules_count = len(target_modules)

        for name, module in model.named_modules():
            if name.split('.')[-1] not in target_modules:
                continue

            # Check original weights are frozen
            if hasattr(module, 'weight') and module.weight.requires_grad:
                raise RuntimeError(f"Module {name}'s original weights are not frozen")

            # Collect LoRA parameters
            current_lora = 0
            current_active = 0
            if hasattr(module, 'lora_A'):
                for p in module.lora_A.parameters():
                    lora_total += p.numel()
                    current_lora += p.numel()
                    if p.requires_grad:
                        lora_active += p.numel()
                        current_active += p.numel()

            if hasattr(module, 'lora_B'):
                for p in module.lora_B.parameters():
                    lora_total += p.numel()
                    current_lora += p.numel()
                    if p.requires_grad:
                        lora_active += p.numel()
                        current_active += p.numel()

            # Verify all LoRA parameters are active
            if current_lora > 0 and current_active != current_lora:
                warnings.warn("The LORA is not active. Please check if it is correct")


    # Print final report
    rank0_print("✓ All checks passed")
    rank0_print(f"Base Model parameters - Total: {base_total:,}, Frozen: {(base_total - base_trainable):,}, Active: {base_trainable:,}")

    if mm_projector is not None:
        rank0_print(f"Vision Tower parameters - Total: {vision_total:,}, Frozen: {(vision_total - vision_trainable):,}, Active: {vision_trainable}")
        rank0_print(f"Neuro Tower parameters - Total: {neuro_total:,}, Frozen: {(neuro_total - neuro_trainable):,}, Active: {neuro_trainable:,}")
        rank0_print(f"MM Projector parameters - Total: {mm_total:,},  Frozen: {mm_frozen:,}, Active: {mm_active:,}")

    if isinstance(model, PeftModel):
        rank0_print(f"LoRA modules: {target_modules_count}")
        rank0_print(f"LoRA parameters - Total: {lora_total:,}, Active: {lora_active:,}, Frozen: {(lora_total - lora_active):,}")
        model.print_trainable_parameters()






import torch.nn.functional as F
import numpy as np
import warnings

def check_neuroTower_pass_test(model,npz_data):
    statistic_arr=[]
    npz_data=np.load(npz_data)
    eeg_data=torch.tensor(npz_data['eeg_data']).to('cuda:0',dtype=torch.float32)
    img_feature=torch.tensor(npz_data['img_feature']).to('cuda:0',dtype=torch.float32)

    predict_feature=model.forward(eeg_data)
    origin_feature=img_feature
    predict_feature=F.normalize(predict_feature, p=2, dim=-1).to('cuda:0')
    origin_feature=F.normalize(origin_feature, p=2, dim=-1).to('cuda:0')

    statistic_arr.append(abs(torch.nn.functional.cosine_similarity(predict_feature,origin_feature).cpu().detach().numpy()))
    if abs(np.mean(statistic_arr))>0.044:
        print('NeuroTower pass the test, success')
    else:
        warnings.warn(f'check fail,pleack check if checkpoint is load fail,get {np.mean(statistic_arr)}')
    del npz_data, eeg_data, img_feature, predict_feature, origin_feature
    torch.cuda.empty_cache()




def load_eeg_cov_file(root_path,json_file_path='data/Instruction_data/EEGpretrain_total_conv.json'):

    json_file_path=os.path.join(root_path,json_file_path)
    with open(json_file_path, 'r') as f:
        json_datas = json.load(f)
    random_sample= random.choice(json_datas)
    eeg_path= os.path.join(root_path,random_sample['eeg'])
    question= random_sample['conversations'][0]['value']
    Answer= random_sample['conversations'][1]['value']
    return eeg_path,question,Answer




def eval_model_singel_turn(json_conv_file,model,tokenizer,verbose=True,put_modility=True,show_prompt=False,put_random_image_token=False):
    eeg_path, question, answer = load_eeg_cov_file(json_file_path=json_conv_file)
    inp=question
    conv = conv_templates['llava_v1'].copy()
    roles = conv.roles
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    if put_modility:
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt',show_prompt=show_prompt).unsqueeze(0).to(model.device)
    else:
        input_ids = tokenizer(prompt, return_tensors='pt', padding="longest", max_length=tokenizer.model_max_length, truncation=True).input_ids.to(model.device)
    eeg_tensor=np.load(eeg_path)

    if put_random_image_token:
        eeg_tensor = np.random.rand(1, 32,512)

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False) if verbose else None
    if verbose:
        print(f"Question:{question.replace('<image>','').strip()}")
        print(f"GroundTruth:{answer}")
        print(f"LLM Response:",end='')

    generate_kwargs ={

    }
    if put_modility:
        generate_kwargs['modilitys']=[eeg_tensor]
        generate_kwargs['modility_types']=['eeg']

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            do_sample=True,
            temperature=0.5,
            max_new_tokens=200,
            streamer=streamer,
            use_cache=True,
            **generate_kwargs,
        )
        outputs = tokenizer.decode(output_ids[0]).strip()
    return {
        'question': question,
        'answer': answer,
        'LLM_output': outputs
    }
    
 
import glob
def remove_global_step(output_dir):
    for checkpoint in glob.glob(os.path.join(output_dir, "checkpoint*")):
        if not os.path.isdir(checkpoint):
            continue
        for gs_folder in glob.glob(os.path.join(checkpoint, "global_step*")):
            if os.path.isdir(gs_folder):
                import shutil
                try:
                    shutil.rmtree(gs_folder)
                except Exception as e:
                    print(f"",end='')


def process_input_ids_for_llama2_debug(input_ids, tokenizer):
        # Handle numpy array or tensor cases
        if not isinstance(input_ids, list):
            if input_ids.ndim == 2:
                input_ids = input_ids[:, 1:] if (input_ids[0][0] == 1 and input_ids[0][1] == 1) else input_ids
                input_length = input_ids.shape[1]
            else:
                input_ids = input_ids[1:] if (input_ids[0] == 1 and input_ids[1] == 1) else input_ids
                input_length = len(input_ids)
        # Handle list case
        else:
            input_ids = input_ids[1:] if (len(input_ids) > 0 and input_ids[0] == 1 and input_ids[1] == 1) else input_ids
            input_length = len(input_ids)
        
        # Check and truncate if exceeds max length
        if input_length > tokenizer.model_max_length:
            warning = f"Input length {input_length} exceeds model max length {tokenizer.model_max_length}. Truncating input."

            warnings.warn(warning, RuntimeWarning)
            if not isinstance(input_ids, list):
                input_ids = input_ids[:, :tokenizer.model_max_length] if input_ids.ndim == 2 else input_ids[:tokenizer.model_max_length]
            else:
                input_ids = input_ids[:tokenizer.model_max_length]
        
        return input_ids




