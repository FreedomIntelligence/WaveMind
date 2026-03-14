import os
os.environ['CUDA_VISIBLE_DEVICES']='3'
import pandas as pd
import random
import re
import warnings
from tqdm import tqdm
import numpy as np
import random
import json
import argparse

import torch
from llava.constants import IMAGE_TOKEN_INDEX
from llava.eval.run_llava import load_image
from llava.mm_utils import get_model_base_from_path, process_images, tokenizer_image_token
from llava.conversation import conv_templates, SeparatorStyle, get_conversation_template
from transformers import StoppingCriteria, TextStreamer
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path

if os.environ['WaveMind_ROOT_PATH_'] is not None:
    root_path=os.path.join(os.environ['WaveMind_ROOT_PATH_'],'Data_Engineering')
else:
    raise ValueError("Please set the WaveMind_ROOT_PATH_ environment variable to the root path of your project.")



from dataclasses import dataclass


@dataclass
class PDDataset:
    pd_data: pd.DataFrame
    option_number: int
    mode: str
def get_question_set(data, option_number):
    return data[data['option_number'] == option_number].drop(columns=['option_number'])



def get_RAG_search_result(eeg, llava_model,topk=None):
    topk=llava_model.DBtool.recommend_topk if topk is None else topk
    search_result=llava_model.DBtool.get_search_result_from_EEG(eeg_model=llava_model.get_model().get_neuro_tower(),eeg=eeg,topk=topk)
    return search_result


def eval_model_singel_turn_classification(sample_data,model,tokenizer,method,verbose=False,topk=None,model_base=None):
    if method=='only_RAG':
        put_modility=False
        RAG=True
        put_random_image_token=False
    elif method=='only_modility':
        put_modility=True
        RAG=False
        put_random_image_token=False
    elif method=='RAG+modility':
        put_modility=True
        RAG=True
        put_random_image_token=False
    elif method=='random':
        put_modility=True
        RAG=False
        put_random_image_token=True
    else:
        raise ValueError(f"Unknown method: {method}")

    def get_ans(ans):
        pattern = r'.*?([A-Z]{1,2}(?:[、, ，]+[A-Z]{1,2})*)'
        matches = re.findall(pattern, ans)
        
        if matches:
            last_match = matches[-1]
            return ''.join(re.split(r'[、, ，]+', last_match))
        return ''
    
    question=sample_data['question'].replace('<image>','').strip()
    answer=sample_data['correct_answer']
    eeg_path=os.path.join(root_path,sample_data['eegpath'])
    try:
        eeg_tensor=np.load(eeg_path)
        # Validate EEG shape against model's expected sampling rate
        expected_fs = model.config.eeg_sampling_rate
        if eeg_tensor.shape[-1] != expected_fs:
            raise ValueError(
                f"EEG shape mismatch: model expects {expected_fs} samples, "
                f"but got {eeg_tensor.shape[-1]}"
            )
    except (FileNotFoundError, OSError, ValueError) as e:
        print(f"Error loading {eeg_path}: {e}")
        print("1. Are you download the full copy of datasets? (3 Dataset needs extra-auth to download.)")
        print("2. Are you using the correct path? (The path should be relative to the root of the project.)")
        import sys
        sys.exit(1)
    inp = ("<image>" if put_modility==True else "")+question+(f"\n{get_RAG_search_result(eeg_tensor,model,topk=topk)}" if RAG==True else '')

    assert model_base is not None
    conv = get_conversation_template(model_base)

    roles = conv.roles
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()


    

    if put_modility:
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt',show_prompt=verbose).unsqueeze(0).to(model.device)
        
    else:
        input_ids = tokenizer(prompt, return_tensors='pt', padding="longest", max_length=tokenizer.model_max_length, truncation=True).input_ids.to(model.device)
        
    from llava.utils import process_input_ids_for_llama2_debug

    input_ids = process_input_ids_for_llama2_debug(input_ids, tokenizer)
    


    if put_random_image_token:
        expected_fs = model.config.eeg_sampling_rate
        eeg_tensor = np.random.rand(32, expected_fs)

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False) if verbose else None
    # if verbose:
    #     print(f"Question:{question.replace('<image>','').strip()}")
    #     print(f"GroundTruth:{answer}")
    #     print(f"LLM Response:",end='')

    generate_kwargs ={

    }
    
    if put_modility:
        generate_kwargs['modilitys']=[eeg_tensor]
        generate_kwargs['modility_types']=['eeg']
        
    # if tokenizer.pad_token==tokenizer.eos_token:
    #     generate_kwargs['pad_token_id']=tokenizer.eos_token_id
        # generate_kwargs['attention_mask']= torch.ones(input_ids.shape, dtype=torch.long, device=model.device)


    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            do_sample=True,
            temperature=0.5,
            max_new_tokens=200,
            streamer=streamer,
            use_cache=True,
            # pad_token_id=tokenizer.eos_token_id,
            **generate_kwargs,
        )
        outputs = tokenizer.decode(output_ids[0],skip_special_tokens=True).strip()
    
    
    return {
        'question': question,
        'answer': answer,
        'LLM_output': outputs,
        'LLM_answer':get_ans(outputs),
    }


def eval_model_classification(test_data, method, granularity=1000,verbose=False,topk=None,model_base=None):
    def extract_option_names(input_str):
        option_map = {}
        start = 0
        while True:
            left_paren = input_str.find('(', start)
            if left_paren == -1:
                break
            right_paren = input_str.find(')', left_paren)
            if right_paren == -1:
                break
            option_letter = input_str[left_paren+1:right_paren].strip().upper()
            next_left_paren = input_str.find('(', right_paren)
            option_name = input_str[right_paren+1:next_left_paren].strip() if next_left_paren != -1 else input_str[right_paren+1:].strip()
            option_map[option_letter] = option_name
            start = next_left_paren
        return option_map
    count_per_class={}
    total_correct = 0
    for i in (tqdm(range(granularity))):
        
        random_int=i
        sample_data=test_data.iloc[random_int]
        result=eval_model_singel_turn_classification(sample_data=sample_data,model=model,tokenizer=tokenizer,verbose=verbose,method=method,topk=topk,model_base=model_base)

        LLM_answer=result['LLM_answer'].lower().strip()
        correct_answer=result['answer'].lower().strip()

        if LLM_answer=="":
            warnings.warn(f"LLM answer is empty! {result['question']}")


        option_map = extract_option_names(result['question'])
        correct_letter = correct_answer.upper()
        correct_name = option_map.get(correct_letter, None)
          

        # Determine correctness
        LLM_letter = LLM_answer.upper()
        valid_LLM = LLM_letter in option_map
        correct = 1 if (valid_LLM and LLM_letter == correct_letter) else 0

        # Update class statistics
        if correct_name in count_per_class:
            count_per_class[correct_name]['count'] += 1
            count_per_class[correct_name]['correct'] += correct
        else:
            count_per_class[correct_name] = {'count': 1, 'correct': correct}
        total_correct += correct



        if verbose:
            print("[Question]\n", result['question'])
            print("[Options]", option_map)
            print(f"[Correct] Letter: {correct_letter} -> Name: {correct_name}")
            print(f"[LLM Answer] Letter: {LLM_letter} ({'Valid' if valid_LLM else 'Invalid'})")
            print("—"*50)
            
            
    total = granularity
    total_acc = total_correct / total if total > 0 else 0 
    
    
    if total > 0:
        class_accuracies = [c['correct'] / c['count'] for c in count_per_class.values()]
        weighted_acc = sum(class_accuracies) / len(class_accuracies)
    else:
        weighted_acc = 0


    return {
        'weighted_accuracy': weighted_acc,
        'total_accuracy': total_acc, 
        'class_stats': count_per_class
    }



if __name__ == '__main__':
    """
    EEG Classification Evaluation Script
    
    Usage Examples:
    1. Basic evaluation with default parameters:
       python Evaluation_Classification.py
       
    2. Evaluate with specific options and verbosity (only_max_options:40 options, all_options:2/4/40):
       python Evaluation_Classification.py --ops only_max_options --modilities only_modility RAG+modility random --verbose
       
    3. Evaluate with custom model path and time steps:
       python Evaluation_Classification.py --model_path /path/to/model --time_steps 400 450 500
    
    4. Evaluate with specific datasets and granularity:
       python Evaluation_Classification.py --ds_names TUEV ImageNetEEG SEED --granularity 1000
    
    5. Evaluate with specific top-k values for RAG search:
       python Evaluation_Classification.py --topks 1 5 10
    """
    
    parser = argparse.ArgumentParser(description='EEG Classification Evaluation')
    parser.add_argument('--ops', type=str, default='all_options',
                       choices=['all_options', 'only_max_options'],
                       help='Whether to evaluate all options or only max options')
    parser.add_argument('--ds_names', type=str, nargs='+', default=['THING','ImageNetEEG','SEED','TUEV','TUAB'],
                       help='List of dataset names to evaluate')
    parser.add_argument('--time_steps', type=int, nargs='+', default=[None],
                       help='Time steps to evaluate (use None for default)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--modilities', type=str, nargs='+', default=['only_modility'],
                       choices=['only_modility', 'random', 'RAG+modility', 'only_RAG'],
                       help='Modility methods to evaluate')
    parser.add_argument('--granularity', type=int, default=3000,
                       help='Number of samples to evaluate')
    parser.add_argument('--model_path', type=str,
                       help='Path to the model checkpoint')
    parser.add_argument('--model_base', type=str, default=None,
                       help='Base model name (optional, will be auto-detected if None)')
    parser.add_argument('--topks', type=int, nargs='+', default=[None],
                       help='Top-k values for RAG search')
    
    args = parser.parse_args()
    
    ops = args.ops
    ds_names = args.ds_names
    time_steps = args.time_steps
    verbose = args.verbose
    modilities = args.modilities
    granularity = args.granularity
    model_path = args.model_path
    model_base = get_model_base_from_path(model_path, model_base=args.model_base)
    topks = args.topks
    
    
    
    random.seed(666)
    for time_step in time_steps:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tokenizer, model, modility_processor, context_len = load_pretrained_model(
                model_path=model_path,
                model_base=get_model_base_from_path(model_path, model_base=model_base),
                model_name=get_model_name_from_path(model_path),
                timestep=time_step
            )
        
        

        accs,modes, options, methods,ds_names_pd,topk_pd = [], [], [], [],[],[]
        
        for ds_name in ds_names:
            addon='' if ds_name!='THING' else f'_zero-shot'
            all_pd_data = []
            for mode in ['val']:
                total_data=pd.read_csv(f'{root_path}/data/Test_data/test_{ds_name}_{mode}{addon}.csv')
                if verbose:
                    print(f"load file: {root_path}/data/Test_data/test_{ds_name}_{mode}{addon}.csv")
                
                if ops=='only_max_options':
                    op_number=max(total_data['option_number'].unique())
                    pd_data = get_question_set(total_data, op_number)
                    all_pd_data.append(PDDataset(pd_data=pd_data, option_number=op_number, mode=mode))
                    print(f"Loaded Dataset: {ds_name}, Mode: {mode}, Option_number: {op_number}")
                elif ops=='all_options':
                    for op_number in total_data['option_number'].unique():
                        pd_data = get_question_set(total_data, op_number)
                        all_pd_data.append(PDDataset(pd_data=pd_data, option_number=op_number, mode=mode))
                        print(f"Loaded Dataset: {ds_name}, Mode: {mode}, Option_number: {op_number}")
                else:
                    raise ValueError(f"Unknown method: {ops}") 
                
                


            for topk in topks:
                print(f"Topk: {topk}")
                for method in tqdm(modilities):
                    print(f"Method: {method}")
                    for cur_pd in all_pd_data:
                        print(f"Mode: {cur_pd.mode}, Option_number: {cur_pd.option_number}")
                        acc=eval_model_classification(test_data=cur_pd.pd_data,granularity=granularity,method=method,verbose=verbose,topk=topk,model_base=model_base)
                        # only TUEV class suffer unbalanced problem
                        acc= acc['weighted_accuracy'] if ds_name=='TUEV' else acc['total_accuracy']
                        accs.append(acc)
                        modes.append(cur_pd.mode)
                        options.append(cur_pd.option_number)
                        methods.append(method)
                        ds_names_pd.append(ds_name)
                        topk_pd.append(topk)

                        print(f"Accuracy: {acc:.2%}")

        df = pd.DataFrame({
            'mode': modes,
            'option_number': options,
            'accuracy': accs,
            'method': methods,
            'dataset': ds_names_pd,
            'topk': topk_pd
        })
        print(df)
        if time_steps is not [None] and verbose:
            print(f"Time_step: {time_step}")
        
        
        del model,tokenizer,modility_processor
        torch.cuda.empty_cache()

