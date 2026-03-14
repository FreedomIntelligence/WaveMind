import os
import sys
from time import sleep
import os
import random
import numpy as np
import pandas as pd
root_dir = f'{os.environ["WaveMind_ROOT_PATH_"]}/Data_Engineering'

# Load question templates
with open(os.path.join(root_dir, 'prompt/WaveMind_Bench/ImageNet-EEG/Question.txt'), 'r') as f:
    question_templates = [line.strip() for line in f.readlines() if line.strip()]

from EEG_Encoder.Tools.dataBuilder import CLIPDataset

import copy
import re
import secrets
import string
import numpy as np
import json
import torch
import os
from datetime import datetime
from PIL import Image
from tqdm import tqdm
import mne
from Data_Engineering.utils import EidGenerator






def generate_option_letters(num_options):
    letters = []
    for i in range(num_options):
        if i < 26:
            letter = chr(65 + i)
        else:
            first = (i - 26) // 26
            second = (i - 26) % 26
            letter = chr(65 + first) + chr(65 + second)
        letters.append(letter)
    return letters

def generate_qa_dataset(test_dataset, class_all, root_dir, eid_generator,num_options=4):
    eeg_data_dir = os.path.join(root_dir, 'data/WaveMind_Bench/EEG_data')
    os.makedirs(eeg_data_dir, exist_ok=True)
    option_letters = generate_option_letters(num_options)
    options_template = ' '.join([f'({letter}){{{letter.lower()}}}' for letter in option_letters])

    # Will select a new random question template for each question


    data_rows = []

    for _ in tqdm(range(int(3e3))):
        data = random.choice(test_dataset)
        texts=data['text'].decode('utf-8').strip().split(' ')
        first_word=texts[-2]
        second_word=texts[-1].strip('.')
        data_class=f"{first_word} {second_word}"
        if data_class.startswith('a '):
            data_class = data_class[2:]
        eeg_tensor = data['eeg_data']
        assert data_class in class_all, f"Class {data_class} not in class_all"
        wrong_options = random.sample(list(class_all - {data_class}), num_options-1)
        all_options = wrong_options + [data_class]
        random.shuffle(all_options)

        correct_idx = all_options.index(data_class)
        correct_answer = option_letters[correct_idx]
        option_params = {letter.lower(): all_options[i] for i, letter in enumerate(option_letters)}
        base_question = random.choice(question_templates)
        question_str = f'{base_question} {options_template}'.format(**option_params)


        uid = eid_generator.get_eid()
        eeg_filename = f"{uid}.npy"
        eeg_full_path = os.path.join(eeg_data_dir, eeg_filename)

        np.save(eeg_full_path, eeg_tensor.cpu().numpy().astype(np.float16))

        data_rows.append({
            "id": uid,
            "eegpath": os.path.join('data/EEG_data', eeg_filename).replace('\\', '/'),
            "question": question_str,
            "correct_answer": correct_answer,
            "option_number": num_options
        })

    return pd.DataFrame(data_rows)


if __name__ == '__main__':
    
    ds_name='ImageNetEEG'
    # mode='train'
    mode='val'


    promt_dir = os.path.join(root_dir,'prompt')
    eeg_data_dir=os.path.join(root_dir,'data/EEG_data')

    question = os.path.join(promt_dir,'Artifect','Question.txt')
    answer = os.path.join(promt_dir,'Artifect','Answer.json')


    eid_generator=EidGenerator(f"test_{ds_name}_{mode}",cache_dir=f'{os.environ["WaveMind_ROOT_PATH_"]}/Data_Engineering/data/WaveMind_Bench/Record_data')


    ds_dataset = CLIPDataset(f'{os.environ["WaveMind_ROOT_PATH_"]}/data/Total/data_label.h5', mode='test' if mode=='val' else 'train',ground_truth_dir=f'{os.environ["WaveMind_ROOT_PATH_"]}/data/Total/CLIP_groundTruth',  dataset_name=ds_name, float_type='float32',exclude_dataset=None,use_aug=False)
    
    
    from tqdm import tqdm
    class_all=set()
    for i in tqdm(range(int(3e3))):
        sample_ds=random.choice(ds_dataset)
        texts=sample_ds['text'].decode('utf-8').strip().split(' ')
        first_word=texts[-2]
        second_word=texts[-1].strip('.')
        data_class=f"{first_word} {second_word}"
        if data_class.startswith('a '):
            data_class = data_class[2:]
        class_all.add(data_class)
        
    assert len(class_all)==40
    
    
    df_datasets = [generate_qa_dataset(ds_dataset, class_all, root_dir, eid_generator, num_options=2),generate_qa_dataset(ds_dataset, class_all, root_dir, eid_generator, num_options=4),generate_qa_dataset(ds_dataset, class_all, root_dir, eid_generator, num_options=40)]
    df_datasets=pd.concat(df_datasets)
    
    df_datasets.to_csv(f'{root_dir}/data/WaveMind_Bench/test_{ds_name}_{mode}.csv')