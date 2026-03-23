import os
import os
import random
import numpy as np
import pandas as pd
from data.Utils import get_wavemind_root
root_dir = f'{get_wavemind_root()}/Data_Engineering'

from EEG_Encoder.Tools.dataBuilder import CLIPDataset


import numpy as np
import os
from tqdm import tqdm
from Data_Engineering.utils import EidGenerator


# Load question templates
with open(os.path.join(root_dir, 'prompt/WaveMind_Bench/SEED/Question.txt'), 'r') as f:
    question_templates = [line.strip() for line in f.readlines() if line.strip()]



def generate_qa_dataset(test_dataset, class_all, root_dir, eid_generator,num_options=4):
    eeg_data_dir = os.path.join(root_dir, 'data/WaveMind_Bench/EEG_data')
    os.makedirs(eeg_data_dir, exist_ok=True)

    option_letters = [chr(65 + i) for i in range(num_options)]
    options_template = ' '.join([f'({letter}){{{letter.lower()}}}' for letter in option_letters])
    # question_template = f'What is the emotion pattern inside this EEG? Select one letter in the following. Do not explain. {options_template}'

    data_rows = []

    for _ in tqdm(range(int(3e3))):
        data = random.choice(test_dataset)
        texts=data['text'].decode('utf-8').strip().split(' ')
        data_class=texts[7]
        eeg_tensor = data['eeg_data']
        wrong_options = random.sample(list(class_all - {data_class}), num_options-1)
        all_options = wrong_options + [data_class]
        random.shuffle(all_options)

        correct_idx = all_options.index(data_class)
        correct_answer = option_letters[correct_idx]

        option_params = {letter.lower(): all_options[i] for i, letter in enumerate(option_letters)}
        # question_str = question_template.format(**option_params)
        
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
    ds_name='SEED'
    mode='val'

    eid_generator=EidGenerator(f"test_{ds_name}_{mode}",cache_dir=f'{get_wavemind_root()}/Data_Engineering/data/WaveMind_Bench/Record_data')


    ds_dataset = CLIPDataset(f'{get_wavemind_root()}/data/Total/data_label.h5', mode='test' if mode=='val' else 'train',ground_truth_dir=f'{get_wavemind_root()}/data/Total/CLIP_groundTruth',  dataset_name=ds_name, float_type='float32',exclude_dataset=None,use_aug=False)
    
    
    from tqdm import tqdm
    class_all=set()
    for i in tqdm(range(200)):
        sample_ds=random.choice(ds_dataset)
        texts=sample_ds['text'].decode('utf-8').strip().split(' ')
        data_class=texts[7]
        class_all.add(data_class)
    assert len(class_all)==3
    
    
    df_datasets = [generate_qa_dataset(ds_dataset, class_all, root_dir, eid_generator, num_options=2),generate_qa_dataset(ds_dataset, class_all, root_dir, eid_generator, num_options=3)]
    df_datasets=pd.concat(df_datasets)
    
    df_datasets.to_csv(f'{root_dir}/data/WaveMind_Bench/test_{ds_name}_{mode}.csv')
