import copy
from dataclasses import dataclass
import random
from typing import Dict
import warnings
import transformers
from torch.utils.data import Dataset
import json
import os
from PIL import Image
import torch
from typing import Dict, Optional, Sequence, List
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.utils import preprocess, preprocess_multimodal, rank0_print


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))


        def get_arg(args, key, default=None):
            if isinstance(args, dict):
                return args.get(key, default)
            else:
                return getattr(args, key, default)

        filter_keywords = get_arg(data_args, "filter_keywords")
        if filter_keywords is not None:
            list_data_dict = self._filter_data(list_data_dict, filter_key=filter_keywords)

        list_data_dict=self._filter_pure_text_data(list_data_dict)

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

        assert self.data_args.root_folder is None, "when train llava, root_folder should be void"

    def __len__(self):
        return len(self.list_data_dict)

    def _filter_data(self, data_list, filter_key):
        len_of_list_pre= len(data_list)
        from tqdm import tqdm
        filtered_data = []
        for item in tqdm(data_list, desc="Filtering dataset"):
            if 'image' in item and filter_key in item['image']:
                continue
            filtered_data.append(item)
        rank0_print(f"Filtered {len_of_list_pre - len(filtered_data)} items from dataset")
        rank0_print(f"remaining percentage of dataset: {len(filtered_data) / len_of_list_pre * 100:.2f}")
        return filtered_data
    def _filter_pure_text_data(self, data_list):
        len_of_list_pre= len(data_list)
        from tqdm import tqdm
        filtered_data = []
        for item in tqdm(data_list, desc="Filtering dataset"):
            if 'image' not in item and 'images' not in item:
                continue
            filtered_data.append(item)
        rank0_print(f"Filtered {len_of_list_pre - len(filtered_data)} items from dataset")
        rank0_print(f"remaining percentage of dataset: {len(filtered_data) / len_of_list_pre * 100:.2f}")
        return filtered_data


    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample or 'images' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if ('image' in sample or 'images' in sample) else -cur_len
            length_list.append(cur_len)
        print(f"Multimodel samples number: {sum([1 for x in length_list if x > 0])}")
        print(f"Text samples number: {sum([1 for x in length_list if x < 0])}")
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:

        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]

        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            if sources[0]['image']!='':
                image_file = self.list_data_dict[i]['image']
        elif 'images' in sources[0]:
            images_list = self.list_data_dict[i].get('images', [])
            image_file = images_list[0] if images_list else ''
        else:
            image_file = ''
        if image_file:
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            try:
                image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            except Exception as e:
                # print(f"Error opening image {image_file}: {e}, in abspath {os.path.abspath(os.path.join(image_folder, image_file))})")
                return self.__getitem__((i + 1) % len(self))
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)

        else:
            # return self.__getitem__((i + 1) % len(self))
            sources = copy.deepcopy([e["conversations"] for e in sources])


        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))


        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            if self.list_data_dict[i]['image']!='':
                data_dict['image'] = image

        return data_dict




class WaveMindSupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args,train_mode='train',model=None):
        super(WaveMindSupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))
        self.model=model




        # def get_arg(args, key, default=None):
        #     if isinstance(args, dict):
        #         return args.get(key, default)
        #     else:
        #         return getattr(args, key, default)

        # filter_keywords = get_arg(data_args, "filter_keywords")
        # if filter_keywords is not None:
        #     list_data_dict = self._filter_data(list_data_dict, filter_key=filter_keywords)


        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        if train_mode=='train':

            self.list_data_dict = list_data_dict[:int(len(list_data_dict)*0.999)]
            print(f"train data size: {len(self.list_data_dict)}")
        else:
            self.list_data_dict = list_data_dict[int(len(list_data_dict)*0.999):]
            print(f"test data size: {len(self.list_data_dict)}")
        self.data_args = data_args

        self.root_path=self.data_args.root_folder

        assert self.data_args.image_folder is None, "when train mela, para of image_folder should be void"

    def __len__(self):
        return len(self.list_data_dict)

    # def _filter_data(self, data_list, filter_key):
        # len_of_list_pre= len(data_list)
        # from tqdm import tqdm
        # filtered_data = []
        # for item in tqdm(data_list, desc="Filtering dataset"):
        #     if 'image' in item and filter_key in item['image']:
        #         continue
        #     filtered_data.append(item)
        # rank0_print(f"Filtered {len_of_list_pre - len(filtered_data)} items from dataset")
        # rank0_print(f"remaining percentage of dataset: {len(filtered_data) / len_of_list_pre * 100:.2f}")
        # return filtered_data

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample or 'images' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if ('image' in sample or 'images' in sample) else -cur_len
            length_list.append(cur_len)
        print(f"Multimodel samples number: {sum([1 for x in length_list if x > 0])}")
        print(f"Text samples number: {sum([1 for x in length_list if x < 0])}")
        return length_list

    def append_to_question(self, dialogues, append_str):
        dialogue = dialogues[0] if isinstance(dialogues, list) and len(dialogues) > 0 else []
        modified_dialogue = [entry.copy() for entry in dialogue]
        for entry in modified_dialogue:
            if isinstance(entry, dict) and entry.get('from') == 'human':
                original_value = entry['value']
                if '<image>' in original_value:
                    question = original_value.replace('<image>', '').replace('<image>', '').strip()
                    entry['value'] = f'<image>\n{question}. {append_str}' if random.random() < 0.5 else f'{question}. {append_str} <image>\n'
                else:
                    entry['value'] = f'{original_value}. {append_str}'
                break
        return [modified_dialogue]

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        
        sources = self.list_data_dict[i]

        # print("***************************")
        # print(f"i: {i}")
        # print(f"self.list_data_dict[i]: {sources}")


        if isinstance(i, int):
            sources = [sources]
        search_result=None
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        eeg,image=None,None
        if 'image' in sources[0] or 'eeg' in sources[0]:
            if 'image' in sources[0]:
                if sources[0]['image']!='':
                    image_file = self.list_data_dict[i]['image']
                    processor = self.data_args.image_processor
                    img_path=os.path.join(self.root_path, image_file)
                    try:
                        image = Image.open(img_path).convert('RGB')
                    except Exception as e:
                        print(f"Error opening image {image_file}: {e}, in abspath {img_path})")
                        raise "oops"
                        return self.__getitem__((i + 1) % len(self))
                    if self.data_args.image_aspect_ratio == 'pad':
                        def expand2square(pil_img, background_color):
                            width, height = pil_img.size
                            if width == height:
                                return pil_img
                            elif width > height:
                                result = Image.new(pil_img.mode, (width, width), background_color)
                                result.paste(pil_img, (0, (width - height) // 2))
                                return result
                            else:
                                result = Image.new(pil_img.mode, (height, height), background_color)
                                result.paste(pil_img, ((height - width) // 2, 0))
                                return result
                        image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                        image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                    else:
                        image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

            if 'eeg' in sources[0]:
                if sources[0]['eeg']!='':
                    eeg_file = self.list_data_dict[i]['eeg']

                    eeg_path=os.path.join(self.root_path, eeg_file)
                    try:
                        import numpy as np
                        eeg=np.load(eeg_path)
                        # Validate EEG shape against model's expected sampling rate
                        expected_fs = self.model.config.eeg_sampling_rate
                        assert eeg.shape == (32, expected_fs), \
                            f"Expected EEG shape (32, {expected_fs}), but got {eeg.shape}"

                    except Exception as e:
                        print(f"Error opening eeg {eeg_file}: {e}, in abspath {eeg_path})")
                        # return self.__getitem__((i + 1) % len(self))
                        print(f"Error opening eeg {eeg_file}: {e}, in abspath {eeg_path})")
                        raise "oops"

                    if random.random() < 0.015:
                        search_result=self.model.DBtool.get_search_result_from_EEG(eeg_model=self.model.get_model().get_neuro_tower(),eeg=eeg,topk=None)

            # image token random place
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)

        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        ################################

        if search_result is not None:
            sources=self.append_to_question(sources,search_result)
        


        ################################

        
        

        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=(eeg is not None or image is not None),)
        
        
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            if self.list_data_dict[i]['image']!='':
                data_dict['image'] = image

        # eeg exist in the data
        if 'eeg' in self.list_data_dict[i]:
            if self.list_data_dict[i]['eeg']!='':
                data_dict['eeg'] = eeg

        if 'eeg' not in self.list_data_dict[i] and 'image' not in self.list_data_dict[i]:
            warnings.warn("No EEG, no image in this element, please check your data")

        
        return data_dict


import json
from typing import Dict, List
import torch
from torch.utils.data import Dataset
import transformers





@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:

        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        
        
        
        
    

        # if input_ids.shape[1] > self.tokenizer.model_max_length:
        #     warnings.warn(
        #     f"Input length {input_ids.shape[1]} exceeds model max length {self.tokenizer.model_max_length}. "
        #         "Truncating input and labels to model max length. trigger 386 dataloader.py"
        #     )
        #     input_ids = input_ids[:, :self.tokenizer.model_max_length-1]
        #     labels = labels[:, :self.tokenizer.model_max_length-1]
            
        #     for i in range(input_ids.shape[0]):
        #         tmp_input_ids = input_ids[i]
        #         if IMAGE_TOKEN_INDEX not in tmp_input_ids:
        #             input_ids[i][10] = IMAGE_TOKEN_INDEX
        

        if input_ids.shape[1] > self.tokenizer.model_max_length:
            warnings.warn(
            f"Input length {input_ids.shape[1]} exceeds model max length {self.tokenizer.model_max_length}. "
                "Truncating input and labels to model max length. trigger 386 dataloader.py"
            )
            # In case IMAGE_TOKEN_INDEX being truncated
            for i in range(input_ids.shape[0]):
                tmp_input_ids = input_ids[i]
                has_image_before = IMAGE_TOKEN_INDEX in tmp_input_ids
                
                if has_image_before:
                    truncated_tmp = tmp_input_ids[:self.tokenizer.model_max_length-1]
                    has_image_after = IMAGE_TOKEN_INDEX in truncated_tmp
                    if not has_image_after:
                        #Avoid inseart into the bos token/sequnence
                        input_ids[i][10] = IMAGE_TOKEN_INDEX
            # Do truncate
            input_ids = input_ids[:, :self.tokenizer.model_max_length-1]
            labels = labels[:, :self.tokenizer.model_max_length-1]

        
        
        
        
        
        attention_mask=input_ids.ne(self.tokenizer.pad_token_id)
        # in case of pad==eos, eos(this time is pad) shold be True
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for i in range(input_ids.shape[0]):
                tmp_input_ids = input_ids[i]
                
                # find the last non-pad token
                for idx in range(len(tmp_input_ids)-1,-1,-1):
                    if tmp_input_ids[idx] != self.tokenizer.pad_token_id:
                        break
                # set the last non-pad token as True to pay attention to it
                if idx+1<input_ids.shape[1]:
                    attention_mask[i][idx+1]=True
            
        
        
        
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )
        
        # print(input_ids)
        # print(labels)
        # print(attention_mask)

        # sys.exit()
        
        
        
        
        batch['modilitys']=[]
        batch['modility_types']=[]
        for instance in instances:
            if 'eeg' in instance or 'image' in instance:
                if 'eeg' in instance and 'image' not in instance:
                    batch['modilitys'].append(instance['eeg'])
                    batch['modility_types'].append('eeg')
                elif 'image' in instance and 'eeg' not in instance:
                    batch['modilitys'].append(instance['image'])
                    batch['modility_types'].append('image')
                else:
                    raise ValueError('eeg and image cannot exist at the same time')
            else:
                batch['modilitys'].append(None)
                batch['modility_types'].append(None)
        
        return batch




def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args,model=None) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    if data_args.train_data_mode=='llava':
        train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                    data_path=data_args.data_path,
                                    data_args=data_args)
    elif data_args.train_data_mode=='mela' and data_args.train_pure_text==False:
        train_dataset = WaveMindSupervisedDataset(tokenizer=tokenizer,
                                    data_path=data_args.data_path,
                                    data_args=data_args,train_mode='train',model=model)
        test_dataset = WaveMindSupervisedDataset(tokenizer=tokenizer,
                                    data_path=data_args.data_path,
                                    data_args=data_args,train_mode='test',model=model)
    # elif data_args.train_data_mode == 'mela' and data_args.train_pure_text == True:
    #     train_dataset = PureTextDataset(tokenizer=tokenizer,data_path=data_args.data_path,)

    else:
        raise "Error"
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)


    return dict(train_dataset=train_dataset,
                eval_dataset=test_dataset if data_args.train_data_mode=='mela' else None,
                data_collator=data_collator)
