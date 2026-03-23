from collections import defaultdict
import os
import pickle
from typing import Dict
import warnings
import logging
from mne import create_info
import torch.amp
import torch.nn as nn
import torch
from torch.cuda.amp import autocast
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import mne
from EEG_Encoder.Model.baseModel import ATMS,EEGConformer_Encoder,NICE, model_selection
from EEG_Encoder.Model.CommonBlock import Config
from torcheeg import transforms
from data.Utils import FilterTransform, get_wavemind_root
import random

class NeuroTower(nn.Module):
    def __init__(self,config):
        super(NeuroTower, self).__init__()
        self.is_loaded = False
        self.eeg_tower=None
        self.config=config
        # Read from LLM config, None if not present (first-time training)
        self.config_fs = getattr(config, 'eeg_sampling_rate', None)
        self.model_fs = None  # Will be set from model_selection return value
        self.eegProcessor = None  # Will be initialized after fs is determined
    def load_model(self, load_checkpoint=True):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format('NeuroTower'))
            return
        print('load neurotower model')
        
        if hasattr(self.config, 'neuro_tower'):
            neuro_tower_name=self.config.neuro_tower if type(self.config.neuro_tower)==str else self.config.neuro_tower[0]
        else:
            warnings.warn("neuro_tower_name is None in config, set default value ATMSmodify")
            neuro_tower_name='ATMSmodify'
            
        root = get_wavemind_root()
        load_dir=os.path.join(root,'EEG_Encoder/Resource/Checkpoint/ALL')

        # Capture the returned sampling rate from model_selection
        self.eeg_tower, self.model_fs = model_selection(neuro_tower_name,load_dir=load_dir if load_checkpoint else None)

        # Scenario 1: First-time training (cfg doesn't have eeg_sampling_rate)
        if self.config_fs is None or not hasattr(self.config, 'eeg_sampling_rate'):
            # Save model's fs to config for future use
            self.config.eeg_sampling_rate = self.model_fs
            self.config_fs = self.model_fs
            logging.info(
                f"First-time training: Setting config.eeg_sampling_rate = {self.model_fs}Hz "
                f"(from model_selection)"
            )

        # Scenario 2: Re-training or Inference (cfg already has eeg_sampling_rate)
        else:
            # Cross-check: model's fs must match config's fs
            if self.model_fs != self.config_fs:
                raise ValueError(
                    f"Sampling rate mismatch! Model expects {self.model_fs}Hz "
                    f"(from checkpoint/architecture), but LLM config specifies {self.config_fs}Hz. "
                    f"Please ensure model and config are compatible."
                )
            logging.info(
                f"Cross-check passed: model_fs ({self.model_fs}Hz) == config_fs ({self.config_fs}Hz)"
            )

        # Use validated fs
        self.fs = self.model_fs
        self.eegProcessor = EEGProcessor(fs=self.fs)

        self.check_model_pass_test(self.eeg_tower,npz_data=os.path.join(get_wavemind_root()+"/data/Total/test_FeatureExtractor.npz"))
        


        self.is_loaded = True

    def check_model_pass_test(self,model,npz_data):
        statistic_arr=[]
        if not os.path.exists(npz_data):
            return
        npz_data=np.load(npz_data)
        eeg_data=torch.tensor(npz_data['eeg_data'])
        img_feature=torch.tensor(npz_data['img_feature'])

        predict_feature=model.forward(eeg_data)['pooler_output']
        origin_feature=img_feature
        predict_feature=F.normalize(predict_feature, p=2, dim=-1).to('cuda:0')
        origin_feature=F.normalize(origin_feature, p=2, dim=-1).to('cuda:0')

        statistic_arr.append(abs(torch.nn.functional.cosine_similarity(predict_feature,origin_feature).cpu().detach().numpy()))
        del predict_feature,origin_feature
        if abs(np.mean(statistic_arr))>0.044:
            # print('NeuroTower pass the test, success')
            return 
        else:
            print("*********************************************")
            warnings.warn(f'check fail,pleack check if checkpoint is load fail. The mean cosine similarity is {np.mean(statistic_arr)}, which is less than 0.044, indicating a potential issue with the model.')
            print("*********************************************")

    def forward(self, x):
        self.eeg_tower.to(device='cuda',dtype=torch.float32)
        x=x.to(device='cuda',dtype=torch.float32)
        assert next(self.eeg_tower.parameters()).is_cuda, 'NeuroTower must be on GPU'
        assert x.dim() == 3, 'Input must be a 3D tensor'
        assert x.size(1) == 32, 'Input must have 32 channels'
        assert x.size(2) == self.fs, f'Input must have {self.fs} time steps'


        res=self.eeg_tower(x)['pooler_output']
        return res



class EEGProcessor:

    def __init__(self, fs=512):
        """Initialize EEGProcessor with target sampling rate.

        Args:
            fs (int): Target sampling rate in Hz. Default 512Hz.
        """
        self.target_fs = fs
        

    def preprocess(self, eeg_data, fs=None,electrode_list=None,l_freq=0.5,h_freq=100,notch=60,verbose=False,confirm_cut_1s=None,rename_channel_function=None):
        def rename_channel_for_raw(raw,electrode_list):
            mapping = {old: new for old, new in zip(raw.info['ch_names'], electrode_list)}
            raw.rename_channels(mapping)
            return raw
        """
        Process the EEG data using the configured filter transform.
        
        :param data: The raw EEG data to be processed.
        :return: Processed EEG data.
        """
        if type(eeg_data)==np.ndarray:
            assert eeg_data.ndim == 2, "EEG data must be a 2D array"
            assert fs is not None
            assert electrode_list is not None
            eeg_data.astype(np.float32)
        elif type(eeg_data)==str:
            if eeg_data.endswith('.edf') or eeg_data.endswith('.bdf'):
                raw=mne.io.read_raw(eeg_data,preload=True).load_data()
                fs=raw.info['sfreq'] if  fs is None else fs
                if electrode_list is not None:
                    raw= rename_channel_for_raw(raw, electrode_list)
                else:
                    electrode_list= raw.info['ch_names']
                    if rename_channel_function is not None:
                        electrode_list=rename_channel_function(electrode_list)
                        raw= rename_channel_for_raw(raw, electrode_list)
                
                eeg_data = raw.get_data(picks=electrode_list)
                raw.close()
            elif eeg_data.endswith('.npy'):
                eeg_data = np.load(eeg_data)
                if fs is None:
                    raise ValueError("Please input fs")
                if electrode_list is None:
                    raise ValueError("Please input electrode_list")
    

            eeg_data = eeg_data.astype(np.float32)
            del raw
        else:
            raise TypeError("EEG data must be a numpy array or a file path to a .edf/.bdf file")
        
        # print(electrode_list)  

        self.trans=transforms.Compose([
                FilterTransform(fs=fs,
                    l_freq=l_freq,
                    h_freq=h_freq,
                    notch=notch,
                    electrode_list=electrode_list,
                    verbose=verbose,
                    downsample_fs=self.target_fs),
                transforms.MeanStdNormalize(axis=1),
        ])
        self.confirm_cut_1s=confirm_cut_1s
        self.fs=int(fs)
        
        time_point= eeg_data.shape[-1]
        before_time_second= time_point / self.fs
        if time_point / self.fs != 0 and self.confirm_cut_1s==None:
            raise ValueError(f"You are not input the EEG as 1 second segment. Please check your data. Or 1. set confirm_cut_1s='first' when init this class, we will cut and select 1s first.2. set confirm_cut_1s='random' when init this class, we will cut and select 1s randomly.\
                             ")
        eeg= self.trans(eeg=eeg_data)['eeg']
        assert eeg.ndim == 2, "EEG data must be a 2D array after preprocessing"
        assert eeg.shape[0] == 32, "EEG data must have 32 channels after preprocessing"
        self.fs=self.target_fs
        after_time_second = eeg.shape[-1] / self.fs
        assert abs(before_time_second-after_time_second) < 1e-5

        assert self.confirm_cut_1s in [None, 'first', 'random'], "confirm_cut_1s must be 'first', 'random' or None(do nothing)."
        if self.confirm_cut_1s=='random':
            start = np.random.randint(0, max(1, eeg.shape[-1] - self.target_fs))
            eeg = eeg[:, start:start + self.target_fs]
        elif self.confirm_cut_1s=='first':
             eeg = eeg[:, 0:self.target_fs]
        assert eeg.shape[-1] == self.target_fs, f"EEG data must have {self.target_fs} time steps after preprocessing, but got {eeg.shape[-1]}"
        
        eeg = eeg.astype(np.float32)
        
        # modified z score -0 to avoid instability
        # eeg=(eeg-0)/np.std(eeg,axis=-1,keepdims=True)
        
        
        return eeg






class DBsearch:

    
    
    def __init__(self, CLIP_path, dataset=None,THNING_closesetORzeroshot=None):
        self.CLIP_path = CLIP_path
        self.test_data_path = os.path.join(os.path.dirname(CLIP_path), 'test_RAG_data.pkl')
        prefix_list = ['ImageNetEEG', 'SEED', 'THING', 'TUAB', 'TUEV']
        self.prefix_label = ["Visual Stimuli", "Emotion Recognition",
                    "Visual Stimuli", "Abnormal Detection", "Event Detection"]
        # 40+3+200+1654+2+6
        self.check_model = False
        arrays = []
        labels = []
        types = []

        def load_data(prefix, add_str=''):
            filename = os.path.join(CLIP_path, f"{prefix}{add_str}.npy")
            labelname = os.path.join(CLIP_path, f"{prefix}{add_str}.pkl")

            if not os.path.exists(labelname):
                raise FileNotFoundError(f"File {labelname} not exist")
            if not os.path.exists(filename):
                raise FileNotFoundError(f"File {filename} not exist")

            # Load and validate feature array
            try:
                arr = np.load(filename)
            except Exception as e:
                raise RuntimeError(f"Failed to load {filename}: {e}")

            # Validate array shape and dimension
            if arr.ndim != 2:
                raise ValueError(f"{filename} must be 2D array, got {arr.ndim}D with shape {arr.shape}")

            if arr.shape[1] != 768:  # CLIP ViT-L/14-336 text embedding size
                import warnings
                warnings.warn(f"{filename} has unexpected embedding dimension: {arr.shape[1]} (expected 768)")

            # Load and validate label file
            try:
                with open(labelname, 'rb') as f:
                    label_list = pickle.load(f)
            except Exception as e:
                raise RuntimeError(f"Failed to load {labelname}: {e}")

            # Validate label list
            if not isinstance(label_list, list):
                raise TypeError(f"{labelname} should contain a list, got {type(label_list).__name__}")

            # Verify array and label count match
            if len(label_list) != arr.shape[0]:
                raise ValueError(
                    f"Size mismatch in {prefix}{add_str}: "
                    f"{filename} has {arr.shape[0]} embeddings, "
                    f"but {labelname} has {len(label_list)} labels"
                )

            arrays.append(arr)
            labels.extend(label_list)
            types.extend([self.prefix_label[prefix_list.index(prefix)]] * len(label_list))
            


        for prefix in prefix_list:
            if dataset is not None:
                if prefix != dataset:
                    continue
            
            if prefix == 'THING':
                if THNING_closesetORzeroshot == 'closeset' or THNING_closesetORzeroshot == None:
                    load_data(prefix, '_closeset')
                if THNING_closesetORzeroshot == 'zero-shot' or THNING_closesetORzeroshot == None:
                    load_data(prefix)
            else:
                load_data(prefix)

        
        assert len(arrays) > 0, "No data found in the specified files."

        self.db_feature = torch.tensor(np.concatenate(arrays, axis=0)).float()
        self.labels = labels
        self.types = types

        self.groups = {}
        unique_types = set(types)
        for type_name in unique_types:
            indices = [i for i, t in enumerate(types) if t == type_name]
            self.groups[type_name] = {
                'indices': torch.tensor(indices),
                'features': self.db_feature[indices],
                'labels': [labels[i] for i in indices]
            }
        
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.label_to_indices[label].append(idx)


        if len(self.labels) == 0:
            raise ValueError("No labels found in the database. Please check your data files.")

        # Validate database size consistency
        if not (len(self.db_feature) == len(self.labels) == len(self.types)):
            raise ValueError(
                f"Database size mismatch: "
                f"features={len(self.db_feature)}, "
                f"labels={len(self.labels)}, "
                f"types={len(self.types)}. "
                "This indicates corrupted database files."
            )

        # Warn if database size doesn't match expected value (informational)
        if len(self.labels) != 1824:
            import warnings
            warnings.warn(
                f"Database contains {len(self.labels)} labels (expected 1824). "
                "This may be intentional if you're using a subset of datasets. "
                "Full database includes: ImageNetEEG(40), SEED(3), THING(1654+200), TUAB(2), TUEV(6)."
            )
    
    
    def reset(self, *args, **kwargs):
        
        assert 'THNING_closesetORzeroshot' in kwargs, "THNING_closesetORzeroshot must be in kwargs."
        self.__init__(self.CLIP_path, *args, **kwargs)
        
    @property
    def recommend_topk(self):
        return min(int(len(self.labels)*0.25),420)
        
        

    def get_most_similar(self, vector, topk=1):
        if isinstance(vector, np.ndarray):
            vector = torch.from_numpy(vector).float()
        vector = F.normalize(vector, dim=-1)
        if vector.dim() == 1:
            vector = vector.unsqueeze(0)

        num_categories = len(self.groups)
        adjusted_topk = max(topk, num_categories)

        selected_indices = []
        visual_stimuli_indices = []

        for type_name, group in self.groups.items():
            features_norm = F.normalize(group['features'], p=2, dim=1)
            features_norm = features_norm.to(vector.device)
            sim = torch.matmul(features_norm, vector.T).squeeze()

            max_idx = torch.argmax(sim)
            original_idx = group['indices'].to(vector.device)[max_idx].item()
            selected_indices.append(original_idx)

            if type_name == "Visual Stimuli":
                visual_stimuli_indices = group['indices'].tolist()

        if adjusted_topk > num_categories:
            available_vs = [idx for idx in visual_stimuli_indices
                           if idx not in selected_indices]

            if available_vs:
                vs_features = self.db_feature[available_vs]
                vs_features_norm = F.normalize(vs_features, p=2, dim=1)
                vs_features_norm = vs_features_norm.to(vector.device)
                vs_sim = torch.matmul(vs_features_norm, vector.T).squeeze()

                k = adjusted_topk - num_categories
                k = min(k, len(available_vs))
                _, top_indices = torch.topk(vs_sim, k)

                additional_indices = [available_vs[i] for i in top_indices.tolist()]
                selected_indices.extend(additional_indices)

        final_indices = selected_indices[:adjusted_topk]
        result_labels = [self.labels[i] for i in final_indices]
        result_types = [self.types[i] for i in final_indices]
        return result_labels, result_types

    
    def get_most_similar_by_candidate(self, vector, candidates, topk=1):
        if isinstance(vector, np.ndarray):
            vector = torch.from_numpy(vector).float()
        vector = F.normalize(vector, dim=-1)
        if vector.dim() == 1:
            vector = vector.unsqueeze(0)
        
        candidate_indices = []
        for label in candidates:
            if label in self.label_to_indices:
                candidate_indices.extend(self.label_to_indices[label])
        
        if not candidate_indices:
            return [], []
        
        candidate_features = self.db_feature[candidate_indices]
        candidate_features_norm = F.normalize(candidate_features, p=2, dim=1).to(vector.device)
        vector = vector.to(candidate_features_norm.device)
        
        sim = torch.matmul(candidate_features_norm, vector.T).squeeze()
        
        _, sorted_indices = torch.sort(sim, descending=True)
        topk_indices = sorted_indices[:topk]
        
        original_indices = [candidate_indices[i] for i in topk_indices.tolist()]
        result_labels = [self.labels[i] for i in original_indices]
        result_types = [self.types[i] for i in original_indices]
        
        return result_labels, result_types
    
    
    
    
    def get_most_similar_by_dsType(self, vector, type, topk=1):
        if isinstance(vector, np.ndarray):
            vector = torch.from_numpy(vector).float()
        vector = F.normalize(vector, dim=-1)
        if vector.dim() == 1:
            vector = vector.unsqueeze(0)

        if type not in self.groups:
            warnings.warn(f"Type '{type}' not found in the database, returning empty results. We have {list(self.groups.keys())} in the database.")
            return [], []

        group = self.groups[type]
        features_norm = F.normalize(group['features'], p=2, dim=1)
        features_norm = features_norm.to(vector.device)
        sim = torch.matmul(features_norm, vector.T).squeeze()

        _, sorted_indices = torch.sort(sim, descending=True)
        topk_indices = sorted_indices[:topk]
        original_indices = group['indices'].to(vector.device)[topk_indices].tolist()
        result_labels = [self.labels[i] for i in original_indices]
        result_types = [self.types[i] for i in original_indices]
        return result_labels, result_types

    def format_results(self, labels, types):
        group_dict = defaultdict(list)
        ordered_types = []


        seen = set()
        for label, type_name in zip(labels, types):
            if type_name not in seen:
                seen.add(type_name)
                ordered_types.append(type_name)
            group_dict[type_name].append(str(label))

        random.shuffle(ordered_types)
        output = []
        for type_name in ordered_types:
            label_str = " | ".join(group_dict[type_name])
            output.append(f"If task is {type_name}: {label_str}")

        return "\n".join(output)

    def get_search_result(self, vector, type_=None, candidates=None, topk=None):
        topk = int(self.recommend_topk * random.uniform(0.9, 1.1)) if topk is None else topk
        vector = vector.to('cuda', dtype=torch.float32)
        if candidates is not None:
            result_labels, result_types = self.get_most_similar_by_candidate(vector, candidates, topk=topk)
        elif type_ is None:
            result_labels, result_types = self.get_most_similar(vector, topk=topk)
        else:
            result_labels, result_types = self.get_most_similar_by_dsType(vector, type_, topk=topk)
        
        formatted_results = "Following is the Feature Database Search Result, you can consider but it may be wrong: \n" + self.format_results(result_labels, result_types)
        return formatted_results
    
    def get_search_result_from_EEG(self, eeg_model, eeg, type=None,candidates=None,topk=None):
        
        topk = int(self.recommend_topk * random.uniform(0.9, 1.1)) if topk is None else topk
        
        if candidates is not None:
            candidates = None if candidates==[] or len(candidates)<=1 else candidates
        if candidates is not None:
            assert isinstance(candidates, list), "candidates should be a list"
            assert all(isinstance(i, str) for i in candidates), "candidates should be a list of strings"
            assert all(i in self.labels for i in candidates), "candidates should be a list of labels in the database"
            assert topk<len(candidates), f"topk {topk} should be less than candidates {len(candidates)}"

        assert topk<=len(set(self.labels)), f"topk {topk} should be less than types {len(set(self.labels))}"

        original_mode = eeg_model.training
        try:
            eeg_model.eval()
            if isinstance(eeg, np.ndarray):
                eeg = torch.from_numpy(eeg).float()
            eeg=eeg.to(dtype=torch.float32)
            if len(eeg.shape)==2:
                eeg=eeg.unsqueeze(0)

            eeg_model = eeg_model.to('cuda')
            eeg=eeg.to('cuda')
            with torch.no_grad():
                result = eeg_model.forward(eeg)
            vector=result
            search_res=self.get_search_result(vector,type_=type,topk=topk,candidates=candidates,)
            return search_res
        finally:
            # Always restore original training mode, even if exception occurs
            eeg_model.train(original_mode)
    
    def check_model_ok_for_RAG(self,eeg_model):
        if self.check_model:
            return
        original_mode = eeg_model.training
        eeg_model.eval() 
        # print("Check model for RAG")
        with open(self.test_data_path, 'rb') as f:
            eeg_datas,text_data=pickle.load(f)
        eeg_datas=torch.tensor(eeg_datas)
        total_count=0
        correct_count=0
        for i in tqdm(range(len(eeg_datas))):
            eeg=eeg_datas[i]
            cls_label=text_data[i].decode().split(' ')[7]
            eeg=eeg.to(dtype=torch.float32)
            if len(eeg.shape)==2:
                eeg=eeg.unsqueeze(0)
            with torch.no_grad():
                result = eeg_model.forward(eeg)
            eeg_feature=result
            predict_label=self.get_most_similar_by_dsType(eeg_feature,'Emotion Recognition',topk=1)[0][0]
            if predict_label==cls_label:
                correct_count+=1
            total_count+=1
        acc=correct_count/total_count
        if acc<0.4:
            print(acc)
            warnings.warn('Model may not init, please check')
        else:
            print("Pass test for RAG")
        self.check_model=True
        eeg_model.train(original_mode)
