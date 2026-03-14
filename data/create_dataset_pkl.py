


import os
import pickle
import torch
import numpy as np
import torch.nn.functional as F
from data.Utils import get_each_type_caption_feature
root_path=os.environ['WaveMind_ROOT_PATH_']

groundTruth_dir=os.path.join(os.environ['WaveMind_ROOT_PATH_']  , 'data/Total/CLIP_groundTruth')

def get_and_save_info_TUAB():
    from data.Utils import TUABDatasetInfo
    ds_name='TUAB'
    label_dict=TUABDatasetInfo().id2label()
    sorted_labels = [label_dict[key] for key in sorted(label_dict.keys()) if key >= 0]
    with open(f'{root_path}/data/Total/CLIP_groundTruth/{ds_name}.pkl', 'wb') as f:
        pickle.dump(sorted_labels, f)

    np.save(os.path.join(groundTruth_dir,f'{ds_name}.npy'),get_each_type_caption_feature(TUABDatasetInfo()))
        
def get_and_save_info_ImageNetEEG():
    from data.Utils import ImageNetEEGDatasetInfo
    ds_name='ImageNetEEG'
    label_dict=ImageNetEEGDatasetInfo().id2label()
    sorted_labels = [label_dict[key] for key in sorted(label_dict.keys()) if key >= 0]
    with open(f'{root_path}/data/Total/CLIP_groundTruth/{ds_name}.pkl', 'wb') as f:
        pickle.dump(sorted_labels, f)
        
def get_and_save_info_TUEV():
    from data.Utils import TUEVDatasetInfo
    ds_name='TUEV'
    label_dict=TUEVDatasetInfo().id2label()
    sorted_labels = [label_dict[key] for key in sorted(label_dict.keys()) if key >= 0]
    with open(f'{root_path}/data/Total/CLIP_groundTruth/{ds_name}.pkl', 'wb') as f:
        pickle.dump(sorted_labels, f)
    np.save(os.path.join(groundTruth_dir,f'{ds_name}.npy'),get_each_type_caption_feature(TUEVDatasetInfo()))


def get_and_save_info_THNING_closeset():
    from EEG_Encoder.Tools.dataBuilder import CLIPDataset_ThingEEG
    ds_dataset = CLIPDataset_ThingEEG(train=True,train_Val_Same_set='none',model_type='ViT-L-14-336',subjects=['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08','sub-09','sub-10'])
    from tqdm import tqdm
    db={}

    for i in tqdm(range(len(ds_dataset))):
        sample_data=ds_dataset[i]
        data_class=sample_data['img_path'].split('/')[-1].split('_')[0]
        data_feature=sample_data['img_features']
        if data_class not in db:
            db[data_class] = []
        db[data_class].append(data_feature)
    for key in tqdm(db):
        db[key] = np.array(db[key])
        db[key] = np.mean(db[key],axis=0)
        db[key] = torch.from_numpy(db[key])
        db[key] = db[key].unsqueeze(0)
        db[key] = F.normalize(db[key],p=2,dim=-1)
        db[key] = db[key].squeeze(0)
    feature_data=[]
    class_label=[]
    for key in tqdm(db):
        feature_data.append(db[key])
        class_label.append(key)
        
    feature_data=np.array(feature_data).astype(np.float16)

    np.save(f'{root_path}/data/Total/CLIP_groundTruth/thingEEG_closeset.npy',feature_data)
    with open(f'{root_path}/data/Total/CLIP_groundTruth/thingEEG_closeset.pkl', 'wb') as f:
        pickle.dump(class_label, f)
        
def get_and_save_info_THNING_zero_shot():
    from EEG_Encoder.Tools.dataBuilder import CLIPDataset_ThingEEG
    ds_dataset = CLIPDataset_ThingEEG(train=False,train_Val_Same_set='none',model_type='ViT-L-14-336',subjects=['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08','sub-09','sub-10'])
    from tqdm import tqdm
    db={}

    for i in tqdm(range(len(ds_dataset))):
        sample_data=ds_dataset[i]
        data_class=sample_data['img_path'].split('/')[-1].split('_')[0]
        data_feature=sample_data['img_features']
        if data_class not in db:
            db[data_class] = []
        db[data_class].append(data_feature)
    for key in tqdm(db):
        db[key] = np.array(db[key])
        db[key] = np.mean(db[key],axis=0)
        db[key] = torch.from_numpy(db[key])
        db[key] = db[key].unsqueeze(0)
        db[key] = F.normalize(db[key],p=2,dim=-1)
        db[key] = db[key].squeeze(0)
    feature_data=[]
    class_label=[]
    for key in tqdm(db):
        feature_data.append(db[key])
        class_label.append(key)
        
    feature_data=np.array(feature_data).astype(np.float16)

    np.save(f'{root_path}/data/Total/CLIP_groundTruth/thingEEG.npy',feature_data)
    with open(f'{root_path}/data/Total/CLIP_groundTruth/thingEEG.pkl', 'wb') as f:
        pickle.dump(class_label, f)
        

def get_and_save_info_CHB_MIT():
    from data.Utils import CHBMITDatasetInfo
    ds_name='CHB-MIT'
    label_dict=CHBMITDatasetInfo().id2label()
    sorted_labels = [label_dict[key] for key in sorted(label_dict.keys()) if key >= 0]
    with open(f'{root_path}/data/Total/CLIP_groundTruth/{ds_name}.pkl', 'wb') as f:
        pickle.dump(sorted_labels, f)   

def get_and_save_info_Siena():
    from data.Utils import SienaDatasetInfo
    ds_name='Siena'
    label_dict=SienaDatasetInfo().id2label()
    sorted_labels = [label_dict[key] for key in sorted(label_dict.keys()) if key >= 0]
    with open(f'{root_path}/data/Total/CLIP_groundTruth/{ds_name}.pkl', 'wb') as f:
        pickle.dump(sorted_labels, f) 


def get_and_save_info_SEED():
    from data.Utils import SEEDDatasetInfo
    ds_name='SEED'
    label_dict=SEEDDatasetInfo().id2label()
    sorted_labels = [label_dict[key] for key in sorted(label_dict.keys()) if key >= 0]
    with open(f'{root_path}/data/Total/CLIP_groundTruth/{ds_name}.pkl', 'wb') as f:
        pickle.dump(sorted_labels, f)

    np.save(os.path.join(groundTruth_dir,f'{ds_name}.npy'),get_each_type_caption_feature(SEEDDatasetInfo()))  


if __name__ == '__main__':
    get_and_save_info_TUAB()
    get_and_save_info_ImageNetEEG()
    get_and_save_info_TUEV()
    get_and_save_info_THNING_closeset()
    get_and_save_info_THNING_zero_shot()
    get_and_save_info_SEED()
    
    # we are working on CHB-MIT and Siena now
    # get_and_save_info_CHB_MIT()
    # get_and_save_info_Siena()
    
    
