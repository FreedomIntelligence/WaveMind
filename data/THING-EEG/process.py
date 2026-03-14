"""
THING-EEG Dataset Processing

This module processes the THING-EEG dataset which contains EEG recordings of participants
viewing visual stimuli from the THINGS object concept database.

Data Splits:
- train: Subjects 1-9, closed-set 1,573 object categories (training data)
- test: Subjects 1-9, closed-set 1,573 object categories (test data)
- cross: Subject 10, 200 zero-shot object categories (held-out subject)

Dataset Source: https://huggingface.co/datasets/LidongYang/EEG_Image_decode

HDF5 Keys Generated:
- thingEEG_train
- thingEEG_test
- thingEEG_cross
"""

import glob
import numpy as np
import os
from data.Utils import *
from EEG_Encoder.Tools.dataBuilder import CLIPDataset_ThingEEG
import gc


ds_name = 'THING-EEG'
hdf5_path=f'{os.environ["WaveMind_ROOT_PATH_"]}/data/Total/data_label.h5'

def get_the_data(dataset):
    """
    Extract EEG data and features from THING-EEG dataset.

    Args:
        dataset: CLIPDataset_ThingEEG instance

    Returns:
        Tuple of (eegs, img_features, labels, captions, image_paths)
            - eegs: np.ndarray of shape (N, 32, 512)
            - img_features: np.ndarray of shape (N, 768) - CLIP image embeddings
            - labels: np.ndarray of shape (N,) - integer labels
            - captions: List[str] - text descriptions
            - image_paths: List[str] - relative paths to images
    """
    # dict_keys(['eeg_data', 'label', 'text', 'text_features', 'img_path', 'img_features', 'dataset_name'])
    eegs,img_features,labels,captions,image_paths=[],[],[],[],[]
    for i in range(len(dataset)):
        eegs.append(dataset[i]['eeg_data'])
        img_features.append(dataset[i]['img_features'])
        labels.append(dataset[i]['label'])
        captions.append(dataset[i]['text'])
        image_paths.append(dataset[i]['img_path'])
    assert len(eegs) == len(img_features) == len(labels) == len(captions) == len(image_paths), "Number of samples in eegs, img_features, labels, captions, and image_paths must match."
    img_features=np.stack(img_features).squeeze()
    eegs=np.stack(eegs).squeeze()
    labels=np.stack(labels).squeeze()
        
    return eegs,img_features,labels,captions,image_paths

# Train logic
train_dataset = CLIPDataset_ThingEEG(train=True,model_type='ViT-L-14-336',subjects=['sub-01', 'sub-02', 'sub-05', 'sub-04', 'sub-03', 'sub-06', 'sub-07', 'sub-08', 'sub-09'],use_aug=False)
train_eegs,train_img_features,train_labels,train_captions,train_image_paths=get_the_data(train_dataset)

# Use Convert_and_Save for train data
train_item = Convert_and_Save()
train_item.save_to_hdf5_new(
    eegdata=train_eegs, 
    text_feature=train_img_features, 
    caption=train_captions, 
    label=train_labels, 
    dataset_name='thingEEG_train', 
    hdf_path=hdf5_path,
    image_paths=train_image_paths
)

del train_dataset, train_eegs,train_img_features,train_labels,train_captions,train_image_paths
gc.collect()


# Test logic
test_dataset = CLIPDataset_ThingEEG(train=False,model_type='ViT-L-14-336',subjects=['sub-01', 'sub-02', 'sub-05', 'sub-04', 'sub-03', 'sub-06', 'sub-07', 'sub-08', 'sub-09'],use_aug=False)
test_eegs,test_img_features,test_labels,test_captions,test_image_paths=get_the_data(test_dataset)

# Use Convert_and_Save for test data
test_item = Convert_and_Save()
test_item.save_to_hdf5_new(
    eegdata=test_eegs, 
    text_feature=test_img_features, 
    caption=test_captions, 
    label=test_labels, 
    dataset_name='thingEEG_test', 
    hdf_path=hdf5_path,
    image_paths=test_image_paths
)

del test_dataset, test_eegs,test_img_features,test_labels,test_captions,test_image_paths
gc.collect()


# Cross logic
cross_dataset = CLIPDataset_ThingEEG(train=False,model_type='ViT-L-14-336',subjects=['sub-10'],use_aug=False)
cross_eegs,cross_img_features,cross_labels,cross_captions,cross_image_paths=get_the_data(cross_dataset)

# Use Convert_and_Save for cross data
cross_item = Convert_and_Save()
cross_item.save_to_hdf5_new(
    eegdata=cross_eegs, 
    text_feature=cross_img_features, 
    caption=cross_captions, 
    label=cross_labels, 
    dataset_name='thingEEG_cross', 
    hdf_path=hdf5_path,
    image_paths=cross_image_paths
)

del cross_dataset, cross_eegs,cross_img_features,cross_labels,cross_captions,cross_image_paths
gc.collect()