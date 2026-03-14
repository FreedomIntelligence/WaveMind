import os
from torcheeg.datasets import SEEDDataset
from torcheeg import transforms
from torcheeg.model_selection import train_test_split_cross_subject, train_test_split
from data.Utils import load_in_memory, Convert_and_Save, SEEDDatasetInfo, get_each_type_caption_feature


def get_each_type_caption_feature(ds_info):
    """
    Generate CLIP text embeddings for all emotion labels in SEED dataset.

    Args:
        ds_info: SEEDDatasetInfo instance

    Returns:
        np.ndarray of shape (num_classes, 768) - CLIP text embeddings
    """
    from transformers import CLIPTextModelWithProjection, AutoTokenizer

    text_model = CLIPTextModelWithProjection.from_pretrained('openai/clip-vit-large-patch14-336')
    text_model.to('cuda')
    text_tokenizer = AutoTokenizer.from_pretrained('openai/clip-vit-large-patch14-336')
    text_features = []
    for type in range(ds_info.id_min,ds_info.id_max+1):
        print(type,ds_info.get_caption(type))
        text_inputs = text_tokenizer(ds_info.get_caption(type), padding=True, return_tensors="pt").to('cuda')
        text_feature = text_model(**text_inputs).text_embeds
        text_features.append(text_feature.detach().cpu().numpy())
        assert text_feature.shape[-1]==768

    del text_model,text_tokenizer
    return np.concatenate(text_features,axis=0)


if __name__ =="__main__":
    """
    SEED dataset preprocessing: Load, filter, split, and save to HDF5.

    Splits:
    - train_test_split_cross_subject: Splits by subject for cross-subject validation
    - train_test_split: Further splits training data into train/val
    """
    ds_name='SEED'
    iopath=f'~/.cache/torcheeg/{ds_name}'
    ds_info=SEEDDatasetInfo()
    dataset = SEEDDataset(
        io_path=iopath,
        root_path='./Preprocessed_EEG',
        offline_transform=transforms.Compose([
            FilterTransform(ds_info, downsample_fs=512,h_freq=None),
        ]),
        online_transform=transforms.Compose([
        ]),
        label_transform=transforms.Compose([
                transforms.Select('emotion'),
                transforms.Lambda(lambda x: x+1),
            ]),
        # 1 -1 0
        num_worker=16)


    # Split dataset: cross-subject validation, then train/val split
    train_val_dataset, test_dataset = train_test_split_cross_subject(dataset=dataset)
    train_dataset, val_dataset = train_test_split(dataset=train_val_dataset)

    print(f"Dataset splits: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    del dataset,train_val_dataset

    # Process and save train dataset
    train_item = Convert_and_Save()
    eeg_all_train, label_all_train=load_in_memory(train_dataset,cut=False)
    print(f"Train data shape: EEG={eeg_all_train.shape}, labels={label_all_train.shape}")
    train_item.process_and_save(ds_info = ds_info,eegdata=eeg_all_train,label=label_all_train,dataset_name=f'{ds_name}_train',path=os.path.join(os.environ['WaveMind_ROOT_PATH_'],'data/Total/data_label.h5'))
    del train_item,eeg_all_train,label_all_train,train_dataset

    # Process and save validation dataset
    test_item = Convert_and_Save()
    eeg_all_val,label_all_val=load_in_memory(val_dataset,cut=False)
    print(f"Validation data shape: EEG={eeg_all_val.shape}, labels={label_all_val.shape}")
    test_item.process_and_save(ds_info = ds_info,eegdata=eeg_all_val,label=label_all_val,dataset_name=f'{ds_name}_test',path=os.path.join(os.environ['WaveMind_ROOT_PATH_'],'data/Total/data_label.h5'))

    del test_item,eeg_all_val,label_all_val,val_dataset

    # Process and save cross-subject dataset
    cross_sub = Convert_and_Save()
    eeg_all_test,label_all_test=load_in_memory(test_dataset,cut=False)
    print(f"Cross data shape: EEG={eeg_all_test.shape}, labels={label_all_test.shape}")
    cross_sub.process_and_save(ds_info = ds_info,eegdata=eeg_all_test,label=label_all_test,dataset_name=f'{ds_name}_cross',path=os.path.join(os.environ['WaveMind_ROOT_PATH_'],'data/Total/data_label.h5'))
    del cross_sub,eeg_all_test,label_all_test,test_dataset

    print("SEED dataset processing complete!")
    print(f"Note: CLIP groundtruth files are generated via data/create_dataset_pkl.py")
    


    
    
    


    

    
