# https://github.com/wjq-learning/CBraMod/blob/main/preprocessing/preprocessing_tuab.py

import os
import glob
import shutil
from torcheeg import transforms
import numpy as np
from tqdm import tqdm

from data.Utils import TUABDatasetInfo, FilterTransform, Convert_and_Save, get_wavemind_root

hdf5_path = "data/Total/data_label.h5"

if __name__ == "__main__":

    """
    TUAB dataset processing - Stage 2 only.
    Stage 1 (data splitting) has already been completed.

    TUAB dataset is downloaded from https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml
    """

    # Stage 2: Load pre-split data and convert to HDF5
    root = f"{get_wavemind_root()}/data/TUAB/process_refine"
    train_dump_folder = os.path.join(root, "train")
    val_dump_folder = os.path.join(root, "val")
    test_dump_folder = os.path.join(root, "test")
    
    # glob npy in train, val, test
    train_files = glob.glob(os.path.join(train_dump_folder, "*.npz"))
    val_files = glob.glob(os.path.join(val_dump_folder, "*.npz"))
    test_files = glob.glob(os.path.join(test_dump_folder, "*.npz"))
    
    # random get 20% sample and random
    train_files = np.random.choice(train_files, int(len(train_files) * 0.2), replace=False)
    val_files = np.random.choice(val_files, int(len(val_files) * 0.2), replace=False)
    test_files = np.random.choice(test_files, int(len(test_files) * 0.2), replace=False)
    
    
    print('Done split,start convert and save!')
    
    
    train_item = Convert_and_Save()
    test_item = Convert_and_Save()
    cross_sub = Convert_and_Save()
    
    # Create dataset info object and define HDF5 path
    ds_info = TUABDatasetInfo()
    trans = transforms.Compose([
        FilterTransform(ds_info, downsample_fs=None,h_freq=None, l_freq=None, notch=None),
    ])

    
    
    # Batch processing function, save every 10000 signals
    def process_in_batches(files, convert_save_obj, dataset_type, batch_size=10000):
        eegs_batch = []
        labels_batch = []
        
        for i, file in enumerate(tqdm(files)):
            data = np.load(file)
            eeg,label=data['signal'],data['label']
            eeg=trans.__call__(eeg=eeg)['eeg']
            eegs_batch.append(eeg)
            labels_batch.append(label)
            
            # Process every batch_size samples
            if len(eegs_batch) >= batch_size or i == len(files) - 1:
                if len(eegs_batch) > 0:
                    eeg_data = np.array(eegs_batch)
                    label_data = np.array(labels_batch)
                    
                    # Call process_and_save method
                    convert_save_obj.process_and_save(
                        ds_info=ds_info,
                        eegdata=eeg_data,
                        label=label_data,
                        dataset_name=f'TUAB_{dataset_type}',
                        path=os.path.join(get_wavemind_root(), hdf5_path)
                    )
                    
                    # Clear batch
                    eegs_batch = []
                    labels_batch = []
    
    # Process train dataset
    print("Processing train files...")
    process_in_batches(train_files, train_item, 'train')
    
    # Process val dataset
    print("Processing val files...")
    process_in_batches(val_files, test_item, 'test')
    
    # Process test dataset
    print("Processing test files...")
    process_in_batches(test_files, cross_sub, 'cross')

    # Clean up temporary directory
    process_refine_dir = os.path.join(get_wavemind_root(), 'data/TUAB/process_refine')
    if os.path.exists(process_refine_dir):
        print(f"Cleaning up temporary directory: {process_refine_dir}")
        shutil.rmtree(process_refine_dir)

    print("TUAB dataset processing complete!")
    
    
    
    
    