# https://github.com/wjq-learning/CBraMod/blob/main/preprocessing/preprocessing_tuab.py

import os
import re
import glob
import shutil
from torcheeg import transforms
import numpy as np
import mne
import scipy.signal
from tqdm import tqdm

# Import Convert_and_Save class
from data.Utils import *



hdf5_path = "data/Total/data_label.h5"



def apply_10_20_mapping(raw):
    electrode_mapping = {
        'FP1': 'Fp1', 'FP2': 'Fp2', 
        'T3': 'T7', 'T4': 'T8', 
        'T5': 'P7', 'T6': 'P8', 
        'FZ': 'Fz', 'CZ': 'Cz', 'PZ': 'Pz' 
    }

    eeg_chs = raw.copy().pick_types(eeg=True)
    selected_chs, rename_dict = [], {}
    for ch_name in eeg_chs.ch_names:

        match = re.match(r'^EEG ([A-Z0-9]+)-REF$', ch_name, re.IGNORECASE)
        if match:
            electrode = match.group(1).upper()
            if electrode in electrode_mapping or electrode in {'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7',
                                                               'F8'}:
                selected_chs.append(ch_name)
                new_name = electrode_mapping.get(electrode, electrode)
                rename_dict[ch_name] = new_name

    raw_filtered = raw.copy().pick_channels(selected_chs)
    raw_filtered.rename_channels(rename_dict)

    raw_filtered.set_montage(
        'standard_1020',
        on_missing='ignore'
    )

    return raw_filtered


def split_and_dump(params):
    """
    Process a single EDF file: apply filters, remap electrodes, resample, and split into 1-second segments.

    Args:
        params: Tuple of (fetch_folder, subject_id, dump_folder, label)

    Returns:
        None (saves processed segments to disk as .npz files)
    """
    fetch_folder, sub, dump_folder, label = params
    for file in os.listdir(fetch_folder):
        if sub in file:
            file_path = os.path.join(fetch_folder, file)

            raw = mne.io.read_raw_edf(file_path, preload=True)
            raw = raw.filter(0.1, None, fir_design='firwin')
            raw = raw.notch_filter(50, fir_design='firwin')
            raw = raw.notch_filter(60, fir_design='firwin')
            raw = apply_10_20_mapping(raw)
            raw = raw.resample(512)
            

            raw_data = raw.get_data(units='uV')
            
            
            # Split into 1-second segments (512 samples)
            for i in range(raw_data.shape[-1] // 512):
                sub_signal = raw_data[:, i * 512 : (i + 1) * 512]
                dump_path = os.path.join(
                    dump_folder, file.split(".")[0] + "_" + str(i) + ".npz"
                )
                np.savez(dump_path, signal=sub_signal, label=label)
                        
            


if __name__ == "__main__":
    """
    TUAB dataset processing - Stage 2 only.
    Stage 1 (data splitting) has already been completed.

    TUAB dataset is downloaded from https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml
    """

    # Stage 2: Load pre-split data and convert to HDF5
    root = f"{os.environ['WaveMind_ROOT_PATH_']}/data/TUAB/process_refine"
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
                        path=os.path.join(os.environ['WaveMind_ROOT_PATH_'], hdf5_path)
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
    process_refine_dir = os.path.join(os.environ['WaveMind_ROOT_PATH_'], 'data/TUAB/process_refine')
    if os.path.exists(process_refine_dir):
        print(f"Cleaning up temporary directory: {process_refine_dir}")
        shutil.rmtree(process_refine_dir)

    print("TUAB dataset processing complete!")
    
    
    
    
    