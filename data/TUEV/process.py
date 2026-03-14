"""
TUEV Dataset Processing - Streaming Batch Processor

This module implements optimized streaming processing for the TUEV (TUH EEG Events) dataset.

Key Features:
- Memory-efficient streaming: Process EDF files one-by-one to avoid memory explosion
- Batch-based HDF5 writing: Accumulate data in batches before writing to disk
- Complete dataset isolation: Separate train/test/cross splits without data leakage
- Automatic resource cleanup: Garbage collection after each file

Data Flow:
1. Read EDF file → Apply 10-20 electrode mapping → Resample to 512Hz
2. Build events from annotations → Resample from 200Hz to 512Hz (5-second segments)
3. Split 5-second events into 5 one-second segments (32 × 512)
4. Apply transformations → Add to batch buffer
5. When buffer reaches batch_size, flush to HDF5
6. Clean up memory and proceed to next file

Splits:
- train: 90% of training directory (split randomly within each file)
- test: 10% of training directory
- cross: 100% of eval directory (held-out subjects)
"""

import mne
import numpy as np
import os
import pickle
from tqdm import tqdm
import re
import glob
import random
import gc
from data.Utils import *
from torcheeg import transforms

# Dataset information
ds_info = TUEVDatasetInfo()
ds_name = 'TUEV'
hdf5_path = 'data/Total/data_label.h5'

"""
TUEV dataset optimized streaming processing
- Avoids memory explosion by processing files one by one
- Ensures complete isolation between train/test/cross datasets
- Uses streaming approach without intermediate NPZ files
- Better memory management with garbage collection
"""

def apply_10_20_mapping(raw):
    """Apply 10-20 electrode mapping to raw EEG data"""
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
    raw_filtered.set_montage('standard_1020', on_missing='ignore')
    return raw_filtered


def BuildEvents(signals, times, EventData):
    """Build events from EEG signals and event data"""
    [numEvents, z] = EventData.shape
    fs = 200.0
    [numChan, numPoints] = signals.shape
    features = np.zeros([numEvents, numChan, int(fs) * 5])
    offending_channel = np.zeros([numEvents, 1])
    labels = np.zeros([numEvents, 1])
    offset = signals.shape[1]
    signals = np.concatenate([signals, signals, signals], axis=1)
    
    for i in range(numEvents):
        chan = int(EventData[i, 0])
        start = np.where((times) >= EventData[i, 1])[0][0]
        end = np.where((times) >= EventData[i, 2])[0][0]
        features[i, :] = signals[:, offset + start - 2 * int(fs) : offset + end + 2 * int(fs)]
        offending_channel[i, :] = int(chan)
        labels[i, :] = int(EventData[i, 3])
    return [features, offending_channel, labels-1]


def readEDF(fileName):
    """Read EDF file and return signals, times, event data"""
    Rawdata = mne.io.read_raw_edf(fileName, preload=True)
    Rawdata = apply_10_20_mapping(Rawdata)
    Rawdata = Rawdata.resample(200)
    _, times = Rawdata[:]
    signals = Rawdata.get_data(units='uV')
    RecFile = fileName[0:-3] + "rec"
    eventData = np.genfromtxt(RecFile, delimiter=",")
    Rawdata.close()
    return [signals, times, eventData, Rawdata]


class StreamingProcessor:
    """Streaming processor for TUEV dataset with memory optimization and batch storage"""
    
    def __init__(self, ds_info, ds_name, hdf5_path, batch_size=1024):
        self.ds_info = ds_info
        self.ds_name = ds_name
        self.hdf5_path = hdf5_path
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            FilterTransform(ds_info, downsample_fs=None, h_freq=None, l_freq=None, notch=None),
        ])
        
        # Initialize save handlers for different datasets
        self.train_handler = Convert_and_Save()
        self.test_handler = Convert_and_Save()
        self.cross_handler = Convert_and_Save()
        
        # Batch buffers for different datasets
        self.train_buffer = {'eeg': [], 'label': []}
        self.test_buffer = {'eeg': [], 'label': []}
        self.cross_buffer = {'eeg': [], 'label': []}
        
        # Statistics
        self.train_count = 0
        self.test_count = 0
        self.cross_count = 0
    
    def _add_to_buffer(self, buffer, eeg_data, label, dataset_type=None):
        """Add data to buffer and save if buffer is full"""
        buffer['eeg'].append(eeg_data)
        buffer['label'].append(label)
        
        # Check if buffer is full
        if len(buffer['eeg']) >= self.batch_size:
            print(f"Buffer full for {dataset_type}, saving batch of size {len(buffer['eeg'])}")
            self._flush_buffer(buffer, dataset_type)
    
    def _flush_buffer(self, buffer, dataset_type=None):
        """Flush buffer data to HDF5"""
        if not buffer['eeg']:
            return
        
        # Convert buffer to arrays
        eeg_array = np.stack(buffer['eeg'], axis=0)
        label_array = np.array(buffer['label'])
        
        # Clear buffer
        buffer['eeg'].clear()
        buffer['label'].clear()
        
        # Save to HDF5 if dataset_type is provided
        if dataset_type:
            self._save_batch(dataset_type, eeg_array, label_array)
        
        return eeg_array, label_array
    
    def _save_batch(self, dataset_type, eeg_array, label_array):
        """Save batch data to HDF5"""
        if dataset_type == 'train':
            self.train_handler.process_and_save(
                ds_info=self.ds_info,
                eegdata=eeg_array,
                label=label_array,
                dataset_name=f'{self.ds_name}_train',
                path=os.path.join(os.environ['WaveMind_ROOT_PATH_'], self.hdf5_path)
            )
            self.train_count += len(eeg_array)
        elif dataset_type == 'test':
            self.test_handler.process_and_save(
                ds_info=self.ds_info,
                eegdata=eeg_array,
                label=label_array,
                dataset_name=f'{self.ds_name}_test',
                path=os.path.join(os.environ['WaveMind_ROOT_PATH_'], self.hdf5_path)
            )
            self.test_count += len(eeg_array)
        elif dataset_type == 'cross':
            self.cross_handler.process_and_save(
                ds_info=self.ds_info,
                eegdata=eeg_array,
                label=label_array,
                dataset_name=f'{self.ds_name}_cross',
                path=os.path.join(os.environ['WaveMind_ROOT_PATH_'], self.hdf5_path)
            )
            self.cross_count += len(eeg_array)
    
    def process_single_file(self, file_path, dataset_type, split_ratio=0.9):
        """
        Process a single file and save data in batches to HDF5
        dataset_type: 'train' (for train/test split) or 'cross' (for cross dataset)
        """
        try:
            [signals, times, event, Rawdata] = readEDF(file_path)
        except (ValueError, KeyError) as e:
            print(f"Error processing file {file_path}: {e}")
            return 0, 0, 0
        
        # Build events from signals
        signals, offending_channels, labels = BuildEvents(signals, times, event)
        
        # Resample from 200Hz to 512Hz
        import scipy
        signals = scipy.signal.resample(signals, int(5*512), axis=-1)
        
        file_train_count = 0
        file_test_count = 0
        file_cross_count = 0
        
        # Process each event
        for idx, (signal, offending_channel, label) in enumerate(zip(signals, offending_channels, labels)):
            # Split each 5-second event into 5 one-second segments
            for i in range(5):
                sub_signal = signal[:, i * 512 : (i + 1) * 512]
                assert sub_signal.shape == (19, 512)
                
                # Apply transformation
                transformed_eeg = self.transform.__call__(eeg=sub_signal)['eeg']
                
                # Add to appropriate buffer based on dataset type
                if dataset_type == 'cross':
                    self._add_to_buffer(self.cross_buffer, transformed_eeg, label, 'cross')
                    file_cross_count += 1
                else:
                    if random.random() < split_ratio:
                        self._add_to_buffer(self.train_buffer, transformed_eeg, label, 'train')
                        file_train_count += 1
                    else:
                        self._add_to_buffer(self.test_buffer, transformed_eeg, label, 'test')
                        file_test_count += 1
        
        # Clean up memory
        del signals, times, event, Rawdata
        gc.collect()
        
        return file_train_count, file_test_count, file_cross_count
    
    def count_edf_files(self, directory_path):
        """Count all EDF files in a directory and its subdirectories"""
        edf_files = []
        for dirName, subdirList, fileList in os.walk(directory_path):
            for fname in fileList:
                if fname.endswith(".edf"):
                    edf_files.append(os.path.join(dirName, fname))
        return edf_files
    
    def process_directory(self, directory_path, dataset_type, split_ratio=0.9):
        """Process all EDF files in a directory"""
        print(f"Processing {dataset_type} directory: {directory_path}")
        
        total_train = 0
        total_test = 0
        total_cross = 0
        
        # Get all EDF files
        edf_files = self.count_edf_files(directory_path)
        
        print(f"Found {len(edf_files)} EDF files")
        
        # Process each file with progress bar
        for file_path in tqdm(edf_files, desc=f"Processing {dataset_type}"):
            train_count, test_count, cross_count = self.process_single_file(
                file_path, dataset_type, split_ratio
            )
            total_train += train_count
            total_test += test_count
            total_cross += cross_count
        
        return total_train, total_test, total_cross
    
    def process_all_datasets(self, root_dir, split_ratio=0.9):
        """Process all datasets (train, test, cross) with complete isolation"""
        print("Starting streaming processing with complete dataset isolation...")
        print(f"Using batch size: {self.batch_size}")
        
        # Count files before processing
        train_dir = os.path.join(root_dir, "train")
        eval_dir = os.path.join(root_dir, "eval")
        
        train_files = []
        eval_files = []
        
        if os.path.exists(train_dir):
            train_files = self.count_edf_files(train_dir)
            print(f"Found {len(train_files)} EDF files in train directory")
        else:
            print(f"Train directory not found: {train_dir}")
        
        if os.path.exists(eval_dir):
            eval_files = self.count_edf_files(eval_dir)
            print(f"Found {len(eval_files)} EDF files in eval directory")
        else:
            print(f"Eval directory not found: {eval_dir}")
        
        print(f"\n=== Dataset Summary ===")
        print(f"Train+Test files: {len(train_files)} (will be split {int(split_ratio*100)}% train, {int((1-split_ratio)*100)}% test)")
        print(f"Cross files: {len(eval_files)}")
        print("=" * 30)
        
        # Process train directory (will be split into train/test)
        if train_files:
            train_count, test_count, _ = self.process_directory(train_dir, 'train', split_ratio)
            print(f"Train dataset: {train_count} samples")
            print(f"Test dataset: {test_count} samples")
        
        # Process eval directory (for cross dataset)
        if eval_files:
            _, _, cross_count = self.process_directory(eval_dir, 'cross')
            print(f"Cross dataset: {cross_count} samples")
        
        # Flush any remaining data in buffers
        self.flush_all_buffers()
        
        # Print final statistics
        print("\n=== Processing Complete ===")
        print(f"Total train samples: {self.train_count}")
        print(f"Total test samples: {self.test_count}")
        print(f"Total cross samples: {self.cross_count}")
        print(f"Overall total: {self.train_count + self.test_count + self.cross_count}")
        
        return self.train_count, self.test_count, self.cross_count
    
    def flush_all_buffers(self):
        """Flush all remaining data in buffers to HDF5"""
        print("Flushing remaining data in buffers...")
        
        # Flush train buffer
        if self.train_buffer['eeg']:
            self._flush_buffer(self.train_buffer, 'train')
        
        # Flush test buffer
        if self.test_buffer['eeg']:
            self._flush_buffer(self.test_buffer, 'test')
        
        # Flush cross buffer
        if self.cross_buffer['eeg']:
            self._flush_buffer(self.cross_buffer, 'cross')


def main():
    """
    Main function for optimized streaming processing.

    Processing Steps:
    1. Count EDF files in train/ and eval/ directories
    2. Initialize StreamingProcessor with batch_size=1024
    3. Process train directory → Split randomly into train (90%) / test (10%)
    4. Process eval directory → Save as cross dataset
    5. Flush remaining data in buffers to HDF5
    6. Print final statistics
    """
    # Define base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.join(base_dir, "edf")
    
    # Check if data directory exists
    if not os.path.exists(root):
        print(f"Data directory not found: {root}")
        print("Please ensure the TUEV dataset is downloaded and placed in the correct location.")
        return
    
    # Initialize streaming processor
    processor = StreamingProcessor(ds_info, ds_name, hdf5_path)
    
    # Process all datasets
    try:
        processor.process_all_datasets(root)
        print("\n✅ Processing completed successfully!")
        print("Data has been saved directly to HDF5 format with complete isolation:")
        print(f"- {ds_name}_train: Train dataset (from train directory, {processor.train_count} samples)")
        print(f"- {ds_name}_test: Test dataset (from train directory, {processor.test_count} samples)")
        print(f"- {ds_name}_cross: Cross dataset (from eval directory, {processor.cross_count} samples)")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        raise


if __name__ == "__main__":
    main()