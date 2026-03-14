import json
import random
import re
from typing import Optional
import warnings
from EEG_Encoder.Tools.Utils import EEGNoiseGenerator
from EEG_Encoder.Tools.baseDataset import BaseEEGDataset
import h5py
import h5py.h5f as h5f
import numpy as np
import torch
from PIL import Image
from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from tqdm import tqdm
from transformers import CLIPVisionModelWithProjection, CLIPTextModelWithProjection, AutoTokenizer, CLIPImageProcessor
import os
from data.Utils import safe_barrier
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_warn

import torch
import shutil
from collections import Counter


# Lambda function to decode bytes and clean strings
decode_and_clean = lambda value: (
    value.decode('utf-8').strip() if isinstance(value, (bytes, np.bytes_))
    else value.strip() if isinstance(value, str)
    else str(value).strip()
)



class CLIPDataset(BaseEEGDataset):
    def __init__(self, hdf5_file_path, mode, ground_truth_dir=None, dataset_name='all',
                 exclude_dataset=['Siena', 'HMC'], augmentation_config=None, use_aug=None, limit_sample=None,
                 preload_metadata=True, aug_verbose=False):

        # Current support dataset_name: 'all' or list of dataset names or single dataset name
        # single dataset name: thingEEG, ImageNetEEG, TUEV, TUAB,SEED

        # Call base class __init__ to initialize common attributes
        super().__init__(
            augmentation_config=augmentation_config,
            use_aug=use_aug,
            sample_rate=512,  # CLIPDataset default
            limit_sample=limit_sample,
            float_type='float32',
            aug_verbose=aug_verbose
        )

        # Initialize parameters
        if exclude_dataset is None:
            exclude_dataset = []
        self.hdf5_file_path = hdf5_file_path


        rank_zero_info('--------------------------------')
        rank_zero_info(f"Dataset Info: Using HDF5 file: {hdf5_file_path}")
        rank_zero_info(f"Dataset Info: Init Mode: {mode}")
        rank_zero_info('--------------------------------')


        self.ground_truth_dir = ground_truth_dir if ground_truth_dir is not None else os.path.join(
            os.environ['WaveMind_ROOT_PATH_'], 'data/Total/CLIP_groundTruth')
        self.mode = mode
        self.dataset_name = dataset_name
        self._file = None
        self.exclude_dataset = exclude_dataset
        self.preload_metadata = preload_metadata  # New: preload metadata flag

        assert mode in ['train', 'test', 'cross']

        if self.limit_sample is None and mode in ['test', 'cross']:
            self.limit_sample = int(2e3)

        # Initialize metadata and preload if requested
        self._initialize_metadata()
        self.ground_truth = self._load_ground_truth()
        
        if self.__len__() == 0 and mode in ['test', 'train']:
            raise ValueError("No samples found in the dataset")
        
        
        self.is_in_shm = "/dev/shm" in hdf5_file_path
        if self.is_in_shm:
            rank_zero_info("Using shared memory for HDF5 file")
        else:
            rank_zero_info("Using local HDF5 file")
        
        # Preload dataset references if enabled
        self.group_refs = {}
        if self.preload_metadata:
            for ds_name in self.used_datasets:
                self.group_refs[ds_name] = self.file[ds_name]
                
        

    def _initialize_metadata(self):
        # Use a context manager with optimized settings
        with h5py.File(self.hdf5_file_path, 'r', libver='latest', swmr=True,
                       rdcc_nbytes=0, rdcc_nslots=1) as temp_file:
            self.total_datasets = list(temp_file.keys())
            self.used_datasets = []
            self.index_to_dataset_entry = []

            if self.dataset_name == 'all':
                ds_count = 0
                for ds_name in self.total_datasets:
                    if ds_name.split('_')[0] in self.exclude_dataset:
                        continue
                    if ds_name.endswith(self.mode):
                        ds = temp_file[ds_name]
                        self._add_dataset(ds_name, ds)
                        ds_count += 1
                rank_zero_info(f"Using {ds_count} datasets for {self.mode} mode")
            elif isinstance(self.dataset_name, list):
                for original_ds_name in self.dataset_name:
                    ds_name = f"{original_ds_name}_{self.mode}"
                    if ds_name in self.total_datasets:
                        ds = temp_file[ds_name]
                        self._add_dataset(ds_name, ds)
                    else:
                        warnings.warn(f"Dataset '{original_ds_name}' not found in {self.mode} mode, skipping. Available datasets: {self.total_datasets}")
            else:
                target_name = f"{self.dataset_name}_{self.mode}"
                if target_name in temp_file:
                    self._add_dataset(target_name, temp_file[target_name])
                else:
                    warnings.warn(f"Dataset {target_name} not found, available: {self.total_datasets}")

            self.length = len(self.index_to_dataset_entry)
            rank_zero_info(f"Total len of datasets: {self.length}")
            rank_zero_info(f"Total datasets: {len(self.total_datasets)}, Used: {self.used_datasets}")

    def _add_dataset(self, ds_name, dataset):
        self.used_datasets.append(ds_name)
        if self.limit_sample is None:
            self.index_to_dataset_entry.extend([(ds_name, i) for i in range(len(dataset))])
        else:
            sample_size = min(len(dataset), int(self.limit_sample))
            rank_zero_info(f"Dataset {ds_name} has {len(dataset)} samples, using {sample_size} samples in {self.mode} mode")
            if self.mode in ['test', 'cross']:
                indices = random.sample(range(len(dataset)), sample_size)
                self.index_to_dataset_entry.extend([(ds_name, i) for i in indices])
            else:
                self.index_to_dataset_entry.extend([(ds_name, i) for i in range(sample_size)])

    @property
    def file(self):
        if self._file is None:
            self._file = h5py.File(
                self.hdf5_file_path,
                'r',
                libver='latest',
                swmr=True,
                rdcc_nbytes=0 if self.is_in_shm else 500 * 1024**2,  # Disable cache for /dev/shm, else 500MB
                rdcc_nslots=1 if self.is_in_shm else 10000,
                rdcc_w0=0
            )
        return self._file

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure proper cleanup"""
        self.close()

    def _load_ground_truth(self):
        gt = {}
        assert self.ground_truth_dir is not None, "Ground truth directory not provided"
        for ds_name in self.used_datasets:
            base_name = ds_name.split('_')[0]
            path = os.path.join(self.ground_truth_dir, f"{base_name}.npy")
            
            if not os.path.exists(path):
                raise FileNotFoundError(f"Ground truth missing: {path}")
                
            data = torch.from_numpy(np.load(path)).float()
            gt[base_name] = data
        return gt

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        try:
            ds_name, entry_idx = self.index_to_dataset_entry[idx]
        except IndexError:
            raise IndexError(f"Index {idx} out of range, dataset length: {self.length}")
        
        # Use preloaded group reference if available
        if self.preload_metadata and ds_name in self.group_refs:
            group = self.group_refs[ds_name]
            result = group[entry_idx]
        else:
            result = self.file[ds_name][entry_idx]
        
        img_path = ''
        if len(result) == 4:
            eeg_data, text_feat, text, label = result
        else:
            eeg_data, text_feat, text, label, img_path = result
            img_path = decode_and_clean(img_path)
            
        text = decode_and_clean(text)

        # Defensive handling for byte-encoded labels
        if isinstance(label, (bytes, np.bytes_)):
            label = label.decode('utf-8').strip()
            # Try to convert to numeric if possible
            try:
                label = int(label)
            except ValueError:
                # If label is text like "Sleep stage N2", we cannot convert it
                # This should not happen in properly preprocessed data
                raise ValueError(
                    f"Label is a text string '{label}' that cannot be converted to int. "
                    f"Dataset: {ds_name}, Index: {entry_idx}. "
                    f"The HDF5 file may have corrupted label fields."
                )

        if np.isscalar(label):
            # int, np.int*, np.float*, Python float, etc.
            label = [int(label)]
        else:
            assert isinstance(label, (list, np.ndarray)), f"Invalid label type: {type(label)}"
            assert len(label) == 1, f"Label length should be 1, got {len(label)}"
            # Decode first element if it's bytes
            label_value = label[0]
            if isinstance(label_value, (bytes, np.bytes_)):
                label_value = label_value.decode('utf-8').strip()
                try:
                    label_value = int(label_value)
                except ValueError:
                    raise ValueError(
                        f"Label is a text string '{label_value}' that cannot be converted to int. "
                        f"Dataset: {ds_name}, Index: {entry_idx}. "
                        f"The HDF5 file may have corrupted label fields."
                    )
            label = [int(label_value)]



        # Resample if needed
        if eeg_data.shape[-1] != self.sample_rate:
            from scipy.signal import resample
            eeg_data = resample(eeg_data, self.sample_rate, axis=-1)

        eeg_data = torch.from_numpy(eeg_data).float()
        text_feat = torch.from_numpy(text_feat).float()


        if torch.sum(text_feat) == 0:
            raise ValueError(f"All-zero text features at index {idx} in {ds_name}")


        # Validate EEG data before augmentation
        if not self._validate_eeg_data(eeg_data):
            warnings.warn(f"Invalid data at index {idx} in {ds_name}")
            return self.__getitem__((idx + 1) % len(self))  # Wrap around

        # Apply augmentation using base class method
        is_train = (self.mode == 'train')
        eeg_data = self._apply_augmentation(eeg_data, is_train)

        # Validate after augmentation
        if not self._validate_eeg_data(eeg_data):
            warnings.warn(f"Invalid data after augmentation at index {idx}")
            return self.__getitem__((idx + 1) % len(self))



        return self._standardize_return_dict(
            eeg_data=eeg_data,
            label=label,
            text=text,
            text_features=text_feat,
            img_path=img_path,
            img_features=torch.zeros(768),
            dataset_name=ds_name
        )

    @property
    def ground_truths(self):
        if self.mode != 'test':
            warnings.warn("Ground truth is recommended only in test mode")
        return self.ground_truth

    @property
    def dataset_names(self):
        return {entry[0] for entry in self.index_to_dataset_entry}

    def benchmark(self, num_iterations=5000, warmup_iterations=100):
        """
        Benchmark the data loading performance by randomly accessing samples.
        
        Args:
            num_iterations (int): Number of benchmark iterations (default: 5000)
            warmup_iterations (int): Number of warmup iterations to exclude from timing
        
        Returns:
            dict: Benchmark results including average access time and statistics
        """
        import time
        
        # Warmup phase to exclude initialization overhead
        rank_zero_info(f"Running {warmup_iterations} warmup iterations...")
        for i in range(warmup_iterations):
            idx = random.randint(0, len(self) - 1)
            _ = self[idx]
        
        # Benchmark phase
        rank_zero_info(f"Running {num_iterations} benchmark iterations...")
        total_time = 0.0
        times = []
        
        for i in range(num_iterations):
            idx = random.randint(0, len(self) - 1)
            
            start_time = time.perf_counter()
            sample = self[idx]
            end_time = time.perf_counter()
            
            access_time = (end_time - start_time) * 1000  # Convert to milliseconds
            times.append(access_time)
            total_time += access_time
        
        # Calculate statistics
        avg_time = total_time / num_iterations
        min_time = min(times)
        max_time = max(times)
        median_time = sorted(times)[len(times) // 2]
        
        # Calculate percentiles
        sorted_times = sorted(times)
        p95 = sorted_times[int(len(times) * 0.95)]
        p99 = sorted_times[int(len(times) * 0.99)]
        
        results = {
            'total_iterations': num_iterations,
            'average_time_ms': avg_time,
            'min_time_ms': min_time,
            'max_time_ms': max_time,
            'median_time_ms': median_time,
            'p95_time_ms': p95,
            'p99_time_ms': p99,
            'total_time_ms': total_time,
            'throughput_samples_per_second': 1000 / avg_time if avg_time > 0 else float('inf')
        }
        
        # Print results
        rank_zero_info("\n=== CLIPDataset Benchmark Results ===")
        rank_zero_info(f"Total iterations: {results['total_iterations']}")
        rank_zero_info(f"Average access time: {results['average_time_ms']:.4f} ms")
        rank_zero_info(f"Min access time: {results['min_time_ms']:.4f} ms")
        rank_zero_info(f"Max access time: {results['max_time_ms']:.4f} ms")
        rank_zero_info(f"Median access time: {results['median_time_ms']:.4f} ms")
        rank_zero_info(f"95th percentile: {results['p95_time_ms']:.4f} ms")
        rank_zero_info(f"99th percentile: {results['p99_time_ms']:.4f} ms")
        rank_zero_info(f"Total time: {results['total_time_ms']:.4f} ms")
        rank_zero_info(f"Throughput: {results['throughput_samples_per_second']:.2f} samples/second")
        rank_zero_info("=====================================")
        
        return results



class CLIPDataset_ThingEEG(BaseEEGDataset):
    """
    subjects = ['sub-01', 'sub-02', 'sub-05', 'sub-04', 'sub-03', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10']
    """
    def __init__(self, model_type,adap_subject=None, subjects=None, train=True, train_Val_Same_set='none',time_window=[0, 1.0], classes = None, pictures = None,auto_convert=True,filter_channel=True,augmentation_config=None,use_aug=None,float_type='float32',limit_sample=None):
        # Call base class __init__ to initialize common attributes
        super().__init__(
            augmentation_config=augmentation_config,
            use_aug=use_aug,
            sample_rate=512,  # ThingEEG target rate (after resampling from 250Hz)
            limit_sample=limit_sample,
            float_type=float_type,
            aug_verbose=False
        )

        # Access the paths from the config
        data_path = f"{os.environ['WaveMind_ROOT_PATH_']}/data/THING-EEG/Data/Preprocessed_data_250Hz"
        self.img_directory_training = f"{os.environ['WaveMind_ROOT_PATH_']}/data/THING-EEG/images_set/training_images"
        self.img_directory_test = f"{os.environ['WaveMind_ROOT_PATH_']}/data/THING-EEG/images_set/test_images"
        self.feature_path=f"{os.environ['WaveMind_ROOT_PATH_']}/data/THING-EEG/Data/Feature"
        self.filter_channel=filter_channel


        assert model_type=="ViT-L-14-336"

        self.data_path = data_path
        self.train = train
        self.subject_list = [x for x in os.listdir(data_path) if not x.endswith('npz')]
        self.subjects = self.subject_list if subjects is None else subjects
        self.n_sub = len(self.subjects)
        self.time_window = time_window
        self.n_cls = 1654 if train else 200
        self.classes = classes
        self.pictures = pictures
        self.adap_subject = adap_subject  # Save this parameter

        # Assert any subjects in subject_list
        assert any(sub in self.subject_list for sub in self.subjects)



        self.data, self.labels, self.text, self.img = self.load_data()
        self.data = self.extract_eeg(self.data, time_window)
        


        if train_Val_Same_set not in ['none','train','val']:
            rank_zero_info(train_Val_Same_set)
            rank_zero_info(type(train_Val_Same_set))
            raise ValueError(f"Invalid train_Val_Same_set: {train_Val_Same_set}")

        self.train_Val_Same_set = train_Val_Same_set

        if self.train_Val_Same_set != 'none':
            self.train_indices = [i for i in range(len(self.data)) if i % 10 < 9]
            self.val_indices = [i for i in range(len(self.data)) if i % 10 == 9]


        caption_filename = os.path.join(self.feature_path,f"{model_type}_caption_train.json") if self.train else os.path.join(self.feature_path,f"{model_type}_caption_test.json")
        if os.path.exists(caption_filename):
            with open(caption_filename) as f:
                image_cap=json.load(f)
                self.text=[f"{sentence1}, {sentence2}" for sentence1, sentence2 in zip(self.text, image_cap)]


        

        self._text_features, self._img_features = None, None
        self.img_processor, self.img_model, self.text_model, self.text_tokenizer = None, None, None, None

        if self.classes is None and self.pictures is None:
            # Try to load the saved features if they exist
            features_filename = os.path.join(self.feature_path,f"{model_type}_features_train.pt") if self.train else os.path.join(self.feature_path,f"{model_type}_features_test.pt")

            if os.path.exists(features_filename):
                rank_zero_info(f"Loading features from {features_filename}")
                saved_features = torch.load(features_filename,weights_only=False)
                # self._text_features = saved_features['text_features']
                self._text_features = None
                self._img_features = saved_features['img_features']
            else:
                rank_zero_info(f"Features not found at {features_filename}, computing features...")

                if auto_convert:
                    self.img_processor, self.img_model, self.text_model, self.text_tokenizer = self.get_vl_txt_model()

                    # image_cap=self.CaptionEncoder()
                    self._img_features = self.ImageEncoder(self.img)
                    assert len(self.img)==len(self.text)
                    # assert len(image_cap)==len(self.img)==len(self.text)
                    # self.text=[f"{sentence1}, {sentence2}" for sentence1, sentence2 in zip(self.text, image_cap)]
                    # self._text_features = self.Textencoder(self.text)
                    # assert type(self._text_features) == torch.Tensor
                    # assert type(self._text_features) is None
                    assert type(self._img_features) == torch.Tensor
                    torch.save({
                        # 'text_features': self._text_features.cpu(),
                        'text_features': None,
                        'img_features': self._img_features.cpu(),
                    }, features_filename)
                    # with open(caption_filename,'w') as f:
                    #     json.dump(image_cap,f)
        else:
            if auto_convert:
                self.img_processor, self.img_model, self.text_model, self.text_tokenizer = self.get_vl_txt_model()
                # image_cap=self.CaptionEncoder()
                assert len(image_cap)==len(self.img)
                assert len(image_cap)==len(self.text)
                # self.text=[f"{sentence1},{sentence2}" for sentence1, sentence2 in zip(self.text, image_cap)]
                # self._text_features = self.Textencoder(self.text)
                self._text_features=None
                self._img_features = self.ImageEncoder(self.img)



        del self.img_processor, self.img_model, self.text_model, self.text_tokenizer


    def get_vl_txt_model(self):
        img_processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-large-patch14-336')
        img_model = CLIPVisionModelWithProjection.from_pretrained('openai/clip-vit-large-patch14-336')
        text_model = CLIPTextModelWithProjection.from_pretrained('openai/clip-vit-large-patch14-336')
        text_tokenizer = AutoTokenizer.from_pretrained('openai/clip-vit-large-patch14-336')

        return img_processor,img_model,text_model, text_tokenizer

    @property
    def text_features(self):
        self._text_features=self._text_features.squeeze()
        return self._text_features.view(1654, 10, -1).mean(dim=1) if self.train else self._text_features

    @property
    def img_features(self):
        self._img_features=self._img_features.squeeze()
        return self._img_features

    def load_data(self):
        data_list = []
        label_list = []
        texts = []
        images = []

        if self.train:
            directory = self.img_directory_training
        else:
            directory = self.img_directory_test
        # Get all directories in the path
        dirnames = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        dirnames.sort()

        if self.classes is not None:
            dirnames = [dirnames[i] for i in self.classes]

        for dir in dirnames:
            # Try to find the first occurrence of '_'
            try:
                idx = dir.index('_')
                description = dir[idx+1:]  # Get all content after the first '_'
            except ValueError:
                rank_zero_info(f"Skipped: {dir} due to no '_' found.")
                continue

            new_description = f"This picture is {description}"
            texts.extend([new_description] * (10 if self.train else 1))  # Add the same description 10 times

        if self.train:
            img_directory = self.img_directory_training  # Replace with your new address
        else:
            img_directory = self.img_directory_test

        all_folders = [d for d in os.listdir(img_directory) if os.path.isdir(os.path.join(img_directory, d))]
        all_folders.sort()  # Ensure the order of folders

        if self.classes is not None and self.pictures is not None:
            images = []  # Initialize images list
            for i in range(len(self.classes)):
                class_idx = self.classes[i]
                pic_idx = self.pictures[i]
                if class_idx < len(all_folders):
                    folder = all_folders[class_idx]
                    folder_path = os.path.join(img_directory, folder)
                    all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    all_images.sort()
                    if pic_idx < len(all_images):
                        images.append(os.path.join(folder_path, all_images[pic_idx]))
        elif self.classes is not None and self.pictures is None:
            images = []  # Initialize images list
            for i in range(len(self.classes)):
                class_idx = self.classes[i]
                if class_idx < len(all_folders):
                    folder = all_folders[class_idx]
                    folder_path = os.path.join(img_directory, folder)
                    all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    all_images.sort()
                    images.extend(os.path.join(folder_path, img) for img in all_images)
        elif self.classes is None:
            images = []  # Initialize images list
            for folder in all_folders:
                folder_path = os.path.join(img_directory, folder)
                all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                all_images.sort()
                images.extend(os.path.join(folder_path, img) for img in all_images)
        else:
            # Handle other cases, such as mismatched lengths of self.classes and self.pictures
            rank_zero_warn("Error")


        # rank_zero_info("adap_subject", self.adap_subject)
        rank_zero_info("Start loading ThingEEG")
        rank_zero_info("self.subjects", self.subjects)
        train_test = 'train' if self.train else 'test'
        if not (self.subjects==['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09'] and os.path.exists(os.path.join(self.data_path, f'preprocessed_1-9{train_test}.npz'))):
            for subject in self.subjects:
                if self.train:
                    file_name = 'preprocessed_eeg_training.npy'

                    file_path = os.path.join(self.data_path, subject, file_name)
                    data = np.load(file_path, allow_pickle=True)

                    preprocessed_eeg_data = torch.from_numpy(data['preprocessed_eeg_data']).float().detach()
                    times = torch.from_numpy(data['times']).detach()[50:]
                    ch_names = data['ch_names']  # Keep as a Python list or encode appropriately


                    n_classes = 1654  # Each class contains 10 images
                    samples_per_class = 10  # Each class has ten samples

                    if self.classes is not None and self.pictures is not None:
                        for c, p in zip(self.classes, self.pictures):
                            start_index = c * 1 + p
                            if start_index < len(preprocessed_eeg_data):  # Ensure index is within range
                                preprocessed_eeg_data_class = preprocessed_eeg_data[start_index: start_index+1]  # Select only one sample
                                labels = torch.full((1,), c, dtype=torch.long).detach()  # Add class label
                                data_list.append(preprocessed_eeg_data_class)
                                label_list.append(labels)  # Add labels to the label list

                    elif self.classes is not None and self.pictures is None:
                        for c in self.classes:
                            start_index = c * samples_per_class
                            preprocessed_eeg_data_class = preprocessed_eeg_data[start_index: start_index+samples_per_class]
                            labels = torch.full((samples_per_class,), c, dtype=torch.long).detach()  # Add class label
                            data_list.append(preprocessed_eeg_data_class)
                            label_list.append(labels)

                    else:
                        for i in range(n_classes):
                            start_index = i * samples_per_class
                            preprocessed_eeg_data_class = preprocessed_eeg_data[start_index: start_index+samples_per_class]
                            labels = torch.full((samples_per_class,), i, dtype=torch.long).detach()  # Add class label
                            data_list.append(preprocessed_eeg_data_class)
                            label_list.append(labels)

                else:
                    if subject == self.adap_subject or self.adap_subject == None:
                        file_name = 'preprocessed_eeg_test.npy'
                        file_path = os.path.join(self.data_path, subject, file_name)
                        data = np.load(file_path, allow_pickle=True)
                        preprocessed_eeg_data = torch.from_numpy(data['preprocessed_eeg_data']).float().detach()
                        times = torch.from_numpy(data['times']).detach()[50:]
                        ch_names = data['ch_names']  # Keep as a Python list or encode appropriately
                        n_classes = 200  # Each class contains 1 image

                        samples_per_class = 1  # Each class has one sample

                        for i in range(n_classes):
                            if self.classes is not None and i not in self.classes:  # Skip if class not in the specified list
                                continue
                            start_index = i * samples_per_class  # Update start_index for each class
                            preprocessed_eeg_data_class = preprocessed_eeg_data[start_index:start_index+samples_per_class]
                            labels = torch.full((samples_per_class,), i, dtype=torch.long).detach()  # Add class labels
                            preprocessed_eeg_data_class = torch.mean(preprocessed_eeg_data_class.squeeze(0), 0)
                            data_list.append(preprocessed_eeg_data_class)
                            label_list.append(labels)  # Add labels to the label list
                    else:
                        continue
            # Data list: (subjects * classes) * (10 * 4 * 17 * 100)
            # Data tensor: (subjects * classes * 10 * 4) * 17 * 100
            if self.train:
                data_tensor = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape[2:])
                rank_zero_info("data_tensor", data_tensor.shape)
            else:
                data_tensor = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape)
            label_tensor = torch.cat(label_list, dim=0)
            if self.subjects==['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09']:

                np.savez(os.path.join(self.data_path, f'preprocessed_1-9{train_test}.npz'), data_tensor=data_tensor.numpy(), label_tensor=label_tensor.numpy(),times=times, ch_names=ch_names)

        else:
            rank_zero_info("1-9,load cache")
            npz_file=os.path.join(self.data_path, f'preprocessed_1-9{train_test}.npz')
            npz_data=np.load(npz_file)
            data_tensor = torch.from_numpy(npz_data['data_tensor']).float().detach()
            label_tensor = torch.from_numpy(npz_data['label_tensor']).long().detach()
            times = npz_data['times']
            ch_names = npz_data['ch_names']



        if self.train:
            # Label tensor: (subjects * classes * 10 * 4)
            label_tensor = label_tensor.repeat_interleave(4)
            if self.classes is not None:
                unique_values = list(label_tensor.numpy())
                lis = []
                for i in unique_values:
                    if i not in lis:
                        lis.append(i)
                    unique_values = torch.tensor(lis)
                    mapping = {val.item(): index for index, val in enumerate(unique_values)}
                    label_tensor = torch.tensor([mapping[val.item()] for val in label_tensor], dtype=torch.long)

        else:
            pass

        self.times = times
        self.ch_names = ch_names
        rank_zero_info(f"Data tensor shape: {data_tensor.shape}, label tensor shape: {label_tensor.shape}, text length: {len(texts)}, image length: {len(images)}")
        
        
        
        data_tensor=data_tensor.type(self.float_type)

        return data_tensor, label_tensor, texts, images

    def extract_eeg(self, eeg_data, time_window):

        start, end = time_window

        # Get the indices of the times within the specified window
        indices = (self.times >= start) & (self.times <= end)
        # Use these indices to select the corresponding data
        extracted_data = eeg_data[..., indices]
        # print0(f"extracted_data shape: {extracted_data.shape}")

        # resample the data to 512Hz

        extracted_data = F.interpolate(extracted_data, size=512, mode='linear', align_corners=False)

        # Reorder the EEG channels
        if self.filter_channel:
            extracted_data = self.reorder_eeg_channels(extracted_data)

        return extracted_data

    def Textencoder(self, text):
        rank_zero_info("start convert text")

        self.text_model.to('cuda')
        text_inputs = text
        rank_zero_info("text_inputs", len(text_inputs))

        # Use the CLIP model to encode text
        text_features = []
        with torch.no_grad():
            batch_size = 96
            for i in tqdm(range(0, len(text_inputs), batch_size)):
                batch_text_inputs = text_inputs[i:i + batch_size]
                batch_text_inputs=self.text_tokenizer(batch_text_inputs,padding=True,return_tensors="pt", truncation=True).to('cuda')
                batch_text_features = self.text_model(**batch_text_inputs).text_embeds
                if len(batch_text_features.shape) == 2:
                    batch_text_features = batch_text_features.unsqueeze(1)
                text_features.append(batch_text_features.cpu())
        text_features = torch.cat(text_features, dim=0)
        rank_zero_info("text_features", text_features.shape)
        self.text_model.to('cpu')

        return text_features

    def ImageEncoder(self, images):
        rank_zero_info("start convert image")
        batch_size = 20  # Set to an appropriate value
        image_features_list = []

        self.img_model.to('cuda')
        for i in tqdm(range(0, len(images), batch_size)):
            batch_images = images[i:i + batch_size]
            batch_images = [Image.open(img).convert("RGB") for img in batch_images]
            batch_image_features = self.img_processor(images=batch_images, return_tensors="pt").to('cuda')
            with torch.no_grad():
                batch_image_features = self.img_model(**batch_image_features, return_dict=True).image_embeds
                if len(batch_image_features.shape) == 2:
                    batch_image_features = batch_image_features.unsqueeze(1)
                # L2 normalize to unit hypersphere for CLIP alignment
                batch_image_features = F.normalize(batch_image_features, p=2, dim=-1)
            image_features_list.append(batch_image_features.cpu())

        image_features = torch.cat(image_features_list, dim=0)
        rank_zero_info("image_features", image_features.shape)
        self.img_model.to('cpu')
        return image_features

    def CaptionEncoder(self):
        rank_zero_info("Start captioning")
        image_pattern = re.compile(r'\b(image|picture|photo|img|photograph|snapshot)s?\b', re.IGNORECASE)
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        model_path='/mnt/nvme0n1/model_dir/Qwen2.5-VL-7B-Instruct'
        # model_path='Qwen/Qwen2.5-VL-7B-Instruct'
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path,device_map='cuda:0', torch_dtype=torch.bfloat16)
        processor = AutoProcessor.from_pretrained(model_path, use_fast=True)

        all_generated_texts = []
        img_cap={}
        for img_path in tqdm(self.img):
            if img_path in img_cap:
                caption = img_cap[img_path]
            else:
                obj_class = img_path.split('/')[-1].split('_')[0]
                with Image.open(img_path) as img:
                    inputs = processor.apply_chat_template(
                        [{"role": "user", "content": [
                            {"type": "image", "image": img},
                            {"type": "text",
                             "text": f"Describe this images concisely. We already know {obj_class} in this image. Only focus on the necessary content with in the image, no external knowledge is needed. Try your best to describe the color/shape/action of the object. Keep within 1 sentences."}
                        ]}],
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt"
                    ).to('cuda:0')

                    with torch.no_grad():
                        output = model.generate(**inputs, max_new_tokens=200, pad_token_id=processor.tokenizer.eos_token_id,
                                                temperature=0.9)

                    caption = processor.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True,
                                                     clean_up_tokenization_spaces=True)[0].strip()

                    caption = image_pattern.sub('EEG', caption)
                    img_cap[img_path]=caption

            all_generated_texts.append(caption)

        return all_generated_texts

    def reorder_eeg_channels(self,original_data):
        """
        Reorders EEG channels in the input data according to the specified target channels.

        Parameters:
        original_data (numpy.ndarray): Input EEG data with shape (samples, 63, 512)

        Returns:
        numpy.ndarray: Output EEG data with reordered channels, shape (samples, num_target_channels, 512)

        Raises:
        ValueError: If any channel in target_channels is not found in original_channels
        """
        # Original channel names in exact order
        original_channels = [
            'Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'Fz',
            'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2',
            'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4',
            'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6',
            'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7',
            'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2'
        ]

        # Target channels and order specification
        target_channels = [
            'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
            'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8',
            'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8',
            'POz', 'O1', 'Oz', 'O2', 'AFz', 'CPz', 'FCz'
        ]

        # Validate channels and get indices
        channel_indices = []
        missing_channels = []
        for ch in target_channels:
            if ch in original_channels:
                idx = original_channels.index(ch)
                channel_indices.append(idx)
            else:
                missing_channels.append(ch)

        if missing_channels:
            raise ValueError(f"No found: {missing_channels}")

        # Reorder the channels
        reordered_data = original_data[:, channel_indices, :]

        return reordered_data



    def __getitem__(self, index):


        if self.train_Val_Same_set != 'none':
            if self.train_Val_Same_set == 'train':
                index = self.train_indices[index]
            else:  # 'val'
                index = self.val_indices[index]
                
        if self._img_features is None:
            warnings.warn("Features not extracted. Please set auto_convert=True to extract features automatically.")
        

        if self.pictures is None:
            if self.classes is None:
                index_n_sub_train = self.n_cls * 10 * 4
                index_n_sub_test = self.n_cls * 1 * 80
            else:
                index_n_sub_train = len(self.classes)* 10 * 4
                index_n_sub_test = len(self.classes)* 1 * 80
            # text_index: classes*10
            if self.train:
                text_index = (index % index_n_sub_train) // (4)
                img_index = (index % index_n_sub_train) // (4)
            else:
                text_index = (index % index_n_sub_test)
                img_index = (index % index_n_sub_test)
        else:
            if self.classes is None:
                index_n_sub_train = self.n_cls * 1 * 4
                index_n_sub_test = self.n_cls * 1 * 80
            else:
                index_n_sub_test = len(self.classes)* 1 * 80
                index_n_sub_train = len(self.classes)* 1 * 4
            # text_index: classes
            if self.train:
                text_index = (index % index_n_sub_train) // (1 * 4)
            else:
                text_index = (index % index_n_sub_test)
            # img_index: classes * 10
            if self.train:
                img_index = (index % index_n_sub_train) // (4)
            else:
                img_index = (index % index_n_sub_test)

        if self.train==False:
            text_index=text_index%(self.n_cls)
            img_index%=(self.n_cls)
            text_index=int(text_index)
            img_index=int(img_index)



        x = self.data[index]
        
        
        label = self.labels[index]
        text = self.text[text_index]
        img = self.img[img_index]

        text_features, img_features = None, None
        # if self._text_features is not None and self._img_features is not None:
        #     text_features = self._text_features[text_index]
        #     img_features = self._img_features[img_index]
        if self._img_features is not None:
            img_features = self._img_features[img_index]
            img_features.squeeze()

        if self._text_features is not None:
            text_features = self._text_features[text_index]
            text_features.squeeze()




        if int(x.shape[-1]) != self.sample_rate:
            import scipy
            x = scipy.signal.resample(x, self.sample_rate, axis=-1)


        # Convert to tensor if not already
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).float()

        # Validate EEG data before augmentation
        if not self._validate_eeg_data(x):
            warnings.warn(f"NaN or Inf found in EEG data at index {index} in dataset THING, before noise processing")
            return self.__getitem__(index+1)

        # Apply augmentation using base class method
        is_train = self.train and (self.train_Val_Same_set != 'val')
        x = self._apply_augmentation(x, is_train)



        return self._standardize_return_dict(
            eeg_data=x,
            label=label,
            text=text,
            text_features=torch.zeros(768),
            img_path=img,
            img_features=torch.Tensor(img_features) if img_features is not None else torch.zeros(768),
            dataset_name="thingEEG"
        )

    def __len__(self):
        length=0
        if self.train_Val_Same_set == 'none':
            length=self.data.shape[0]
        elif self.train_Val_Same_set == 'train':
            length=len(self.train_indices)
        elif self.train_Val_Same_set == 'val':
            length=len(self.val_indices)
        else:
            raise ValueError("Invalid value for train_Val_Same_set,Expected 'none', 'train', or 'val'.")
        return length if self.limit_sample is None else min(length, self.limit_sample)






# class CLIPDataset_ALL(Dataset):
#     def __init__(self,train,hdf5_file_path,ground_truth_dir,optimize=True,**kwargs):
#         super().__init__()
#         subjects,exclude_dataset = None,['SEED-IV', 'FACE','BCICIV2a','HMC','P2018']
#         if "subjects" in kwargs:
#             subjects = kwargs['subjects']
#         if 'exclude_dataset' in kwargs:
#             exclude_dataset=kwargs['exclude_dataset']
#         if 'use_aug' in kwargs:
#             use_aug=kwargs['use_aug']
#         if 'limit_sample' in kwargs:
#             limit_sample=kwargs['limit_sample']
#         self.thing_dataset=CLIPDataset_ThingEEG(train=train,model_type='ViT-L-14-336',subjects=subjects,use_aug=use_aug,limit_sample=limit_sample)
#         self.mode='train' if train else 'test'
#         self.second_dataset=CLIPDataset(hdf5_file_path=hdf5_file_path, mode=self.mode,dataset_name='all',float_type='float32',ground_truth_dir=ground_truth_dir,exclude_dataset=exclude_dataset,use_aug=use_aug,limit_sample=limit_sample)
#         self.total_length=len(self.thing_dataset)+len(self.second_dataset)
#         # rank_zero_info(f"Statistic: Total Thing EEG samples: {len(self.thing_dataset)}, Total genral dataset samples: {len(self.second_dataset)}, Total samples: {self.total_length}")

#     def change_sample_rate(self,sample_rate):
#         self.thing_dataset.change_sample_rate(sample_rate)
#         self.second_dataset.change_sample_rate(sample_rate)
    
    
#     @property
#     def ground_truths(self):
#         assert self.mode=='test', "ground_truths only support test mode"
#         ground_truths={}
#         ground_truths.update({'thingEEG':[self.thing_dataset.img_features,200]})
#         ground_truths.update(self.second_dataset.ground_truth)
#         return ground_truths

#     def __len__(self):
#         return self.total_length
#     def __getitem__(self, idx):
#         eeg=None
#         if idx<len(self.thing_dataset):
#             eeg=self.thing_dataset[idx]
            
#         else:
#             eeg=self.second_dataset[idx-len(self.thing_dataset)]
            
#         eeg['eeg_data'] = eeg['eeg_data'].to(dtype=torch.float32)
        
#         return eeg
            
            





class CLIPDataModule(LightningDataModule):
    def __init__(self, task,load_dir,batch_size=64, num_workers=64,thing_train_val_same_set=False,dataset_name='all',float_type='float32',force_recompute_weight=False,hdf5_file_path=None,use_shm=False,use_dynamic_sampling=False,sampling_strategy='class_weight',clean=True,auto_clean_shm_when_exit=True,**kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.hdf5_file_path=os.path.join(load_dir,'data_label.h5') if hdf5_file_path==None else hdf5_file_path
        self.ground_truth_dir=os.path.join(load_dir,'CLIP_groundTruth')
        self.weights_file_path=os.path.join(load_dir,'dataset_weights.pth')
        self.dataset_name=dataset_name
        self.float_type=float_type
        self.task=task
        self.force_recompute_weight=force_recompute_weight
        
        # self.use_weighted_sampler=use_weighted_sampler
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.thing_train_val_same_set=thing_train_val_same_set
        self.feature_all_test={}
        self.sample_rate=512
        self.use_shm = use_shm
        self.use_dynamic_sampling = use_dynamic_sampling
        self.sampling_strategy = sampling_strategy
        self.shm_path=None
        self.clean = clean  # New parameter: whether to clean up resources
        self.auto_clean_shm_when_exit = auto_clean_shm_when_exit  # New parameter: whether to clean up shared memory file when exiting
        self.kwargs=kwargs
        self.save_hyperparameters()

    def setup(self, stage=None):
        # specific setting for thing-EEG
        if self.task=='THING':
            if self.thing_train_val_same_set==False:
                self.train_dataset = CLIPDataset_ThingEEG(train=True,model_type='ViT-L-14-336',**self.kwargs)
                self.val_dataset = CLIPDataset_ThingEEG(train=False,model_type='ViT-L-14-336',**self.kwargs)
                self.feature_all_test.update({'thingEEG':[self.val_dataset.img_features,200]})
            elif self.thing_train_val_same_set==True:
                self.train_dataset = CLIPDataset_ThingEEG(train=True,train_Val_Same_set='train',model_type='ViT-L-14-336',**self.kwargs)
                self.val_dataset = CLIPDataset_ThingEEG(train=True,train_Val_Same_set='val',model_type='ViT-L-14-336',**self.kwargs)
                val_features=self.val_dataset.img_features[::10]
                self.feature_all_test.update({'thingEEG':[val_features,1654]})
                assert len(val_features)==1654
            else:
                raise "ERROR"

        elif self.task=='BCI':
            
            
            rank_zero_info(f"Setup BCI dataset from: {self.hdf5_file_path}")
            
            tmp_hdf5_file_path=self.setup_shared_memory(self.hdf5_file_path)
        
            
            # Select dataset class based on dynamic sampling setting
            dataset_class = dynamicCLIPDataset if self.use_dynamic_sampling else CLIPDataset
            
            # Create train dataset with dynamic sampling support
            train_dataset_kwargs = {
                'hdf5_file_path': tmp_hdf5_file_path,
                'mode': 'train',
                'dataset_name': self.dataset_name,
                'ground_truth_dir': self.ground_truth_dir,
                **self.kwargs
            }
            
            # Add dynamic sampling parameters if enabled
            if self.use_dynamic_sampling:
                train_dataset_kwargs.update({
                    'sampling_strategy': self.sampling_strategy,
                    'weight_file': os.path.join(self.ground_truth_dir, 'dynamic_weights.pth')
                })
            
            self.train_dataset = dataset_class(**train_dataset_kwargs)
            
            # Create validation and test datasets (always use CLIPDataset)
            self.val_dataset = CLIPDataset(
                hdf5_file_path=tmp_hdf5_file_path,
                mode='test',
                dataset_name=self.dataset_name,
                ground_truth_dir=self.ground_truth_dir,
                **self.kwargs
            )
            
            self.test_dataset = CLIPDataset(
                hdf5_file_path=tmp_hdf5_file_path,
                mode='cross',
                dataset_name=self.dataset_name,
                ground_truth_dir=self.ground_truth_dir,
                **self.kwargs
            )
            
            self.test_dataset = self.test_dataset if self.test_dataset.__len__() > 0 else None
            # Fix: Convert ground_truths to the expected format [features, k_value]
            ground_truths = self.val_dataset.ground_truths
            for ds_name, features in ground_truths.items():
                # Determine k_value based on features length (number of samples)
                k_value = features.shape[0]  # Use the number of features as k_value
                self.feature_all_test[ds_name] = [features, k_value]

        # elif self.task=='ALL':
        #     self.train_dataset = CLIPDataset_ALL(train=True,hdf5_file_path=self.hdf5_file_path,ground_truth_dir=self.ground_truth_dir,**self.kwargs)
        #     self.val_dataset = CLIPDataset_ALL(train=False,hdf5_file_path=self.hdf5_file_path,ground_truth_dir=self.ground_truth_dir,**self.kwargs)
        #     self.test_dataset =None
        #     self.feature_all_test.update(self.val_dataset.ground_truths)
        else:
            raise ValueError(f"task should be 'THING' or 'BCI' or 'ALL', but got {self.task}")

        if self.sample_rate!=512:
            self.train_dataset.change_sample_rate(self.sample_rate)
            self.val_dataset.change_sample_rate(self.sample_rate)
            if self.test_dataset is not None:
                self.test_dataset.change_sample_rate(self.sample_rate)
        
        
        rank_zero_info("*****************************")
        rank_zero_info('DM Setup done')
        rank_zero_info("Data Statistic")
        rank_zero_info(f"Train dataset length: {len(self.train_dataset)}")
        rank_zero_info(f"Test dataset length: {len(self.val_dataset)}")
        rank_zero_info(f"Cross dataset length: {len(self.test_dataset) if self.test_dataset is not None else 0}")
        rank_zero_info("*****************************")

    @property
    def dataset_names(self):
        return list(self.feature_all_test.keys())
    
    
    def change_sample_rate_hook(self,sample_rate):
        self.sample_rate=sample_rate
        self.train_dataset.change_sample_rate(self.sample_rate)
        self.val_dataset.change_sample_rate(self.sample_rate)
        if self.test_dataset is not None:
            self.test_dataset.change_sample_rate(self.sample_rate)
        
        
    def setup_shared_memory(self,hdf5_path: str, shm_path: Optional[str] = None) -> str:
        """
        Safely copies HDF5 file to shared memory (/dev/shm) with proper cache warming.
        Handles both distributed and single-process environments with robust error handling.
        
        Args:
            hdf5_path: Original HDF5 file path on disk
            shm_path: Target path in shared memory (defaults to /dev/shm/<filename>)
            use_shm: Whether to attempt shared memory setup (set to False to disable)
        
        Returns:
            Path to use for dataset (may be original path if setup fails)
        """
        # Skip shared memory setup if explicitly disabled
        if not self.use_shm:
            return hdf5_path
        
        if not os.path.exists(hdf5_path):
            raise FileNotFoundError(f"Original HDF5 file not found: {hdf5_path}")
        
        # Auto-generate shm_path if not provided
        if shm_path is None:
            shm_path = f"/dev/shm/{os.path.basename(hdf5_path)}"
        self.shm_path=shm_path
        
        # Check shared memory available space (only check /dev/shm)
        try:
            file_size = os.path.getsize(hdf5_path)
            # Check available space in /dev/shm
            shm_usage = shutil.disk_usage('/dev/shm')
            shm_available = shm_usage.free
            # Reserve 20% safety buffer
            safe_available_mem = shm_available * 0.8
            
            rank_zero_info(f"[Memory Check] File size: {file_size/1024**3:.2f} GB, "
                f"SHM available: {shm_available/1024**3:.2f} GB, "
                f"Safe limit: {safe_available_mem/1024**3:.2f} GB")
            
            if file_size > safe_available_mem:
                rank_zero_warn(f"[Memory Warning] File size exceeds safe shared memory limit. "
                    f"Skipping shared memory setup.")
                return hdf5_path
        except Exception as e:
            rank_zero_warn(f"[Memory Check Error] {str(e)}. Proceeding without size check.")
        
        # Detect distributed environment status
        is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        rank = 0
        world_size = 1
        
        if is_distributed:
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank_zero_info("[Info] Running in single-process mode (non-distributed)")
        
        # Only rank 0 performs the copy operation
        if rank == 0:
            # Use file lock to prevent concurrent training jobs from corrupting SHM file
            import fcntl
            lock_file_path = shm_path + '.lock'

            try:
                # Create and acquire exclusive lock (blocks if another process holds it)
                with open(lock_file_path, 'w') as lock_fd:
                    rank_zero_info(f"[Rank0] Acquiring file lock: {lock_file_path}")
                    fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
                    rank_zero_info(f"[Rank0] Lock acquired successfully")

                    # Double-check file existence inside lock (atomic with lock)
                    if not os.path.exists(shm_path):
                        rank_zero_info(f"[Rank0] Shared memory file not exists: {shm_path}")
                        rank_zero_info(f"[Rank0] Copying {hdf5_path} to shared memory ({shm_path})...")
                        try:
                            # Efficient copy using Python with progress display
                            rank_zero_info(f"[Rank0] Starting copy with progress display...")
                            self._copy_file_with_progress(hdf5_path, shm_path)
                            rank_zero_info(f"[Rank0] Copy completed successfully")

                            # Verify file integrity after copy
                            if not os.path.exists(shm_path):
                                raise RuntimeError("Copy completed but file not found")

                            file_size_original = os.path.getsize(hdf5_path)
                            file_size_shm = os.path.getsize(shm_path)
                            if file_size_original != file_size_shm:
                                raise RuntimeError(f"File size mismatch: original={file_size_original}, shm={file_size_shm}")

                            rank_zero_info(f"[Rank0] File integrity verified (size: {file_size_shm/1024**3:.2f} GB)")

                        except Exception as e:
                            rank_zero_warn(f"[Rank0] Copy failed: {str(e)}. Falling back to original path.")
                            # Clean up incomplete file
                            if os.path.exists(shm_path):
                                os.remove(shm_path)
                                rank_zero_info(f"[Rank0] Removed incomplete shared memory file")
                            return hdf5_path
                    else:
                        rank_zero_info(f"[Rank0] Shared memory file already exists: {shm_path}")

                    # Lock automatically released when exiting 'with' block
                    rank_zero_info(f"[Rank0] Releasing file lock")

            except Exception as e:
                rank_zero_warn(f"[Rank0] File locking error: {str(e)}. Falling back to original path.")
                return hdf5_path
            finally:
                # Clean up lock file (best effort)
                try:
                    if os.path.exists(lock_file_path):
                        os.remove(lock_file_path)
                except:
                    pass  # Ignore cleanup errors
            
        
        # Synchronize all processes (essential for multi-GPU)
        if is_distributed:
            safe_barrier(timeout_minutes=30, context="shared_memory_setup")
            rank_zero_info(f"[Rank{rank}] Synchronized with other {world_size-1} processes after memory setup")
        
        # Verify file accessibility across all processes
        try:
            # Check file existence and readability
            file_ok = True
            if rank == 0:
                if not os.path.exists(shm_path):
                    rank_zero_warn(f"[Rank0] Shared memory file missing: {shm_path}")
                    file_ok = False
                elif not os.access(shm_path, os.R_OK):
                    rank_zero_warn(f"[Rank0] No read access to: {shm_path}")
                    file_ok = False
            
            # In distributed mode, ensure all ranks agree on file status
            if is_distributed:
                file_ok_tensor = torch.tensor([int(file_ok)], dtype=torch.int, device='cuda')
                torch.distributed.all_reduce(file_ok_tensor, op=torch.distributed.ReduceOp.MIN)
                file_ok = bool(file_ok_tensor.item())
            
            if not file_ok:
                rank_zero_warn(f"[Rank{rank}] Shared memory file verification failed. Using original path.")
                return hdf5_path
        except Exception as e:
            rank_zero_warn(f"[Verification Error] {str(e)}. Using original path.")
            # delete possibly corrupted shm file
            if rank == 0 and os.path.exists(shm_path):
                os.remove(shm_path)
                rank_zero_warn(f"[Rank0] Removed corrupted shared memory file: {shm_path}")
            return hdf5_path
        
        rank_zero_info(f"[Rank{rank}] Using shared memory path: {shm_path}")
        return shm_path
        
    
    def _copy_file_with_progress(self, src_path, dst_path):
        """
        Copy file with progress display using manual read/write with tqdm
        
        Args:
            src_path: Source file path
            dst_path: Destination file path
        """
        total_size = os.path.getsize(src_path)
        chunk_size = 1024 * 1024  # 1MB chunks
        
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Copying") as pbar:
            with open(src_path, 'rb') as fsrc, open(dst_path, 'wb') as fdst:
                while True:
                    chunk = fsrc.read(chunk_size)
                    if not chunk:
                        break
                    fdst.write(chunk)
                    pbar.update(len(chunk))

    def _read_file_with_progress(self, file_path, file_size, chunk_size=1024*1024):
        """
        Read file with progress display for cache warming
        
        Args:
            file_path: File path to read
            file_size: Total file size
            chunk_size: Chunk size for reading (default: 1MB)
        """
        with open(file_path, 'rb') as f:
            with tqdm(total=file_size, unit='B', unit_scale=True, desc="Preheating cache") as pbar:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    pbar.update(len(chunk))
    

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    def test_dataloader(self):

        if hasattr(self,'test_dataset'):
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )
        else:
            warnings.warn("No test dataset found, skip test_dataloader")
            return None
        

    def close(self):
        """Close all open file handles and clean up resources"""

        # IMPORTANT: All ranks must close their own file handles
        # (Previously only rank 0 closed, causing resource leaks on other ranks)
        if hasattr(self, '_file') and self._file is not None:
            try:
                self._file.close()
                self._file = None
                # Log from all ranks to verify proper cleanup
                current_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                rank_zero_info(f"[Rank{current_rank}] Closed HDF5 file handle")
            except Exception as e:
                import warnings
                warnings.warn(f"Error closing HDF5 file: {str(e)}")

        # Synchronize all ranks before SHM cleanup to ensure all file handles closed
        if torch.distributed.is_initialized():
            try:
                safe_barrier(timeout_minutes=1, context="cleanup_before_shm_removal")
                if os.environ.get('LOCAL_RANK', '0') == '0':
                    rank_zero_info("[Rank0] All ranks synchronized after closing file handles")
            except Exception as e:
                import warnings
                warnings.warn(f"Barrier timeout during cleanup: {str(e)}")

        # Only rank 0 removes the shared memory file (after all ranks finished using it)
        if os.environ.get('LOCAL_RANK', '0') == '0' and self.clean:
            if hasattr(self, 'shm_path'):
                if self.shm_path is not None and os.path.exists(self.shm_path) and self.auto_clean_shm_when_exit:
                    try:
                        os.remove(self.shm_path)
                        rank_zero_info(f"[Rank0] Removed shared memory file: {self.shm_path}")
                    except Exception as e:
                        import warnings
                        warnings.warn(f"Failed to remove SHM file: {str(e)}")
            

    def __del__(self):
        """Destructor - ensure resources are cleaned up"""
        self.close()
                
                
                
def select_data_module_for_train(dataset:str,limit_sample:int,load_dir:str,augmentation_config=None,use_aug=None,hdf5_path=None,num_workers=64,use_shm=False,use_dynamic_sampling=False,thing_specific_sub=None,auto_clean_shm_when_exit=True,batch_size=512):
    import os

    # Handle OmegaConf ListConfig conversion (fix for Hydra config parsing)
    from omegaconf import ListConfig
    if isinstance(dataset, (list, ListConfig)):
        dataset = list(dataset)

    # Handle backward compatibility for use_aug parameter
    if augmentation_config is None and use_aug is not None:
        from EEG_Encoder.Config.training_config import AugmentationConfig, TrainAugmentationConfig, TestAugmentationConfig
        augmentation_config = AugmentationConfig(
            train=TrainAugmentationConfig(enable=use_aug),
            test=TestAugmentationConfig(enable=True)
        )

    if 'thing_specific' in dataset:
        # Use provided subject list or default to all subjects
        subjects = thing_specific_sub if thing_specific_sub else ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08','sub-09']
        dm = CLIPDataModule(task='THING', thing_train_val_same_set=False,batch_size=batch_size, num_workers=num_workers,subjects=subjects,limit_sample=limit_sample,load_dir=load_dir,augmentation_config=augmentation_config, hdf5_file_path=hdf5_path, auto_clean_shm_when_exit=auto_clean_shm_when_exit)
    else:
        dm = CLIPDataModule(task='BCI', dataset_name=dataset,batch_size=batch_size,num_workers=num_workers,limit_sample=limit_sample,load_dir=load_dir,augmentation_config=augmentation_config,hdf5_file_path=hdf5_path,use_shm=use_shm,use_dynamic_sampling=use_dynamic_sampling,auto_clean_shm_when_exit=auto_clean_shm_when_exit)
    dm.setup()
    return dm





class dynamicCLIPDataset(CLIPDataset):
    """
    A dynamic dataset class that inherits from CLIPDataset and addresses data imbalance issues.

    This class implements dynamic sampling strategies to handle imbalanced data distributions
    while maintaining the same interface and functionality as CLIPDataset.
    """

    def __init__(self, hdf5_file_path, mode, ground_truth_dir=None, dataset_name='all',
                exclude_dataset=['SEED-IV', 'FACE', 'BCICIV2a', 'HMC', 'P2018'], augmentation_config=None, use_aug=None, limit_sample=None,
                preload_metadata=True, aug_verbose=False, sampling_strategy='class_weight',
                class_weights=None, weight_file=None, force_recompute_weights=False):
        """
        Initialize the dynamicCLIPDataset.
        
        Args:
            All arguments from CLIPDataset plus:
            sampling_strategy (str): Strategy for handling imbalanced data.
                Options: 'class_weight', 'undersample', 'oversample'
            class_weights (dict): Predefined class weights. If None, weights will be computed dynamically.
            weight_file (str): Path to save/load computed weights.
            force_recompute_weights (bool): Whether to recompute weights even if weight_file exists.
        """
        # Initialize parent class
        super().__init__(hdf5_file_path, mode, ground_truth_dir, dataset_name,
                        exclude_dataset, augmentation_config, use_aug, limit_sample, preload_metadata, aug_verbose)
        
        # Dynamic sampling parameters
        self.sampling_strategy = sampling_strategy
        self.class_weights = class_weights
        self.weight_file = weight_file
        self.force_recompute_weights = force_recompute_weights
        
        # Compute or load class weights
        self._compute_class_weights()
        
        # Apply sampling strategy
        self._apply_sampling_strategy()

    def _compute_class_weights(self):
        """
        Compute class weights based on the distribution of samples across datasets.
        """
        if self.class_weights is not None and not self.force_recompute_weights:
            rank_zero_info("Using predefined class weights")
            return
        
        # Check if weights are already saved
        if self.weight_file and os.path.exists(self.weight_file) and not self.force_recompute_weights:
            try:
                self.class_weights = torch.load(self.weight_file)
                rank_zero_info(f"Loaded class weights from {self.weight_file}")
                
                # Calculate dataset counts for weight statistics
                dataset_counts = Counter()
                for ds_name, _ in self.index_to_dataset_entry:
                    dataset_counts[ds_name] += 1

                # Print weight statistics regardless of weight source
                rank_zero_info("Weight Statistics:")
                for ds_name, weight in self.class_weights.items():
                    count = dataset_counts.get(ds_name, 0)
                    rank_zero_info(f"[Weight Statistic] Dataset: {ds_name}, Count: {count}, Weight: {weight:.4f}")

                rank_zero_info(f"Loaded weights for {len(self.class_weights)} datasets")
                return
            except Exception as e:
                rank_zero_warn(f"Failed to load weights from {self.weight_file}: {e}")

        rank_zero_info("Computing class weights dynamically...")
        
        # Count samples per dataset
        dataset_counts = Counter()
        for ds_name, _ in self.index_to_dataset_entry:
            dataset_counts[ds_name] += 1
        
        # Compute weights (inverse frequency)
        total_samples = sum(dataset_counts.values())
        self.class_weights = {}
        
        for ds_name, count in dataset_counts.items():
            # Weight is inversely proportional to frequency
            self.class_weights[ds_name] = total_samples / (len(dataset_counts) * count)
        
        # Save weights if file path is provided
        if self.weight_file:
            os.makedirs(os.path.dirname(self.weight_file), exist_ok=True)
            torch.save(self.class_weights, self.weight_file)
            rank_zero_info(f"Saved class weights to {self.weight_file}")


        # Print weight of each class
        rank_zero_info("Weight Statistics:")
        for ds_name, weight in self.class_weights.items():
            count = dataset_counts.get(ds_name, 0)
            rank_zero_info(f"[Weight Statistic] Dataset: {ds_name}, Count: {count}, Weight: {weight:.4f}")


        rank_zero_info(f"Computed weights for {len(self.class_weights)} datasets")

    def _apply_sampling_strategy(self):
        """
        Apply the specified sampling strategy to handle data imbalance.
        """
        if self.sampling_strategy == 'class_weight':
            # For class weighting, we modify the sample weights
            self.sample_weights = self._compute_sample_weights()
        elif self.sampling_strategy in ['undersample', 'oversample']:
            # For resampling strategies, we modify the index mapping
            self._resample_dataset()
        else:
            rank_zero_warn(f"Unknown sampling strategy: {self.sampling_strategy}. Using default sampling.")
            self.sample_weights = None

    def _compute_sample_weights(self):
        """
        Compute sample weights based on class weights for weighted random sampling.
        """
        weights = []
        for ds_name, _ in self.index_to_dataset_entry:
            # Use class weight for the dataset
            weight = self.class_weights.get(ds_name, 1.0)
            weights.append(weight)
        
        # Normalize weights
        weights = torch.tensor(weights, dtype=torch.float32)
        weights = weights / weights.sum()
        
        return weights

    def _resample_dataset(self):
        """
        Resample the dataset according to the specified strategy.
        """
        # Group samples by dataset
        dataset_samples = {}
        for i, (ds_name, _) in enumerate(self.index_to_dataset_entry):
            if ds_name not in dataset_samples:
                dataset_samples[ds_name] = []
            dataset_samples[ds_name].append(i)
        
        # Determine target count based on strategy
        if self.sampling_strategy == 'undersample':
            target_count = min(len(indices) for indices in dataset_samples.values())
        else:  # oversample
            target_count = max(len(indices) for indices in dataset_samples.values())
        
        # Resample
        resampled_indices = []
        for ds_name, indices in dataset_samples.items():
            if self.sampling_strategy == 'undersample':
                # Randomly select target_count samples
                selected = random.sample(indices, min(target_count, len(indices)))
            else:  # oversample
                # Repeat samples to reach target_count
                selected = []
                while len(selected) < target_count:
                    remaining = target_count - len(selected)
                    if remaining <= len(indices):
                        selected.extend(random.sample(indices, remaining))
                    else:
                        selected.extend(indices)
                        remaining -= len(indices)
            
            resampled_indices.extend(selected)
        
        # Update index mapping
        self.resampled_index_to_dataset_entry = [self.index_to_dataset_entry[i] for i in resampled_indices]
        self.length = len(self.resampled_index_to_dataset_entry)
        rank_zero_info(f"Resampled dataset: {len(self.index_to_dataset_entry)} -> {self.length} samples")

    def __getitem__(self, idx):
        """
        Get item with dynamic sampling support.
        """
        # Use resampled indices if resampling is applied
        if hasattr(self, 'resampled_index_to_dataset_entry'):
            ds_name, entry_idx = self.resampled_index_to_dataset_entry[idx]
            # Find the original index for parent class access
            original_idx = None
            for i, (orig_ds_name, orig_entry_idx) in enumerate(self.index_to_dataset_entry):
                if orig_ds_name == ds_name and orig_entry_idx == entry_idx:
                    original_idx = i
                    break
            if original_idx is None:
                # Fallback to current index if mapping not found
                original_idx = idx
            sample = super().__getitem__(original_idx)
        else:
            ds_name, entry_idx = self.index_to_dataset_entry[idx]
            sample = super().__getitem__(idx)
        
        # Add sample weight if using class weighting strategy
        if self.sampling_strategy == 'class_weight' and self.sample_weights is not None:
            sample['weight'] = self.sample_weights[idx]
        
        return sample

    def get_sample_weight(self, idx):
        """
        Get the weight for a specific sample.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            float: Weight of the sample
        """
        if self.sample_weights is not None:
            return self.sample_weights[idx]
        return 1.0





    

