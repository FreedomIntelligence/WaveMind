"""
WaveMind Unified EEG Preprocessing Pipeline
Magicconch with cc
Created: 2026-03-23
Modified: 2026-03-23

Unified preprocessing script for all 5 WaveMind datasets (SEED, TUAB, TUEV, ImageNetEEG, THING-EEG).
Produces:
  - data/Total/data_label.h5 (unified HDF5 with all datasets)
  - data/Total/CLIP_groundTruth/*.npy + *.pkl (RAG ground truth)

Usage:
  python preprocess_wavemind.py --all --seed 42
  python preprocess_wavemind.py --dataset SEED --seed 42
  python preprocess_wavemind.py --rag-only
"""

import os
import sys
import re
import glob
import hashlib
import argparse
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import h5py
import mne
import scipy.signal
from tqdm import tqdm
from PIL import Image
import gc

# ====== PATH SETUP ======
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = get_wavemind_root()
HDF5_PATH = os.path.join(ROOT_DIR, 'data/Total/data_label.h5')
CLIP_CACHE_DIR = os.path.join(ROOT_DIR, 'data/Total/.clip_cache')
RAG_DIR = os.path.join(ROOT_DIR, 'data/Total/CLIP_groundTruth')

for d in [CLIP_CACHE_DIR, RAG_DIR]:
    os.makedirs(d, exist_ok=True)

# ====== IMPORTS FROM EXISTING UTILS ======
sys.path.insert(0, SCRIPT_DIR)
from data.Utils import (
    SEEDDatasetInfo, TUABDatasetInfo, TUEVDatasetInfo,
    ImageNetEEGDatasetInfo, FilterTransform, load_in_memory, get_each_type_caption_feature,
)
from data.preUtils import eeg_filter_all

# ====== CONSTANTS ======
TARGET_CHANNELS = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8',
    'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8',
    'POz', 'O1', 'Oz', 'O2', 'AFz', 'CPz', 'FCz'
]
TARGET_CHANNELS_SET = set(TARGET_CHANNELS)

# ====== CLIP MODEL CACHE (Singleton) ======
class CLIPModelCache:
    """
    Singleton cache for CLIP models to avoid repeated loading (~2GB per load).
    Reuses text and vision models across all processors.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        from transformers import CLIPTextModelWithProjection, CLIPVisionModelWithProjection, AutoTokenizer, CLIPImageProcessor

        self.text_model = CLIPTextModelWithProjection.from_pretrained('openai/clip-vit-large-patch14-336')
        self.text_model.to('cuda')
        self.text_model.eval()
        self.text_tokenizer = AutoTokenizer.from_pretrained('openai/clip-vit-large-patch14-336')

        self.img_processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-large-patch14-336')
        self.img_model = CLIPVisionModelWithProjection.from_pretrained('openai/clip-vit-large-patch14-336')
        self.img_model.to('cuda')
        self.img_model.eval()

        self._initialized = True

    @torch.no_grad()
    def get_text_features(self, texts: List[str]) -> np.ndarray:
        """Get CLIP text embeddings for a list of texts."""
        inputs = self.text_tokenizer(texts, padding=True, return_tensors="pt").to('cuda')
        outputs = self.text_model(**inputs)
        features = F.normalize(outputs.text_embeds, p=2, dim=-1)
        return features.cpu().numpy()

    @torch.no_grad()
    def get_image_features(self, image_path: str) -> np.ndarray:
        """Get CLIP image embedding with persistent disk cache."""
        cache_key = self._hash_path(image_path)
        cache_file = os.path.join(CLIP_CACHE_DIR, f"{cache_key}.npy")

        if os.path.exists(cache_file):
            return np.load(cache_file)

        real_path = image_path
        if not os.path.isabs(image_path):
            real_path = os.path.join(ROOT_DIR, image_path)

        image = Image.open(real_path).convert('RGB')
        inputs = self.img_processor(images=image, return_tensors="pt").to('cuda')
        outputs = self.img_model(**inputs)
        features = F.normalize(outputs.image_embeds, p=2, dim=-1)
        features_np = features.cpu().numpy()

        np.save(cache_file, features_np)
        return features_np

    @staticmethod
    def _hash_path(path: str) -> str:
        return hashlib.md5(path.encode()).hexdigest()


# ====== UNIFORM ELECTRODE MAPPING ======
def apply_10_20_mapping(raw: mne.io.Raw) -> mne.io.Raw:
    """
    Apply standard 10-20 electrode naming convention to EDF data.
    Used by TUAB and TUEV.
    """
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
            if electrode in electrode_mapping or electrode in {
                    'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8'}:
                selected_chs.append(ch_name)
                new_name = electrode_mapping.get(electrode, electrode)
                rename_dict[ch_name] = new_name

    raw_filtered = raw.copy().pick_channels(selected_chs)
    raw_filtered.rename_channels(rename_dict)
    raw_filtered.set_montage('standard_1020', on_missing='ignore')
    return raw_filtered


# ====== FIXED FilterTransform warning logic ======
def _create_filter_transform(ds_info) -> FilterTransform:
    """
    Create FilterTransform with corrected warning logic.
    """
    return FilterTransform(ds_info, downsample_fs=None, h_freq=None, l_freq=None, notch=None)


# ====== HDF5 WRITER ======
class HDF5Writer:
    """
    Unified HDF5 writer with file locking and compression.
    """
    def __init__(self, hdf5_path: str, feature_cache: Optional[Dict] = None):
        self.hdf5_path = hdf5_path
        self.feature_cache = feature_cache or {}
        self.lock_file = hdf5_path + '.lock'

    def _acquire_lock(self, timeout: int = 600):
        import fcntl
        start = os.path.getmtime(self.hdf5_path) if os.path.exists(self.hdf5_path) else 0
        for _ in range(timeout):
            try:
                lock_fd = open(self.lock_file, 'w')
                fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return lock_fd
            except (IOError, BlockingIOError):
                import time; time.sleep(0.1)
        raise TimeoutError(f"Could not acquire lock for {self.hdf5_path}")

    def _release_lock(self, lock_fd):
        import fcntl
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
        lock_fd.close()
        if os.path.exists(self.lock_file):
            os.remove(self.lock_file)

    def save(self, ds_name: str, eeg_data: np.ndarray, text_features: np.ndarray,
             captions: List[str], labels: np.ndarray, image_paths: Optional[List[str]] = None):
        """Save a batch to HDF5 with structured array format."""
        caption_length = 192
        path_length = 192
        num_samples = eeg_data.shape[0]

        assert not np.isnan(eeg_data).any(), f"{ds_name}: NaN in EEG data"
        assert not np.isinf(eeg_data).any(), f"{ds_name}: Inf in EEG data"
        if eeg_data.shape[2] == 513:
            eeg_data = eeg_data[:, :, :512]

        caption_array = np.array([str(c)[:caption_length].ljust(caption_length) for c in captions],
                                  dtype=f'S{caption_length}')
        if image_paths is not None:
            image_path_array = np.array([str(p)[:path_length].ljust(path_length) for p in image_paths],
                                       dtype=f'S{path_length}')
        else:
            image_path_array = np.array([''.ljust(path_length)] * num_samples, dtype=f'S{path_length}')

        dtype_fields = [
            ('eeg_data', np.float32, eeg_data.shape[1:]),
            ('text_feature', np.float16, text_features.squeeze().shape[1:]),
            ('caption', f'S{caption_length}'),
            ('label', labels.dtype, labels.shape[1:]),
            ('image_path', f'S{path_length}')
        ]
        structured_data = np.zeros(num_samples, dtype=np.dtype(dtype_fields))
        structured_data['eeg_data'] = eeg_data.astype(np.float32)
        structured_data['text_feature'] = text_features.squeeze().astype(np.float16)
        structured_data['caption'] = caption_array
        structured_data['label'] = labels
        structured_data['image_path'] = image_path_array

        try:
            lock_fd = self._acquire_lock()
            with h5py.File(self.hdf5_path, 'a', locking=True, swmr=True) as f:
                if ds_name in f:
                    f[ds_name].resize(f[ds_name].shape[0] + num_samples, axis=0)
                    f[ds_name][-num_samples:] = structured_data
                    print(f"  [{ds_name}] Appended {num_samples} samples. Total: {f[ds_name].shape[0]}")
                else:
                    f.create_dataset(ds_name, data=structured_data, maxshape=(None,), chunks=(64,), compression='gzip')
                    print(f"  [{ds_name}] Created with {num_samples} samples")
            self._release_lock(lock_fd)
        except Exception as e:
            self._release_lock(lock_fd)
            raise


# ====== TEXT FEATURE GENERATOR ======
class TextFeatureGenerator:
    """
    Generate CLIP text embeddings with in-memory caching.
    """
    def __init__(self, clip_cache: CLIPModelCache):
        self.clip_cache = clip_cache
        self.text_cache: Dict[str, np.ndarray] = {}

    def generate(self, labels: np.ndarray, ds_info) -> Tuple[np.ndarray, List[str]]:
        """Generate text features for all labels, with caching."""
        unique_labels = np.unique(labels)
        texts = []
        for label in tqdm(unique_labels, desc="Generating text captions"):
            caption = ds_info.get_caption(int(label))
            if caption in self.text_cache:
                texts.append(self.text_cache[caption])
            else:
                feat = self.clip_cache.get_text_features([caption])[0]
                self.text_cache[caption] = feat
                texts.append(feat)

        # Map all labels to features
        label_to_feat = {int(l): self.text_cache[ds_info.get_caption(int(l))] for l in unique_labels}
        features = np.stack([label_to_feat[int(l)] for l in labels])
        captions = [ds_info.get_caption(int(l)) for l in labels]
        return features, captions


# ====== BASE PROCESSOR ======
class BaseProcessor(ABC):
    """Base class for all EEG dataset processors."""

    def __init__(self, args, clip_cache: CLIPModelCache):
        self.args = args
        self.clip_cache = clip_cache
        self.writer = HDF5Writer(args.hdf5)
        self.text_gen = TextFeatureGenerator(clip_cache)

    @abstractmethod
    def process(self):
        """Process the dataset and save to HDF5."""
        pass

    def validate_eeg(self, eeg: np.ndarray, name: str = ""):
        """Validate EEG data quality."""
        assert eeg.shape[-1] == 512, f"{name}: Expected 512 samples, got {eeg.shape[-1]}"
        assert not np.isnan(eeg).any(), f"{name}: Contains NaN"
        assert not np.isinf(eeg).any(), f"{name}: Contains Inf"


# ====== SEED PROCESSOR ======
class SEEDProcessor(BaseProcessor):
    """Process SEED emotion dataset using torcheeg."""

    def process(self):
        from torcheeg.datasets import SEEDDataset
        from torcheeg import transforms
        from torcheeg.model_selection import train_test_split_cross_subject, train_test_split

        ds_name = 'SEED'
        ds_info = SEEDDatasetInfo()
        iopath = f'~/.cache/torcheeg/{ds_name}'

        # Load dataset with torcheeg
        dataset = SEEDDataset(
            io_path=iopath,
            root_path=os.path.join(SCRIPT_DIR, 'SEED/Preprocessed_EEG'),
            offline_transform=transforms.Compose([
                FilterTransform(ds_info, downsample_fs=512, h_freq=None),
            ]),
            online_transform=transforms.Compose([]),
            label_transform=transforms.Compose([
                transforms.Select('emotion'),
                transforms.Lambda(lambda x: x + 1),  # {-1,0,1} → {0,1,2}
            ]),
            num_worker=self.args.num_workers,
        )

        # Split
        train_val, test = train_test_split_cross_subject(dataset=dataset)
        train, val = train_test_split(dataset=train_val)
        splits = {'train': train, 'test': val, 'cross': test}
        print(f"[SEED] train={len(train)}, val={len(val)}, cross={len(test)}")

        for split_name, split_data in splits.items():
            print(f"[SEED] Processing {split_name} ({len(split_data)} samples)...")
            eeg_all, labels = load_in_memory(split_data, cut=False)
            self.validate_eeg(eeg_all, f"SEED_{split_name}")

            # Generate text features
            text_features, captions = self.text_gen.generate(labels, ds_info)

            # Save
            self.writer.save(
                ds_name=f'{ds_name}_{split_name}',
                eeg_data=eeg_all,
                text_features=text_features,
                captions=captions,
                labels=labels,
            )

            del eeg_all, labels, split_data
            gc.collect()

        del dataset, train_val, train, val, test
        gc.collect()


# ====== TUAB PROCESSOR ======
class TUABProcessor(BaseProcessor):
    """Process TUAB abnormal EEG detection dataset."""

    def process(self):
        np.random.seed(self.args.seed)
        ds_name = 'TUAB'
        ds_info = TUABDatasetInfo()
        filter_trans = _create_filter_transform(ds_info)

        root = os.path.join(ROOT_DIR, f'data/{ds_name}/process_refine')
        splits = {
            'train': os.path.join(root, 'train'),
            'test': os.path.join(root, 'val'),
            'cross': os.path.join(root, 'test'),
        }

        for split_name, folder in splits.items():
            if not os.path.exists(folder):
                print(f"[TUAB] Skipping {split_name}: folder not found ({folder})")
                continue

            files = glob.glob(os.path.join(folder, '*.npz'))
            # Randomly sample 20% for efficiency
            files = np.random.choice(files, int(len(files) * 0.2), replace=False).tolist()
            print(f"[TUAB] Processing {split_name}: {len(files)} files...")

            eegs, labels = [], []
            for fpath in tqdm(files, desc=f"TUAB {split_name}"):
                data = np.load(fpath)
                eeg = filter_trans(eeg=data['signal'])['eeg']
                self.validate_eeg(eeg, f"TUAB_{split_name}")
                eegs.append(eeg)
                labels.append(int(data['label']))

            eegs = np.stack(eegs)
            labels = np.array(labels)

            text_features, captions = self.text_gen.generate(labels, ds_info)

            self.writer.save(
                ds_name=f'{ds_name}_{split_name}',
                eeg_data=eegs,
                text_features=text_features,
                captions=captions,
                labels=labels,
            )

            del eegs, labels
            gc.collect()


# ====== TUEV PROCESSOR ======
class TUEVProcessor(BaseProcessor):
    """Process TUEV event detection dataset with streaming EDF processing."""

    def process(self):
        np.random.seed(self.args.seed)
        ds_name = 'TUEV'
        ds_info = TUEVDatasetInfo()
        filter_trans = _create_filter_transform(ds_info)

        edf_root = os.path.join(SCRIPT_DIR, ds_name, 'edf')
        train_dir = os.path.join(edf_root, 'train')
        eval_dir = os.path.join(edf_root, 'eval')

        for dataset_type, folder in [('train', train_dir), ('eval', eval_dir)]:
            if not os.path.exists(folder):
                print(f"[TUEV] Skipping {dataset_type}: folder not found")
                continue

            edf_files = []
            for dirName, _, fnames in os.walk(folder):
                edf_files.extend(os.path.join(dirName, f) for f in fnames if f.endswith('.edf'))

            print(f"[TUEV] Processing {dataset_type}: {len(edf_files)} EDF files...")

            buffers = {
                'train': {'eeg': [], 'label': []},
                'test': {'eeg': [], 'label': []},
                'cross': {'eeg': [], 'label': []},
            }

            for fpath in tqdm(edf_files, desc=f"TUEV {dataset_type}"):
                try:
                    raw = mne.io.read_raw_edf(fpath, preload=True)
                    raw = apply_10_20_mapping(raw)
                    raw = raw.resample(200)
                    signals = raw.get_data(units='uV')
                    times = raw.times
                    rec_file = fpath[:-3] + 'rec'
                    event_data = np.genfromtxt(rec_file, delimiter=',')
                    raw.close()

                    signals, _, labels = self._build_events(signals, times, event_data)

                    # Resample 200Hz → 512Hz
                    signals = scipy.signal.resample(signals, 5 * 512, axis=-1)

                    for idx, (sig, label) in enumerate(zip(signals, labels)):
                        for i in range(5):
                            sub_sig = sig[:, i * 512:(i + 1) * 512]
                            assert sub_sig.shape == (19, 512)
                            eeg = filter_trans(eeg=sub_sig)['eeg']
                            self.validate_eeg(eeg, f"TUEV_{dataset_type}")

                            if dataset_type == 'eval':
                                buffers['cross']['eeg'].append(eeg)
                                buffers['cross']['label'].append(int(label))
                            else:
                                if np.random.random() < 0.9:
                                    buffers['train']['eeg'].append(eeg)
                                    buffers['train']['label'].append(int(label))
                                else:
                                    buffers['test']['eeg'].append(eeg)
                                    buffers['test']['label'].append(int(label))
                except Exception as e:
                    print(f"  [TUEV] Error processing {fpath}: {e}")
                    continue

            # Flush buffers
            for split_name, buf in buffers.items():
                if len(buf['eeg']) == 0:
                    continue
                eegs = np.stack(buf['eeg'])
                labels = np.array(buf['label'])
                text_features, captions = self.text_gen.generate(labels, ds_info)
                self.writer.save(
                    ds_name=f'{ds_name}_{split_name}',
                    eeg_data=eegs,
                    text_features=text_features,
                    captions=captions,
                    labels=labels,
                )
                del eegs, labels, buf
                gc.collect()

    @staticmethod
    def _build_events(signals, times, EventData):
        """Build EEG events from annotation data."""
        [numEvents, _] = EventData.shape
        fs = 200.0
        [numChan, _] = signals.shape
        features = np.zeros([numEvents, numChan, int(fs) * 5])
        labels = np.zeros([numEvents, 1])
        offset = signals.shape[1]
        signals = np.concatenate([signals, signals, signals], axis=1)

        for i in range(numEvents):
            chan = int(EventData[i, 0])
            start = np.where(times >= EventData[i, 1])[0][0]
            end = np.where(times >= EventData[i, 2])[0][0]
            features[i] = signals[:, offset + start - 2 * int(fs):offset + end + 2 * int(fs)]
            labels[i] = int(EventData[i, 3]) - 1  # {1-6} → {0-5}
        return features, np.zeros([numEvents]), labels.flatten()


# ====== IMAGE NET EEG PROCESSOR ======
class ImageNetEEGProcessor(BaseProcessor):
    """Process ImageNetEEG dataset with CLIP image features."""

    def process(self):
        ds_name = 'ImageNetEEG'
        ds_info = ImageNetEEGDatasetInfo()

        eeg_signals_path = os.path.join(ROOT_DIR, f'data/{ds_name}/eeg_signals_raw_with_mean_std.pth')
        if not os.path.exists(eeg_signals_path):
            print(f"[ImageNetEEG] Data not found: {eeg_signals_path}")
            return

        loaded = torch.load(eeg_signals_path, weights_only=False)
        class_mapping = self._load_class_mapping(
            os.path.join(ROOT_DIR, f'data/{ds_name}/image_class.txt'))

        # Build channel index list
        all_channels = loaded.get('channel_names', None)
        if all_channels is None:
            all_channels = [
                "Fp1", "Fpz", "Fp2", "AF3", "AF4", "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
                "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "Cz",
                "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8",
                "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POz", "PO4",
                "PO6", "PO8", "CB1", "O1", "Oz", "O2", "CB2"
            ]

        # Build target channel index mapping with proper interpolation
        channel_to_idx = {ch.lower(): idx for idx, ch in enumerate(all_channels)}
        index_list = []
        for target in TARGET_CHANNELS:
            idx = channel_to_idx.get(target.lower())
            if idx is None:
                # Find nearest channel instead of random
                idx = self._find_nearest_channel(target, channel_to_idx, all_channels)
            index_list.append(idx)

        img_dir = os.path.join(ROOT_DIR, f'data/{ds_name}/Image')

        for split, subjects in [('cross', [6])]:
            print(f"[ImageNetEEG] Processing {split} (subjects={subjects})...")
            dataset = [(loaded['dataset'][i], i)
                       for i in range(len(loaded['dataset']))
                       if loaded['dataset'][i]['subject'] in subjects]

            if not dataset:
                print(f"[ImageNetEEG] No data for {split}")
                continue

            eegs, labels, text_feats, captions, img_paths = [], [], [], [], []

            for item, _ in tqdm(dataset, desc=f"ImageNetEEG {split}"):
                eeg = item['eeg'].float()[index_list]
                eeg = self._stack_and_truncate(eeg, 1000)
                eeg = F.interpolate(eeg.unsqueeze(0).unsqueeze(0), size=512,
                                     mode='linear', align_corners=False).squeeze().numpy()
                if np.isnan(eeg).any() or np.isinf(eeg).any():
                    continue

                label = int(item['label'])
                img_name = loaded['images_name'][item['image']]
                img_rel_path = f'data/{ds_name}/Image/{img_name}.JPEG'
                img_full_path = os.path.join(ROOT_DIR, img_rel_path)

                if not os.path.exists(img_full_path):
                    continue

                # Get CLIP image feature
                img_feat = self.clip_cache.get_image_features(img_rel_path)
                label_name = class_mapping[loaded['labels'][label]]

                eegs.append(eeg)
                labels.append(label)
                text_feats.append(img_feat)
                captions.append(f'This is a {label_name}')
                img_paths.append(img_rel_path)

            if not eegs:
                print(f"[ImageNetEEG] No valid samples for {split}")
                continue

            eegs = np.stack(eegs)
            labels = np.array(labels)
            text_feats = np.stack(text_feats).squeeze(1)

            self.writer.save(
                ds_name=f'{ds_name}_{split}',
                eeg_data=eegs,
                text_features=text_feats,
                captions=captions,
                labels=labels,
                image_paths=img_paths,
            )

            del eegs, labels, text_feats
            gc.collect()

    @staticmethod
    def _load_class_mapping(filepath):
        mapping = {}
        with open(filepath) as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) < 2:
                    continue
                key, values = parts
                cleaned = [v.strip() for v in values.split(',')]
                min_word_val = min(cleaned, key=lambda v: len(v.split()))
                mapping[key] = str.lower(min_word_val)
        return mapping

    @staticmethod
    def _stack_and_truncate(eeg, target_len):
        n_steps = eeg.shape[1]
        repeats = (target_len + n_steps - 1) // n_steps
        stacked = eeg.repeat(1, repeats)
        return stacked[:, :target_len]

    @staticmethod
    def _find_nearest_channel(target, channel_map, all_channels):
        """Find nearest channel using MNE standard montage distances."""
        target_pos = None
        montage = mne.channels.make_standard_montage('standard_1020')
        try:
            target_pos = montage.get_positions()['ch_pos'][target]
        except KeyError:
            pass

        if target_pos is None:
            # Fallback: use Fpz as default
            try:
                target_pos = montage.get_positions()['ch_pos']['Fpz']
            except KeyError:
                return 0

        min_dist, nearest = float('inf'), 0
        for ch in TARGET_CHANNELS:
            if ch == target:
                continue
            try:
                pos = montage.get_positions()['ch_pos'][ch]
                dist = np.linalg.norm(pos - target_pos)
                if dist < min_dist:
                    min_dist = dist
                    idx = channel_map.get(ch.lower())
                    if idx is not None:
                        nearest = idx
            except KeyError:
                continue
        return nearest


# ====== THING-EEG PROCESSOR ======
class ThingEEGProcessor(BaseProcessor):
    """Process THING-EEG dataset using CLIPDataset_ThingEEG."""

    def process(self):
        from EEG_Encoder.Tools.dataBuilder import CLIPDataset_ThingEEG

        ds_info_name = 'THING-EEG'  # Just for naming

        # Close-set: subjects 1-9
        train_subjects = ['sub-01', 'sub-02', 'sub-05', 'sub-04', 'sub-03',
                          'sub-06', 'sub-07', 'sub-08', 'sub-09']
        for split_name, train_flag, subjects in [
            ('train', True, train_subjects),
            ('test', False, train_subjects),
            ('cross', False, ['sub-10']),
        ]:
            print(f"[THING-EEG] Processing {split_name}...")
            dataset = CLIPDataset_ThingEEG(
                train=train_flag,
                model_type='ViT-L-14-336',
                subjects=subjects,
                use_aug=False,
            )

            eegs, img_features, labels, captions, img_paths = [], [], [], [], []
            for i in tqdm(range(len(dataset)), desc=f"THING-EEG {split_name}"):
                sample = dataset[i]
                eegs.append(sample['eeg_data'])
                img_features.append(sample['img_features'])
                labels.append(sample['label'])
                captions.append(sample['text'])
                img_paths.append(sample['img_path'])

            eegs = np.stack(eegs).squeeze()
            img_features = np.stack(img_features).squeeze()
            labels = np.array(labels)
            self.validate_eeg(eegs, f"THING-EEG_{split_name}")

            self.writer.save(
                ds_name=f'thingEEG_{split_name}',
                eeg_data=eegs,
                text_features=img_features,
                captions=captions,
                labels=labels,
                image_paths=img_paths,
            )

            del dataset, eegs, img_features
            gc.collect()


# ====== RAG FEATURE GENERATOR ======
class RagFeatureGenerator:
    """
    Generate CLIP ground truth NPY and PKL files for RAG retrieval.
    Replaces the original create_dataset_pkl.py functionality.
    """
    def __init__(self, clip_cache: CLIPModelCache):
        self.clip_cache = clip_cache
        self.rag_dir = RAG_DIR
        os.makedirs(self.rag_dir, exist_ok=True)

    def generate_all(self):
        """Generate ground truth for all datasets."""
        generators = [
            ('SEED', self._generate_seed),
            ('TUAB', self._generate_tuab),
            ('TUEV', self._generate_tuev),
            ('THING-closeset', self._generate_thing_closeset),
            ('THING-zeroshot', self._generate_thing_zeroshot),
        ]

        for name, func in generators:
            print(f"[RAG] Generating {name}...")
            try:
                func()
            except Exception as e:
                print(f"[RAG] Error generating {name}: {e}")

    def _generate_seed(self):
        ds_info = SEEDDatasetInfo()
        features = get_each_type_caption_feature(ds_info)
        npy_path = os.path.join(self.rag_dir, 'SEED.npy')
        pkl_path = os.path.join(self.rag_dir, 'SEED.pkl')

        np.save(npy_path, features.astype(np.float16))
        with open(pkl_path, 'wb') as f:
            import pickle
            sorted_labels = [ds_info.id2label()[k] for k in sorted(ds_info.id2label().keys())]
            pickle.dump(sorted_labels, f)
        print(f"  [RAG] SEED: {features.shape} features saved")

    def _generate_tuab(self):
        ds_info = TUABDatasetInfo()
        features = get_each_type_caption_feature(ds_info)
        npy_path = os.path.join(self.rag_dir, 'TUAB.npy')
        pkl_path = os.path.join(self.rag_dir, 'TUAB.pkl')

        np.save(npy_path, features.astype(np.float16))
        with open(pkl_path, 'wb') as f:
            import pickle
            sorted_labels = [ds_info.id2label()[k] for k in sorted(ds_info.id2label().keys())]
            pickle.dump(sorted_labels, f)
        print(f"  [RAG] TUAB: {features.shape} features saved")

    def _generate_tuev(self):
        ds_info = TUEVDatasetInfo()
        features = get_each_type_caption_feature(ds_info)
        npy_path = os.path.join(self.rag_dir, 'TUEV.npy')
        pkl_path = os.path.join(self.rag_dir, 'TUEV.pkl')

        np.save(npy_path, features.astype(np.float16))
        with open(pkl_path, 'wb') as f:
            import pickle
            sorted_labels = [ds_info.id2label()[k] for k in sorted(ds_info.id2label().keys())]
            pickle.dump(sorted_labels, f)
        print(f"  [RAG] TUEV: {features.shape} features saved")

    def _generate_thing_closeset(self):
        self._generate_thing(closeset=True)

    def _generate_thing_zeroshot(self):
        self._generate_thing(closeset=False)

    def _generate_thing(self, closeset: bool):
        from EEG_Encoder.Tools.dataBuilder import CLIPDataset_ThingEEG
        import pickle

        dataset = CLIPDataset_ThingEEG(
            train=closeset,
            train_Val_Same_set='none',
            model_type='ViT-L-14-336',
            subjects=['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05',
                      'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10'],
        )

        db = {}
        for i in tqdm(range(len(dataset)), desc=f"THING-EEG {'closeset' if closeset else 'zeroshot'}"):
            sample = dataset[i]
            cls_name = sample['img_path'].split('/')[-1].split('_')[0]
            feat = sample['img_features']
            if cls_name not in db:
                db[cls_name] = []
            db[cls_name].append(feat)

        for cls_name in db:
            arr = np.array(db[cls_name])
            db[cls_name] = F.normalize(torch.from_numpy(arr.mean(axis=0).astype(np.float32)), p=2, dim=-1).numpy()

        feature_data = []
        class_labels = []
        for cls_name in tqdm(db, desc="THING averaging"):
            feature_data.append(db[cls_name])
            class_labels.append(cls_name)

        feature_data = np.array(feature_data).astype(np.float16)

        prefix = 'thingEEG_closeset' if closeset else 'thingEEG'
        npy_path = os.path.join(self.rag_dir, f'{prefix}.npy')
        pkl_path = os.path.join(self.rag_dir, f'{prefix}.pkl')

        np.save(npy_path, feature_data)
        with open(pkl_path, 'wb') as f:
            pickle.dump(class_labels, f)

        print(f"  [RAG] {prefix}: {feature_data.shape} features, {len(class_labels)} classes")


# ====== UTILS: Fix FilterTransform warning logic ======
def _fix_filter_transform_warning():
    """Patch FilterTransform.__call__ to fix the warning logic bug."""
    _original_call = FilterTransform.__call__

    def _fixed_call(self, eeg: np.ndarray) -> Dict[str, np.ndarray]:
        import warnings as _warnings
        if eeg.shape[-1] % self.fs != 0:
            raise ValueError(f"Data length is not multiple of {self.fs}Hz, but {eeg.shape[-1]} samples.")
        raw = mne.io.RawArray(eeg.astype(np.float32), mne.create_info(
            ch_names=self.electrode_list,
            sfreq=self.fs,
            ch_types='eeg',
        ))
        # Fixed: add parentheses for correct operator precedence
        if not (raw.n_times / self.fs == 1 or raw.n_times / self.fs == 2) and not self.warn_length_already:
            _warnings.warn(f"Data length is not 1s or 2s, but {raw.n_times / self.fs}s.")
            self.warn_length_already = True
        if hasattr(self, 'pre_hook'):
            raw = self.pre_hook(raw)

        if self.l_freq is not None or self.h_freq is not None or self.notch:
            raw = eeg_filter_all(raw, self.l_freq, self.h_freq, self.notch)
        raw = self.filterInterpolateCh(raw)

        if self.downsample_fs:
            raw = raw.resample(self.downsample_fs, n_jobs=-1)
        filtered_eeg = raw.get_data()
        return {'eeg': filtered_eeg}

    FilterTransform.__call__ = _fixed_call


# ====== MAIN ======
def main():
    parser = argparse.ArgumentParser(
        description='WaveMind Unified EEG Preprocessing Pipeline')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['all', 'SEED', 'TUAB', 'TUEV', 'ImageNetEEG', 'THING-EEG'],
                        help='Dataset to process')
    parser.add_argument('--hdf5', type=str, default=HDF5_PATH,
                        help='Output HDF5 path')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--rag-only', action='store_true',
                        help='Only generate RAG ground truth files')
    parser.add_argument('--num-workers', type=int, default=16,
                        help='Number of workers for torcheeg')
    parser.add_argument('--clip-model', type=str, default='openai/clip-vit-large-patch14-336',
                        help='CLIP model name')

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 60)
    print("WaveMind Unified EEG Preprocessing Pipeline")
    print(f"  Root: {ROOT_DIR}")
    print(f"  HDF5: {args.hdf5}")
    print(f"  Seed: {args.seed}")
    print("=" * 60)

    # Fix FilterTransform bug
    _fix_filter_transform_warning()

    # Initialize CLIP cache (singleton, loaded once)
    print("[INIT] Loading CLIP model (will be cached)...")
    clip_cache = CLIPModelCache()
    print("[INIT] CLIP model loaded and cached.")

    # RAG-only mode
    if args.rag_only:
        print("[MODE] RAG-only: generating ground truth files...")
        rag_gen = RagFeatureGenerator(clip_cache)
        rag_gen.generate_all()
        print("[DONE] RAG ground truth generation complete.")
        return

    # Process datasets
    processors = []
    if args.dataset in ('all', 'SEED'):
        processors.append(SEEDProcessor(args, clip_cache))
    if args.dataset in ('all', 'TUAB'):
        processors.append(TUABProcessor(args, clip_cache))
    if args.dataset in ('all', 'TUEV'):
        processors.append(TUEVProcessor(args, clip_cache))
    if args.dataset in ('all', 'ImageNetEEG'):
        processors.append(ImageNetEEGProcessor(args, clip_cache))
    if args.dataset in ('all', 'THING-EEG'):
        processors.append(ThingEEGProcessor(args, clip_cache))

    for proc in processors:
        print(f"\n[PROCESS] Starting {proc.__class__.__name__}...")
        try:
            proc.process()
        except Exception as e:
            print(f"[ERROR] {proc.__class__.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
        print(f"[PROCESS] {proc.__class__.__name__} complete.")

    # Generate RAG ground truth
    print("\n[GENERATE] Creating RAG ground truth files...")
    rag_gen = RagFeatureGenerator(clip_cache)
    rag_gen.generate_all()

    print("\n" + "=" * 60)
    print("[DONE] All preprocessing complete!")
    print(f"  HDF5: {args.hdf5}")
    print(f"  RAG: {RAG_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
