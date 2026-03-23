TARGET_CHANNELS = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8',
    'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8',
    'POz', 'O1', 'Oz', 'O2', 'AFz', 'CPz', 'FCz'
]
import os

import torch.nn.functional as F
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from data.Utils import Convert_and_Save, get_wavemind_root
ds_name='ImageNetEEG'
hdf5_path=f'{get_wavemind_root()}/data/Total/data_label.h5'

img_processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-large-patch14-336')
img_model = CLIPVisionModelWithProjection.from_pretrained('openai/clip-vit-large-patch14-336')



class EEGDataset:

    # Constructor
    def __init__(self, eeg_signals_path=os.path.join(get_wavemind_root(),'data/ImageNetEEG/eeg_signals_raw_with_mean_std.pth'), subjects=None):
        # "subjects" can be [0,1,2,3,4,5,6,7,8,9] representing the subject number
        # Load EEG signals
        loaded = torch.load(eeg_signals_path,weights_only=False)
        if subjects is not None:
            self.data = [loaded['dataset'][i] for i in range(len(loaded['dataset'])) if loaded['dataset'][i]['subject'] in subjects]
        else:
            self.data = loaded['dataset']
        self.labels_name = loaded["labels"]
        self.images_name = loaded["images"]

        for item in loaded['dataset']:
            eeg = item['eeg']
            if torch.isnan(eeg).any() or torch.isinf(eeg).any():
                print(f"Invalid data found in sample {item['subject']} - {item['label']}")
                print("NaN count:", torch.isnan(eeg).sum())
                print("Inf count:", torch.isinf(eeg).sum())

        self.project_root = get_wavemind_root()
        # Compute size
        self.size = len(self.data)
        self.image_fir=os.path.join(self.project_root,'data/ImageNetEEG/Image')
        self.class_dict=self.load_class_mapping(os.path.join(self.project_root,'data/ImageNetEEG/image_class.txt'))

        channel_names = [
            "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2",
            "FC6", "T7", "C3", "Cz", "C4", "T8", "TP9", "CP5", "CP1", "CP2",
            "CP6", "TP10", "P7", "P3", "Pz", "P4", "P8", "PO9", "O1", "Oz",
            "O2", "PO10", "AF7", "AF3", "AF4", "AF8", "F5", "F1", "F2", "F6",
            "FT9", "FT7", "FC3", "FC4", "FT8", "FT10", "C5", "C1", "C2", "C6",
            "TP7", "CP3", "CPz", "CP4", "TP8", "P5", "P1", "P2", "P6", "PO7",
            "PO3", "POz", "PO4", "PO8", "Fpz", "F9", "AFF5h", "AFF1h", "AFF2h",
            "AFF6h", "F10", "FTT9h", "FTT7h", "FCC5h", "FCC3h", "FCC1h", "FCC2h",
            "FCC4h", "FCC6h", "FTT8h", "FTT10h", "TPP9h", "TPP7h", "CPP5h",
            "CPP3h", "CPP1h", "CPP2h", "CPP4h", "CPP6h", "TPP8h", "TPP10h",
            "POO9h", "POO1", "POO2", "POO10h", "Iz", "AFp1", "AFp2", "FFT9h",
            "FFT7h", "FFC5h", "FFC3h", "FFC1h", "FFC2h", "FFC4h", "FFC6h",
            "FFT8h", "FFT10h", "TTP7h", "CCP5h", "CCP3h", "CCP1h", "CCP2h",
            "CCP4h", "CCP6h", "TTP8h", "P9", "PPO9h", "PPO5h", "PPO1h", "PPO2h",
            "PPO6h", "PPO10h", "P10", "I1", "OI1h", "OI2h", "I2"
        ]

        # Build channel name to index mapping (case-insensitive)
        channel_to_idx = {ch.lower(): idx for idx, ch in enumerate(channel_names)}
        
        self.index_list = []
        for target in TARGET_CHANNELS:
            idx = channel_to_idx.get(target.lower())
            if idx is None:
                idx = self._find_nearest_channel(target, channel_to_idx)
            self.index_list.append(idx)


        
        
        
        
        
        

    # Get size

    @staticmethod
    def _find_nearest_channel(target, channel_to_idx):
        """Find nearest channel using MNE standard montage distances."""
        import mne
        import numpy as np

        target_pos = None
        montage = mne.channels.make_standard_montage('standard_1020')
        try:
            target_pos = montage.get_positions()['ch_pos'][target]
        except KeyError:
            pass

        if target_pos is None:
            # Fallback: use Fpz
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
                    idx = channel_to_idx.get(ch.lower())
                    if idx is not None:
                        nearest = idx
            except KeyError:
                continue
        return nearest

    def __len__(self):
        return self.size

    def load_class_mapping(self,file_path):
        class_mapping = {}

        with open(file_path, 'r') as f:
            lines = f.readlines()

        lines = [line.strip() for line in lines if line.strip()]

        for line in lines:
            parts = line.split(' ', 1)
            if len(parts) < 2:
                continue

            key = parts[0]
            values = parts[1].split(',')

            cleaned_values = [value.strip() for value in values]
            word_counts = [len(value.split()) for value in cleaned_values]


            min_word_count = min(word_counts)
            min_word_value = cleaned_values[word_counts.index(min_word_count)]

            class_mapping[key] = str.lower(min_word_value)

        return class_mapping
    
    

    # Get item
    def __getitem__(self, i):
        
        
        # Process EEG
        eeg = self.data[i]["eeg"].float()


        eeg = stack_and_truncate_eeg(eeg[self.index_list,:])
        
        
        # eeg = stack_and_truncate_eeg(eeg[32:64,:])

        # eeg=torch.tensor(self.transform(eeg)['eeg'])

        eeg = F.interpolate(torch.tensor(eeg).unsqueeze(1), size=512, mode='linear', align_corners=False).squeeze(1)
        


        assert not torch.isnan(eeg).any(), "EEG after slicing contains NaN"
        assert not torch.isinf(eeg).any(), "EEG after slicing contains inf"




        label = self.data[i]["label"]
        image = self.data[i]["image"]
        image_path=f"{self.image_fir}/{self.images_name[image]}.JPEG"



        if not os.path.exists(image_path):
            return self.__getitem__(i+1)

        image_path = os.path.relpath(image_path, self.project_root)
        return {
            "eeg": eeg,
            "label": label,
            "image": image,
            "label_name": self.class_dict[self.labels_name[label]],
            "image_path": image_path
        }

# Splitter class
class Splitter:
    def __init__(self, dataset, split_path, split_num=0, split_name="train"):
        # Set EEG dataset
        self.dataset = dataset
        # Load split
        loaded = torch.load(split_path)
        self.split_idx = loaded["splits"][split_num][split_name]
        # Filter data
        self.split_idx = [i for i in self.split_idx if 450 <= self.dataset.data[i]["eeg"].size(1) <= 600]
        # Compute size
        self.size = len(self.split_idx)

    # Get size
    def __len__(self):
        return self.size
    # Get item
    def __getitem__(self, i):
        # Get sample from dataset
        return self.dataset[self.split_idx[i]]

def stack_and_truncate_eeg(eeg_tensor, target_length=1000):
    """
    Horizontally stack EEG data and truncate to specified length.

    Args:
        eeg_tensor: EEG data tensor of shape (n_channels, n_samples)
        target_length: Target length in samples (default: 1000 for 1 second at 1kHz)

    Returns:
        Stacked and truncated EEG data of shape (n_channels, target_length)
    """
    # Convert to numpy if input is PyTorch tensor
    eeg_data = eeg_tensor.numpy() if hasattr(eeg_tensor, 'numpy') else np.array(eeg_tensor)

    n_steps = eeg_data.shape[1]

    # Calculate minimum number of repetitions needed
    min_repeats = (target_length + n_steps - 1) // n_steps  # Ceiling division

    # Horizontally stack the data
    stacked_data = np.tile(eeg_data, (1, min_repeats))

    # Check for inf and NaN values in the stacked data
    assert not np.isinf(stacked_data).any(), "Stacked data contains inf"
    assert not np.isnan(stacked_data).any(), "Stacked data contains NaN"

    # Truncate to target length
    return stacked_data[:, :target_length]


# Create datasets for different subjects
# Based on actual data: subjects 1-6 exist, using subject 6 as cross, 1-5 as train/test
ds_train_test=EEGDataset(subjects=[1,2,3,4,5])
ds_cross=EEGDataset(subjects=[6])

class CrossSplitter:
    """
    Custom splitter for cross-subject validation.

    Filters EEG samples to include only those with valid temporal length (450-600 samples).
    This ensures all samples can be properly resampled to the target 512Hz × 1 second format.
    """
    def __init__(self, dataset):
        self.dataset = dataset
        # Filter data to include only samples with valid EEG length
        self.split_idx = [i for i in range(len(dataset.data)) if 450 <= dataset.data[i]["eeg"].size(1) <= 600]
        self.size = len(self.split_idx)
        print(f"CrossSplitter: Found {self.size} valid samples for subject 6")
        if self.size == 0:
            print("Warning: No valid samples found for subject 6!")
            print(f"Total samples in dataset: {len(dataset.data)}")
            if len(dataset.data) > 0:
                print(f"Sample EEG shape: {dataset.data[0]['eeg'].size()}")

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        return self.dataset[self.split_idx[i]]

sp_cross = CrossSplitter(ds_cross)



cache_dict={}

def process(sp):
    """
    Process ImageNetEEG dataset split and extract CLIP image features.

    Args:
        sp: Splitter object containing filtered EEG samples

    Returns:
        Tuple of (eegs, labels, text_features, captions, image_paths)
    """
    verbose=False
    img_model.to('cuda')
    img_model.eval()
    eegs_total, label_total, text_features_total, captions_total,image_path_total = [], [], [], [],[]
    
    # Check if splitter has any data
    if len(sp) == 0:
        print(f"Warning: Splitter has no data! Returning empty arrays.")
        img_model.cpu()
        return np.array([]), np.array([]), np.array([]), [], []
    
    for i in tqdm(range(len(sp))):
        eeg_data_tensor=sp[i]['eeg']

        assert not np.isnan(eeg_data_tensor).any(), "EEG after slicing contains NaN"
        assert not np.isinf(eeg_data_tensor).any(), "EEG after slicing contains inf"

        label=sp[i]['label']
        img_path=sp[i]['image_path']
        captions = 'This is a '+sp[i]['label_name']
        if img_path not in cache_dict:
            real_img_path=os.path.join(get_wavemind_root(),img_path)
            image = Image.open(real_img_path)
            inputs = img_processor(images=image, return_tensors="pt").to('cuda')
            with torch.no_grad():
                outputs = img_model(**inputs)
            image_festure = outputs.image_embeds
            # L2 normalize to unit hypersphere for CLIP alignment
            image_festure = F.normalize(image_festure, p=2, dim=-1)
            cache_dict[img_path]=image_festure.cpu()
        else:
            if verbose:
                print('hit cache',i)
                verbose=False
            image_festure=cache_dict[img_path].cpu()

        eegs_total.append(np.array(eeg_data_tensor))
        label_total.append(int(label))
        text_features_total.append(np.array(image_festure.cpu().detach().numpy()))
        captions_total.append(captions)
        assert len(img_path)>0
        image_path_total.append(img_path)

        torch.cuda.empty_cache()


    for i in range(len(eegs_total)):
        assert not np.isnan(eegs_total[i]).any(), "EEG after slicing contains NaN"
        assert not np.isinf(eegs_total[i]).any(), "EEG after slicing contains inf"

    # Only stack if we have data
    if len(eegs_total) > 0:
        eegs_total=np.stack(eegs_total)

        for i in range(len(eegs_total)):
            assert not np.isnan(eegs_total[i]).any(), "EEG after slicing contains NaN"
            assert not np.isinf(eegs_total[i]).any(), "EEG after slicing contains inf"

        text_features_total=np.stack(text_features_total).squeeze()
    else:
        eegs_total = np.array([])
        text_features_total = np.array([])

    img_model.cpu()
    assert len(eegs_total) == len(label_total) == len(text_features_total) == len(captions_total)
    return eegs_total, np.array(label_total), text_features_total, captions_total,image_path_total


# Process and save cross dataset
cross_eegs, cross_labels, cross_text_features, cross_captions, cross_image_path_total = process(sp_cross)

# Only save cross data if we have valid samples
if len(cross_eegs) > 0:
    # Use Convert_and_Save for cross data
    cross_item = Convert_and_Save()
    cross_item.save_to_hdf5_new(
        eegdata=cross_eegs,
        text_feature=cross_text_features,
        caption=cross_captions,
        label=cross_labels,
        dataset_name=f'{ds_name}_cross',
        hdf_path=hdf5_path,
        image_paths=cross_image_path_total
    )
    print(f"Cross dataset saved with {len(cross_eegs)} samples")
else:
    print("Warning: No cross data to save!")

print("ImageNetEEG cross dataset processing complete!")
print(f"Note: CLIP groundtruth files are generated via data/preprocess_wavemind.py --rag-only")




