from time import sleep

import torch
import os

from kaggle.rest import ApiException

os.environ["HF_HOME"] = "../../../../.cache/HF"
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

class EEGDataset:

    # Constructor
    def __init__(self, eeg_signals_path='eeg_signals_raw_with_mean_std.pth', subjects=None):
        # "subjects" can be [0,1,2,3,4,5,6,7,8,9] representing the subject number
        # Load EEG signals
        loaded = torch.load(eeg_signals_path,weights_only=False)
        if subjects is not None:
            self.data = [loaded['dataset'][i] for i in range(len(loaded['dataset'])) if loaded['dataset'][i]['subject'] in subjects]
        else:
            self.data = loaded['dataset']
        self.labels_name = loaded["labels"]
        self.images_name = loaded["images"]

        # Compute size
        self.size = len(self.data)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Process EEG
        eeg = self.data[i]["eeg"].float()
        label = self.data[i]["label"]
        image = self.data[i]["image"]
        return {
            "eeg": eeg,
            "label": label,
            "image": image,
            "label_name": self.labels_name[label],
            "image_name": self.images_name[image]
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
        eeg, label = self.dataset[self.split_idx[i]]
        # Return
        return eeg, label

if __name__ == "__main__":
    ds = EEGDataset()
    from tqdm import tqdm
    from kaggle.api.kaggle_api_extended import KaggleApi
    import os

    api = KaggleApi()
    api.authenticate()


    def download_file_with_retry(competition, file_name, path):
        try:
            api.competition_download_file(
                competition=competition,
                file_name=file_name,
                path=path,
                quiet=False,
                force=True
            )
            sleep(15)
        except ApiException as e:
            print(e)
            sleep(60*15)
            download_file_with_retry(competition, file_name, path)


    image_name_set = set()

    for i in range(len(ds)):
        image_name = ds[i]['image_name']
        label_name = ds[i]['label_name']
        image_name_set.add(image_name)

    # %%
    file_names = []
    for image_name in image_name_set:
        label_name = image_name.split("_")[0]
        file_name = f"ILSVRC/Data/CLS-LOC/train/{label_name}/{image_name}.JPEG"
        file_names.append(file_name)
    file_names=sorted(file_names)

    for file_name in tqdm(file_names):
        if os.path.exists(f'Image/{file_name.split("/")[-1]}'):
            continue
        download_file_with_retry("imagenet-object-localization-challenge", file_name, "Image")
