
import warnings
import fcntl
import time
from abc import ABC, abstractmethod
from typing import Dict
from transformers import CLIPTextModelWithProjection, AutoTokenizer
import h5py
import mne

from mne import create_info
from tqdm import tqdm
from transformers import CLIPTextModelWithProjection, AutoTokenizer

from data.preUtils import z_score_normalize, processRaw, eeg_filter_all
#%%
from PIL import Image
from matplotlib_inline.backend_inline import FigureCanvas
import torch
import torch.nn.functional as F
import numpy as np
# --------------------------------------
class DatasetInfo(ABC):
    @abstractmethod
    def electrode_list(self):
        pass
    @abstractmethod
    def samplingFrq(self):
        pass
    @abstractmethod
    def id2label(self):
        pass
    def get_label_by_id(self,id):
        return self.id2label()[id]
    def label2id(self):
        return {v: k for k, v in self.id2label().items()}
    @abstractmethod
    def _captionFunc(self):
        pass
    def _de_captionFunc(self):
        pass
    def get_caption(self, label):
        label_name = self.id2label()[label]
        return self._captionFunc()(label_name)
    def get_label_by_caption(self,caption,**kwargs):
        pass
    @property
    def id_max(self):
        return max(self.id2label().keys())

    @property
    def id_min(self):
        return min(self.id2label().keys())

    def get_captions(self,labels):
        return [self.get_caption(label) for label in labels]
    def get_labels_by_captions(self,captions,**kwargs):
        return [self.get_label_by_caption(caption,**kwargs) for caption in captions]
    def get_n_class(self):
        return len(self.id2label())




class BCICIV2aDatasetInfo(DatasetInfo):
    def electrode_list(self):
        return [
            'Fz',
            'FC5',
            'FC3',
            'FCz',
            'FC2',
            'FC4',
            'T7',
            'C3',
            'C1',
            'Cz',
            'C2',
            'C4',
            'T8',
            'CP3',
            'CP1',
            'CPz',
            'CP2',
            'CP4',
            'P3',
            'Pz',
            'P4',
            'Oz',
        ]
    def samplingFrq(self):
        return 1750
    def id2label(self):
        return {
            0: 'left_hand',
            1: 'right_hand',
            2: 'feet',
            3: 'tongue',
        }
    def _captionFunc(self):
        return generate_caption_motor_imaging
    def _de_captionFunc(self):
        return degenerate_caption_motor_imaging
    def get_caption(self, label):
        action = self.id2label()[label]
        return self._captionFunc()(action)
    def get_label_by_caption(self,caption,**kwargs):
        return self._de_captionFunc()(caption)


class HMCDatasetInfo(DatasetInfo):
    def electrode_list(self):
        return [
            'F4', 'C4',
            'O2', 'C3'
        ]
    def samplingFrq(self):
        return 256

    def id2label(self):
        return {
            0: 'Sleep stage W',
            1: 'Sleep stage N1',
            2: 'Sleep stage N2',
            3: 'Sleep stage N3',
            4: 'Sleep stage R',
        }

    def _captionFunc(self):
        return generate_caption_sleep_stage

    def get_caption(self, label):
        label_name = self.id2label()[label]
        if label_name == "Sleep stage W":
            return self._captionFunc()(
                label_name) + " which is characterized by alertness, active brain waves, and occasional eye movements, where the person is fully conscious and aware of their surroundings."
        elif label_name == "Sleep stage N1":
            return self._captionFunc()(
                label_name) + " which is the lightest stage of sleep, where the person is in a light sleep and can be easily awakened."
        elif label_name == "Sleep stage N2":
            return self._captionFunc()(
                label_name) + " which is a deeper stage of sleep, where the person is less responsive to external stimuli and has slower brain waves."
        elif label_name == "Sleep stage N3":
            return self._captionFunc()(
                label_name) + " which is the deepest stage of sleep, where the person is in a deep sleep and is difficult to awaken."
        elif label_name == "Sleep stage R":
            return self._captionFunc()(
                label_name) + " which is the stage of sleep where most dreaming occurs, characterized by rapid eye movements and increased brain activity."
        else:
            raise ValueError(f"Unknown label when get_caption_P2018Dataset: {label}")

    def _de_captionFunc(self):
        return degenerate_caption_sleep_stage

    def get_label_by_caption(self, caption, **kwargs):
        return self._de_captionFunc()(caption)

class FACEDatasetInfo(DatasetInfo):
    def electrode_list(self):
        return [
            'Fp1',
            'Fp2',
            'Fz',
            'F3',
            'F4',
            'F7',
            'F8',
            'FC1',
            'FC2',
            'FC5',
            'FC6',
            'Cz',
            'C3',
            'C4',
            'T7',
            'T8',
            'CP1',
            'CP2',
            'CP5',
            'CP6',
            'Pz',
            'P3',
            'P4',
            'P7',
            'P8',
            'PO3',
            'PO4',
            'Oz',
            'O1',
            'O2',
        ]
    def samplingFrq(self):
        return 250
    def id2label(self):
        return self.id2label_emotion()
    # def id2label_valence(self):
    #     return {
    #         1: 'positive',
    #         0: 'neutral',
    #         -1: 'negative',
    #     }
    def id2label_emotion(self):
        return {
            0: 'anger',
            1: 'disgust',
            2: 'fear',
            3: 'sadness',
            4: 'neutral',
            5: 'amusement',
            6: 'inspiration',
            7: 'joy',
            8: 'tenderness',
        }
    def get_caption(self, label):
        emotion = label
        # label_valence = self.id2label_valence()[valence]
        label_emotion = self.id2label_emotion()[emotion]
        return f"He is feeling {label_emotion}. He may seen something {label_emotion} that makes him feel {label_emotion}."


    def _captionFunc(self):
        return None
    def _de_captionFunc(self):
        return None

    def get_labels_by_captions(self, captions):
        label2id_emotion = {v: k for k, v in self.id2label_emotion().items()}
        for label in label2id_emotion:
            if label in captions:
                return label2id_emotion[label]

class ImageNetEEGDatasetInfo(DatasetInfo):
    def electrode_list(self):
            return [
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2',
        'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2',
        'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz',
        'O2', 'PO10', 'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6',
        'FT9', 'FT7', 'FC3', 'FC4', 'FT8', 'FT10', 'C5', 'C1', 'C2', 'C6',
        'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'P5', 'P1', 'P2', 'P6', 'PO7',
        'PO3', 'POz', 'PO4', 'PO8', 'Fpz', 'F9'
    ]
    def samplingFrq(self):
        return 1000
    def id2label(self):
        return {10: 'canoe', 30: 'grand', 25: 'folding chair', 18: 'broom', 3: 'anemone fish', 8: 'panda', 11: 'lycaenid', 28: 'electric guitar', 38: 'radio telescope', 20: 'missile', 23: 'mailbag', 0: 'sorrel', 34: 'digital watch', 39: 'egyptian cat', 21: 'capuchin', 6: 'off-roader', 26: 'pajama', 1: 'parachute', 27: 'mitten', 37: 'electric locomotive', 19: 'pizza', 9: 'daisy', 12: 'alsatian', 35: 'african elephant', 29: 'reflex camera', 17: 'desktop computer', 32: 'banana', 5: 'coffee mug', 31: 'mountain tent', 2: 'iron', 33: 'bolete', 36: 'airliner', 24: 'convertible', 13: 'running shoe', 7: 'revolver', 4: 'espresso maker', 14: "jack-o'-lantern", 15: 'cellphone', 22: 'pool table', 16: 'golf ball'}
    @property
    def id_min(self):
        return 0
    @property
    def id_max(self):
        return 39

    def _captionFunc(self):
        raise NotImplemented
    def _de_captionFunc(self):
        raise NotImplemented
    def get_caption(self, label):
        raise NotImplemented

class P2018DatasetInfo(DatasetInfo):
    def electrode_list(self):
        return [
            'F3',
            'F4',
            'C3',
            'C4',
            'O1',
            'O2',
        ]
    def samplingFrq(self):
        return 100
    def id2label(self):
        return {
            0: 'Sleep stage W',
            1: 'Sleep stage N1',
            2: 'Sleep stage N2',
            3: 'Sleep stage N3',
            4: 'Sleep stage R',
        }
    def _captionFunc(self):
        return generate_caption_sleep_stage

    def get_caption(self, label):
        label_name = self.id2label()[label]
        if label_name == "Sleep stage W":
            return self._captionFunc()(label_name)+" which is characterized by alertness, active brain waves, and occasional eye movements, where the person is fully conscious and aware of their surroundings."
        elif label_name == "Sleep stage N1":
            return self._captionFunc()(label_name)+" which is the lightest stage of sleep, where the person is in a light sleep and can be easily awakened."
        elif label_name == "Sleep stage N2":
            return self._captionFunc()(label_name)+" which is a deeper stage of sleep, where the person is less responsive to external stimuli and has slower brain waves."
        elif label_name == "Sleep stage N3":
            return self._captionFunc()(label_name)+" which is the deepest stage of sleep, where the person is in a deep sleep and is difficult to awaken."
        elif label_name == "Sleep stage R":
            return self._captionFunc()(label_name)+" which is the stage of sleep where most dreaming occurs, characterized by rapid eye movements and increased brain activity."
        else:
            raise ValueError(f"Unknown label when get_caption_P2018Dataset: {label}")

    def _de_captionFunc(self):
        return degenerate_caption_sleep_stage
    def get_label_by_caption(self,caption,**kwargs):
        return self._de_captionFunc()(caption)

class SleepEDFxDatasetInfo(DatasetInfo):
    def electrode_list(self):
        return [
            'Fpz',
            'Pz',
        ]
    def samplingFrq(self):
        return 100
    def id2label(self):
        return {
            0: 'Sleep stage W',
            1: 'Sleep stage N1',
            2: 'Sleep stage N2',
            3: 'Sleep stage N3',
            4: 'Sleep stage R',
        }
    def _captionFunc(self):
        return generate_caption_sleep_stage
    def _de_captionFunc(self):
        return degenerate_caption_sleep_stage
    def get_caption(self, label):
        label_name = self.id2label()[label]
        if label_name == "Sleep stage W":
            return self._captionFunc()(label_name)+" which is characterized by alertness, active brain waves, and occasional eye movements, where the person is fully conscious and aware of their surroundings."
        elif label_name == "Sleep stage N1":
            return self._captionFunc()(label_name)+" which is the lightest stage of sleep, where the person is in a light sleep and can be easily awakened."
        elif label_name == "Sleep stage N2":
            return self._captionFunc()(label_name)+" which is a deeper stage of sleep, where the person is less responsive to external stimuli and has slower brain waves."
        elif label_name == "Sleep stage N3":
            return self._captionFunc()(label_name)+" which is the deepest stage of sleep, where the person is in a deep sleep and is difficult to awaken."
        elif label_name == "Sleep stage R":
            return self._captionFunc()(label_name)+" which is the stage of sleep where most dreaming occurs, characterized by rapid eye movements and increased brain activity."
        else:
            raise ValueError(f"Unknown label when get_caption_SleepEDFxDataset: {label}")
    def get_label_by_caption(self,caption,**kwargs):
        return self._de_captionFunc()(caption)


class SEEDIVDatasetInfo(DatasetInfo):
    def electrode_list(self):
        return [
            "Fp1", "Fpz", "Fp2", "AF3", "AF4", "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
            "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3",
            "C1", "Cz", "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CpZ", "CP2",
            "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", "PO7",
            "PO5", "PO3", "POz", "PO4", "PO6", "PO8", "CB1", "O1", "Oz", "O2", "CB2"
        ]
    def samplingFrq(self):
        return 200
    def id2label(self):
        return {
            0: 'neutral',
            1: 'sad',
            2: 'fear',
            3: "happy"
        }
    def _captionFunc(self):
        return generate_caption_emotion_recognition_with_singleLabel
    def _de_captionFunc(self):
        def degenerate_SEED(caption):
            if not any(label in caption for label in ['neutral', 'sad', 'fear', 'happy']):
                raise ValueError(
                    f"Unknown label when degenerate_caption_emotion_recognition_with_valenceNarousal: {caption}")
            if 'neutral' in caption:
                return 0
            elif 'sad' in caption:
                return 1
            elif 'fear' in caption:
                return 2
            else:
                return 3
        return degenerate_SEED
    def pre_hook(self,raw):
        return raw.drop_channels(['CB1','CB2','CpZ'])
    def get_caption(self, label):
        label_name = self.id2label()[label]
        if label_name=="neutral":
            return self._captionFunc()(label_name)+" It shows a calm, peaceful, quiet, everyday activity, serene landscape, and a peaceful atmosphere"
        elif label_name=="sad":
            return self._captionFunc()(label_name)+" It shows a loneliness, grief, tears, memories and sorrowful atmosphere"
        elif label_name=="fear":
            return self._captionFunc()(label_name)+" It shows a fear, anxiety, panic, horror, and a scary atmosphere"
        elif label_name=="happy":
            return self._captionFunc()(label_name)+" It shows a joy, happiness, smile, laughter, and a cheerful atmosphere"
        else:
            raise ValueError(f"Unknown label when get_caption_SEEDIV: {label}")





class TUABDatasetInfo(DatasetInfo):
    def electrode_list(self):
        return ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz']
    def samplingFrq(self):
        return 512
    def id2label(self):
        return {
            0:'abnormal',
            1:'normal'
        }
    def _captionFunc(self):
        raise NotImplemented

    def get_caption(self, label):
        if label==0:
            return "Abnormal EEG: Exhibits deviations such as slowed/fast rhythms, asymmetry, or with irregular amplitudes. May include spikes, sharp waves, or spike-wave complexes, or so on."
        elif label==1:
            return "Normal EEG: Characterized by rhythmic patterns with stable amplitudes. The brain wave is normal and stable."

class TUEVDatasetInfo(DatasetInfo):
    def electrode_list(self):
        return ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz']
    def samplingFrq(self):
        return 512
    def id2label(self):
        return {
            0:'spike and sharp wave (SPSW)',
            1:'generalized periodic epileptiform discharges (GPED)',
            2:'periodic lateralized epileptiform discharges (PLED)',
            3:'eye movement (EYEM)',
            4:'artifact (ARTF)',
            5:'background (BCKG)'
        }
    def get_caption(self, label):
        if label==0:
            return 'SPSW (Spike and Sharp Wave): Abrupt, high-amplitude transients with epileptiform morphology, typically localized to focal regions. It has sharp rising phases and asymmetric waveforms.'
        elif label==1:
            return 'GPED (Generalized Periodic Epileptiform Discharges): Exhibits repetitive, bilateral synchronous spikes/sharp waves with uniform morphology. It has symmetric distribution.'
        elif label==2:
            return 'PLED (Periodic Lateralized Epileptiform Discharges): Unilateral or bilateral asynchronous periodic discharges, with "stereotyped" waveforms. It has lateralized localization.'
        elif label==3:
            return 'EYEM (Eye Movement): Low-frequency, high-amplitude frontal slow waves synchronized with blinks or saccades. It has rhythmicity, frontal dominance, and correlation with eye movement artifacts.'
        elif label==4:
            return 'ARTF (Artifact): Non-physiological signals (e.g., muscle noise, electrode pops) with abrupt amplitude shifts or high-frequency content. '
        elif label==5:
            return 'BCKG (Background): Stable, symmetric rhythms with age-appropriate amplitude. Absent epileptiform features, with preserved reactivity to stimuli.'
        else:
            raise ValueError(f"Unknown label when get_caption_TUEV: {label}")
    def _captionFunc(self):
        raise NotImplemented
    def get_n_class(self):
        return len(self.id2label())




class SEEDDatasetInfo(DatasetInfo):
    def electrode_list(self):
        return [
            "Fp1", "Fpz", "Fp2", "AF3", "AF4", "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
            "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3",
            "C1", "Cz", "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CpZ", "CP2",
            "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", "PO7",
            "PO5", "PO3", "POz", "PO4", "PO6", "PO8", "CB1", "O1", "Oz", "O2", "CB2"
        ]
    def samplingFrq(self):
        return 200
    def id2label(self):
        return {
            2:'positive',
            0:'negative',
            1: "neutral"
        }
    def _captionFunc(self):
        return generate_caption_emotion_recognition_with_singleLabel
    def _de_captionFunc(self):
        def degenerate_SEED(caption):
            if not any(label in caption for label in ['positive', 'neutral', 'negative']):
                raise ValueError(
                    f"Unknown label when degenerate_caption_emotion_recognition_with_valenceNarousal: {caption}")
            if 'positive' in caption:
                return 2
            elif 'neutral' in caption:
                return 1
            else:
                return 0
        return degenerate_SEED
    def pre_hook(self,raw):
        return raw.drop_channels(['CB1','CB2','CpZ'])
    def get_caption(self, label):
        label_name = self.id2label()[label]
        if label_name=="positive":
            return self._captionFunc()(label_name)+" It shows a joy, laughter, happiness, bright colors, celebration, warmth atmosphere"
        elif label_name=="neutral":
            return self._captionFunc()(label_name)+" It shows a calm, peaceful, quiet, everyday activity, serene landscape, and a peaceful atmosphere"
        elif label_name=="negative":
            return self._captionFunc()(label_name)+" It shows a sadness, loneliness, tears, grief, sorrowful atmosphere"
        else:
            raise ValueError(f"Unknown label when get_caption_SEED: {label_name}")





class SienaDatasetInfo(DatasetInfo):
    def electrode_list(self):
        return ['Fp1', 'F3', 'C3', 'P3', 'O1', 'F7', 'T3', 'T5', 'F9', 'Fz',  'Pz', 'F4', 'C4', 'P4', 'O2', 'F8', 'T4', 'T6', 'F10']
    def samplingFrq(self):
        return 512
    def id2label(self):
        return {
            0:'Seizure',
            1:'Background'
        }
    def _captionFunc(self):
        raise NotImplemented
    def _de_captionFunc(self):
        raise NotImplemented
    def get_caption(self, label):
        label_name = self.id2label()[label]
        if label_name=="Seizure":
            return "Seizure happen: Shows abnormal, paroxysmal EEG discharges."
        elif label_name=="Background":
            return "No Seizure happen: Stable and symmetric brain rhythms that remain consistent over time."
        else:
            raise ValueError(f"Unknown label when get_caption_SEED: {label_name}")
        
        
class CHBMITDatasetInfo(DatasetInfo):
    def electrode_list(self):
        return [
            "FP1-F7",
            "F7-T7",
            "T7-P7",
            "P7-O1",
            "FP2-F8",
            "F8-T8",
            "T8-P8",
            "P8-O2",
            "FP1-F3",
            "F3-C3",
            "C3-P3",
            "P3-O1",
            "FP2-F4",
            "F4-C4",
            "C4-P4",
            "P4-O2",
        ]
    def samplingFrq(self):
        return 512
    def id2label(self):
        return {
            0:'Seizure',
            1:'Background'
        }
    def _captionFunc(self):
        raise NotImplemented
    def _de_captionFunc(self):
        raise NotImplemented
    def get_caption(self, label):
        label_name = self.id2label()[label]
        if label_name=="Seizure":
            return "Seizure happen: Shows abnormal, paroxysmal EEG discharges."
        elif label_name=="Background":
            return "No Seizure happen: Stable and symmetric brain rhythms that remain consistent over time."
        else:
            raise ValueError(f"Unknown label when get_caption_CHBMIT: {label_name}")
    












def extract_element_from_hdf5(dataset_name, index,path='../Total/data_label.h5'):
    """
    Extract the i-th element from the specified dataset in an HDF5 file.

    Parameters:
        path (str): Path to the HDF5 file.
        dataset_name (str): Name of the dataset.
        index (int): Index of the element to extract.

    Returns:
        dict: A dictionary containing 'eeg_data', 'text_feature', and 'caption' for the specified index.
    """
    with h5py.File(path, 'r') as hdf5_file:
        if dataset_name not in hdf5_file:
            raise ValueError(f"Dataset '{dataset_name}' not found in file '{path}'.")

        dataset = hdf5_file[dataset_name]

        if index < 0 or index >= len(dataset):
            raise IndexError(f"Index {index} is out of bounds for dataset '{dataset_name}' with length {len(dataset)}.")

        # Extract data
        element = dataset[index]
        eeg_data = element['eeg_data']
        text_feature = element['text_feature']
        caption = element['caption'].decode('utf-8').strip()  # Decode and strip padding spaces

        return {
            'eeg_data': eeg_data,
            'text_feature': text_feature,
            'caption': caption
        }

def extract_all_from_hdf5(dataset_name,path='../Total/data_label.h5'):
    """
    Extract all elements from the specified dataset in an HDF5 file.

    Parameters:
        path (str): Path to the HDF5 file.
        dataset_name (str): Name of the dataset.

    Returns:
        dict: A dictionary containing 'eeg_data', 'text_feature', and 'caption' for all elements.
    """
    with h5py.File(path, 'r') as hdf5_file:
        if dataset_name not in hdf5_file:
            raise ValueError(f"Dataset '{dataset_name}' not found in file '{path}'.")

        dataset = hdf5_file[dataset_name]

        # Extract data
        eeg_data = dataset['eeg_data']
        caption = dataset['caption']
        new_captions=[]
        for i in range(len(caption)):
            new_captions.append(str(caption[i].decode('utf-8').strip()))
        return {
            'eeg_data': eeg_data.astype(np.float32),
            'caption': new_captions
        }

class Convert_and_Save():
    def __init__(self):
        self.feature_cache = {}

        
    def convert_to_feature_new_new(self, labels, ds_info:DatasetInfo=None):
        text_model = CLIPTextModelWithProjection.from_pretrained('openai/clip-vit-large-patch14-336')
        text_model.to('cuda')
        text_tokenizer = AutoTokenizer.from_pretrained('openai/clip-vit-large-patch14-336')

        text_features = []
        captions = []

        if ds_info is None:
            raise ValueError("ds_info is None")


    
        if type(labels) is np.ndarray:
            for i, label in enumerate(tqdm(labels)):
                assert label <= ds_info.id_max and label >= ds_info.id_min and label >= 0, f"label out of range,as label is {label}, but id_min is {ds_info.id_min}, id_max is {ds_info.id_max}"
                # 
                caption = ds_info.get_caption(label)
                # if i == 0:
                #     print(caption)

                if caption in self.feature_cache:
                    text_feature = self.feature_cache[caption]
                else:
                    text_inputs = text_tokenizer(caption, padding=True, return_tensors="pt").to('cuda')
                    text_feature = text_model(**text_inputs).text_embeds
                    # L2 normalize to unit hypersphere for CLIP alignment
                    text_feature = F.normalize(text_feature, p=2, dim=-1)
                    self.feature_cache[caption] = text_feature.detach().cpu().numpy()

                text_features.append(self.feature_cache[caption])
                assert text_feature.shape[-1] == 768
                captions.append(caption)
        else:
            raise TypeError("Unsupported dataset type. Expected np.ndarray.")

        del text_model
        torch.cuda.empty_cache()

        return np.concatenate(text_features, axis=0), captions

    def _acquire_file_lock(self, file_path, timeout=30, retry_interval=0.1):
        """
        Acquire an exclusive file lock with timeout and retry mechanism.
        
        Parameters:
            file_path (str): Path to the file to lock
            timeout (float): Maximum time to wait for lock in seconds
            retry_interval (float): Time between retry attempts in seconds
            
        Returns:
            file object: The locked file object
        """
        lock_file_path = file_path + '.lock'
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # Try to create and lock the lock file
                lock_file = open(lock_file_path, 'w')
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return lock_file
            except (IOError, BlockingIOError):
                # Lock is held by another process, wait and retry
                if lock_file:
                    lock_file.close()
                time.sleep(retry_interval)
        
        raise TimeoutError(f"Could not acquire lock for {file_path} within {timeout} seconds")

    def _release_file_lock(self, lock_file):
        """
        Release the file lock and clean up the lock file.
        
        Parameters:
            lock_file: The locked file object to release
        """
        if lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()
            # Remove the lock file
            import os
            if os.path.exists(lock_file.name):
                os.remove(lock_file.name)

    def save_to_hdf5_new(self, eegdata, text_feature, caption, label, dataset_name, image_paths=None,hdf_path='../Total/data_label.h5', ):
        """
        Save eegdata, text_feature, captions, and labels to an HDF5 file with file locking.
        Supports multiple concurrent write operations by using file locking mechanism.

        Parameters:
            eegdata (numpy.ndarray): EEG data array.
            text_feature (numpy.ndarray): Text feature data array.
            caption (list[str]): List of captions.
            label (numpy.ndarray): Label data array.
            dataset_name (str): Name of the dataset.
            path (str): Path to the HDF5 file.
            image_paths (list[str], optional): List of image paths. If provided, will be saved to HDF5.
        """
        caption_length = 192  # Fixed caption length
        path_length = 192  # Fixed path length for image paths
        num_samples = eegdata.shape[0]
        
        assert eegdata.std() > 0.01, "EEG data unit should be in uV"
        
        # Validate input consistency
        if image_paths is not None:
            assert eegdata.shape[0] == text_feature.shape[0] == len(caption) == label.shape[0] == len(image_paths), "Mismatched input sizes."
        else:
            assert eegdata.shape[0] == text_feature.shape[0] == len(caption) == label.shape[0], "Mismatched input sizes."

        # Prepare data types for structured array
        eeg_dtype = np.float32
        text_dtype = np.float16
        caption_dtype = f'S{caption_length}'
        label_dtype = label.dtype
        path_dtype = f'S{path_length}'

        # Acquire file lock before accessing HDF5 file
        lock_file = None
        try:
            # Acquire exclusive lock for HDF5 file operations
            lock_file = self._acquire_file_lock(hdf_path, timeout=1800, retry_interval=0.5)
            
            # Process all data at once with file locking
            with h5py.File(hdf_path, 'a', locking=True, swmr=True) as hdf5_file:
                dataset_key = dataset_name
                existing_dataset = hdf5_file.get(dataset_key, None)

                # Validate data
                assert not np.isnan(eegdata).any(), "NaN in EEG data"
                assert not np.isinf(eegdata).any(), "Inf in EEG data"
                assert eegdata.shape[1] == 32 and eegdata.shape[2] in (512, 513), "Invalid EEG shape"
                if eegdata.shape[2] == 513:
                    eegdata = eegdata[:, :, :512]
                
                # Convert captions to fixed-length strings
                caption_array = np.array([c[:caption_length].ljust(caption_length) for c in caption], dtype=caption_dtype)
                
                # Convert image paths to fixed-length strings if provided
                if image_paths is not None:
                    image_path_array = np.array([p[:path_length].ljust(path_length) for p in image_paths], dtype=path_dtype)
                else:
                    # Create empty image path array if not provided
                    image_path_array = np.array([''.ljust(path_length)] * num_samples, dtype=path_dtype)

                # Create structured array
                dtype_fields = [
                    ('eeg_data', eeg_dtype, eegdata.shape[1:]),
                    ('text_feature', text_dtype, text_feature.squeeze().shape[1:]),
                    ('caption', caption_dtype),
                    ('label', label_dtype, label.shape[1:]),
                    ('image_path', path_dtype)
                ]
                dtype = np.dtype(dtype_fields)
                structured_data = np.zeros(num_samples, dtype=dtype)
                structured_data['eeg_data'] = eegdata.astype(eeg_dtype)
                structured_data['text_feature'] = text_feature.squeeze().astype(text_dtype)
                structured_data['caption'] = caption_array
                structured_data['label'] = label
                structured_data['image_path'] = image_path_array

                # Write to HDF5
                if existing_dataset is None:
                    existing_dataset = hdf5_file.create_dataset(
                        dataset_key, data=structured_data,
                        maxshape=(None,), chunks=(64,), compression='gzip'
                    )
                    print(f"Created new dataset: {dataset_key}")
                else:
                    existing_dataset.resize(existing_dataset.shape[0] + num_samples, axis=0)
                    existing_dataset[-num_samples:] = structured_data
                    print(f"Appended {num_samples} samples. Total: {existing_dataset.shape[0]}")

                print(f"Dataset {dataset_key} updated successfully.")
                
        finally:
            # Always release the file lock
            if lock_file:
                self._release_file_lock(lock_file)

    def process_and_save(self, ds_info, eegdata, label, dataset_name, path='../Total/data_label.h5', image_paths=None):
        text_feature, captions = self.convert_to_feature_new_new(label, ds_info)
        self.save_to_hdf5_new(eegdata, text_feature, captions, label, dataset_name, image_paths, hdf_path=path)




def plot_eeg_and_psds_side_by_side_optimized(eeg_data):
    # Deferred import to avoid side effects on logging at module load time
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ch_names = [
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
        'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8',
        'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8',
        'POz', 'O1', 'Oz', 'O2', 'AFz', 'CPz', 'FCz'
    ]
    sfreq = 256
    eeg_data = torch.tensor(eeg_data, device='cuda')  # Move data to GPU
    if len(ch_names) != eeg_data.shape[0]:
        raise ValueError("The number of channel names must match the number of channels in eeg_data, which eeg channels len is {} and ch_names len is {}".format(eeg_data.shape[0], len(ch_names)))

    n_channels = len(ch_names)

    # Create time and frequency vectors
    times = torch.arange(eeg_data.shape[1], device=eeg_data.device) / sfreq
    n_fft = eeg_data.shape[1]
    freqs = torch.linspace(0, sfreq / 2, n_fft // 2 + 1, device=eeg_data.device)

    # Calculate FFT and PSDs on GPU
    fft_result = torch.fft.rfft(eeg_data, dim=1)
    psds = torch.abs(fft_result) ** 2
    psds_db = 10 * torch.log10(psds + 1e-10)

    # Move data to CPU for plotting
    times = times.cpu().numpy()
    freqs = freqs.cpu().numpy()
    eeg_data_cpu = eeg_data.cpu().numpy()
    psds_db_cpu = psds_db.cpu().numpy()

    # Prepare for plotting
    fig_width = 10
    fig_height_per_channel = 0.5
    fig, axes = plt.subplots(n_channels, 2, figsize=(fig_width, fig_height_per_channel * n_channels), dpi=100)

    # Batch plot EEG and PSDs
    for idx, (eeg_ax, psd_ax) in enumerate(axes):
        eeg_ax.plot(times, eeg_data_cpu[idx], color="black", linewidth=0.8)
        eeg_ax.set_ylim(-2, 2)
        eeg_ax.set_ylabel(ch_names[idx])
        eeg_ax.spines['top'].set_visible(False)
        eeg_ax.spines['right'].set_visible(False)
        eeg_ax.spines['bottom'].set_visible(False)
        eeg_ax.spines['left'].set_visible(False)
        eeg_ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        eeg_ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        if idx == 0:
            eeg_ax.set_title('EEG Time Series Data')

        psd_ax.plot(freqs, psds_db_cpu[idx], color="black", linewidth=0.8)
        psd_ax.spines['top'].set_visible(False)
        psd_ax.spines['right'].set_visible(False)
        psd_ax.spines['bottom'].set_visible(False)
        psd_ax.spines['left'].set_visible(False)
        psd_ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        psd_ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        if idx == 0:
            psd_ax.set_title('Power Spectral Density (PSD)')

    plt.tight_layout(h_pad=0, w_pad=0)


    canvas = FigureCanvas(fig)
    canvas.draw()  # Render the figure

    # Convert to PIL.Image
    pil_image = Image.frombytes(
        "RGB",
        canvas.get_width_height(),
        canvas.tostring_rgb()
    )

    plt.close(fig)

    return np.array(pil_image)


def convert_to_feature_lagacy(eeg_data,captions):
    eeg_data=z_score_normalize(eeg_data)
    image_array=[]
    for i in tqdm(range(eeg_data.shape[0])):
        image_array.append(plot_eeg_and_psds_side_by_side_optimized(eeg_data[i]))
    assert len(image_array)==len(captions)
    image_array =np.array(image_array)
    return eeg_data,image_array,captions


def processRaw4Label(raw,channel_filter_func=None):
    if channel_filter_func is not None:
        raw=channel_filter_func(raw)
    raw=raw.set_montage('standard_1020')
    raw=processRaw(raw,None)
    return raw

def processEpoch4Label(epochs):
    epochs = epochs.resample(256)
    epochs_data = epochs.get_data()
    epochs_data = z_score_normalize(epochs_data)
    epochs._data = epochs_data
    return epochs

class FilterTransform:
    def __init__(self, dataset_info=None,downsample_fs= False,l_freq=0.5, h_freq=120,notch=True,fs=None,electrode_list=None,verbose=False):
        def rename_eeg_channels(channels):
            return [(channel[0].upper() + channel[1:].lower()) if len(channel)>=2 else channel for channel in channels]
        
        self.l_freq = l_freq  # Low-pass frequency limit in Hz
        self.h_freq = h_freq  # High-pass frequency limit in Hz
        self.downsample_fs = downsample_fs
        if dataset_info is not None:
            self.fs = dataset_info.samplingFrq()
            self.electrode_list = dataset_info.electrode_list()
            if hasattr(dataset_info, 'pre_hook'):
                print("Using pre_hook")
                self.pre_hook = dataset_info.pre_hook
        else:
            self.fs = fs
            self.electrode_list = rename_eeg_channels(electrode_list)
        assert self.fs is not None, "Sampling frequency must be provided."
        assert self.electrode_list is not None, "Electrode list must be provided."
        self.notch=notch
        self.warn_length_already=False
        self.verbose = verbose
        
    

    def __call__(self, eeg: np.ndarray) -> Dict[str, np.ndarray] :
        if eeg.shape[-1]%self.fs!=0:
            raise ValueError(f"Data length is not multiple of {self.fs}Hz, but {eeg.shape[-1]} samples.")
        raw = mne.io.RawArray(eeg.astype(np.float32), mne.create_info(
                ch_names=self.electrode_list,
                sfreq=self.fs,
                ch_types='eeg',
                
            ))
        if not (raw.n_times / self.fs == 1 or raw.n_times / self.fs == 2) and not self.warn_length_already:
            warnings.warn(f"Data length is not 1s or 2s, but {raw.n_times/self.fs}s.")
            self.warn_length_already=True
        if hasattr(self, 'pre_hook'):
            raw=self.pre_hook(raw)
        
        if self.l_freq is not None or self.h_freq is not None or self.notch:
            raw = eeg_filter_all(raw, self.l_freq,self.h_freq, self.notch)
        raw=self.filterInterpolateCh(raw)
        
        if self.downsample_fs:
            raw=raw.resample(self.downsample_fs, n_jobs=-1)
        filtered_eeg = raw.get_data()
        return {'eeg': filtered_eeg}
    
    def filterInterpolateCh(self,raw):
        import mne
        def remove_non_standard_1020_channels(raw,verbose):
            standard_1020 = mne.channels.make_standard_montage('standard_1020').ch_names
            current_channels = raw.info['ch_names']
            if verbose:
                print(f"Current channel:{current_channels}")
            valid_channels = [ch for ch in current_channels if ch in standard_1020]
            removed_channels = [ch for ch in current_channels if ch not in standard_1020]
            if removed_channels and verbose:
                print(f"Remove not standard_1020 channel: {removed_channels}")
            raw = raw.pick(valid_channels)
            
            return raw
        
        def pick_eeg_channels(raw):
            eeg_picks = mne.pick_types(raw.info, eeg=True, meg=False, stim=False)
            return raw.pick(eeg_picks)
        raw = pick_eeg_channels(raw)
        raw = remove_non_standard_1020_channels(raw, self.verbose)

        target_channels = [
            'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
            'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8',
            'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8',
            'POz', 'O1', 'Oz', 'O2', 'AFz', 'CPz', 'FCz'
        ]
        raw_ch_names_lower = [ch.lower() for ch in raw.ch_names]
        missing_channels = [ch for ch in target_channels if ch.lower() not in raw_ch_names_lower]
        if missing_channels:
            new_raw_info = create_info(ch_names=missing_channels, sfreq=raw.info['sfreq'], ch_types='eeg')
            new_raw = mne.io.RawArray(np.zeros((len(missing_channels), raw.n_times)), new_raw_info)
            new_raw = eeg_filter_all(new_raw, lowpass=self.h_freq, highpass=self.l_freq)
            raw = raw.add_channels([new_raw])
            raw.info['bads'].extend(missing_channels)
            assert len(raw.info['bads']) != 0
            assert len(raw.info['bads']) == len(missing_channels)
            raw = raw.set_montage('standard_1020')
            raw = raw.interpolate_bads()
        
        raw = raw.set_montage('standard_1020') 
        raw = raw.pick_channels(target_channels)
        
        
        assert len(raw.ch_names) == len(target_channels)
        return raw


def apply_10_20_mapping(raw):
    """
    Apply standard 10-20 electrode naming convention to raw EDF data.
    Shared by TUAB and TUEV processors.

    Args:
        raw (mne.io.Raw): Raw EDF data

    Returns:
        mne.io.Raw: Data with standardized 10-20 channel names
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


def generate_caption_motor_imaging(action):
    captions={
        'right_hand':"Motor Imagery task. This person is trying to lift his right hand.",
        'left_hand':"Motor Imagery task. This person is trying to lift his left hand.",
        'feet':"Motor Imagery task. This person is trying to move his foot.",
        "foot":"Motor Imagery task. This person is trying to move his foot.",
        'tongue':"Motor Imagery task. This person is trying to move his tongue.",
        'right':"Motor Imagery task. This person is trying to lift his right hand.",
        'left':"Motor Imagery task. This person is trying to lift his left hand.",
    }
    if action not in captions:
        raise ValueError(f"Unknown action when generate_caption_motor_imaging: {action}")
    return captions[action]

def degenerate_caption_motor_imaging(caption):
    if not any(label in caption for label in ['right_hand', 'left_hand', 'feet', 'tongue']):
        raise ValueError(f"Unknown label when degenerate_caption_motor_imaging: {caption}")
    if 'right_hand' in caption:
        return 1
    elif 'left_hand' in caption:
        return 0
    elif 'feet' in caption:
        return 2
    elif 'tongue' in caption:
        return 3
    else:
        raise ValueError(f"Unknown label when degenerate_caption_motor_imaging: {caption}")


def generate_caption_emotion_recognition_with_valenceNarousal(label):
    valence, arousal=label
    if valence.isnumeric() or arousal.isnumeric():
        raise ValueError(f"Convert valence and arousal to string before calling this function.")
    return f"Emotion Recognition Task. This person is watching video with {valence} valence and {arousal} arousal emotion."

def generate_caption_emotion_recognition_with_singleLabel(label):
    if label.isnumeric():
        raise ValueError(f"Convert valence and arousal to string before calling this function.")
    return f"Emotion Recognition Task. This person is watching {label} video."


def degenerate_caption_emotion_recognition_with_valenceNarousal(caption,type='valence'):
    if type=='valence':
        if not any(label in caption for label in ['positive', 'neutral', 'negative']):
            raise ValueError(f"Unknown label when degenerate_caption_emotion_recognition_with_valenceNarousal: {caption}")
        if 'positive' in caption:
            return 1
        elif 'neutral' in caption:
            return 0
        else:
            return -1
    elif type=='arousal':
        if caption not in ['anger', 'disgust', 'fear', 'sadness', 'neutral', 'amusement', 'inspiration', 'joy', 'tenderness']:
            raise ValueError(f"Unknown label: {caption}")
        for emotion in ['anger','disgust','fear','sadness','neutral','amusement','inspiration','joy','tenderness']:
            if emotion in caption:
                return emotion



def generate_caption_sleep_stage(label):
    if label.isnumeric():
        raise ValueError(f"Convert label to string before calling this function.")
    return f"Sleep Stage Recognition Task. This person is in {label}."
def degenerate_caption_sleep_stage(label):
    if not any(label in ['Sleep stage W', 'Sleep stage N1', 'Sleep stage N2', 'Sleep stage N3', 'Sleep stage R']):
        raise ValueError(f"Unknown label when degenerate_caption_sleep_stage: {label}")
    if label == 'Sleep stage W':
        return 0
    elif label == 'Sleep stage N1':
        return 1
    elif label == 'Sleep stage N2':
        return 2
    elif label == 'Sleep stage N3':
        return 3
    elif label == 'Sleep stage R':
        return 4

def load_in_memory(dataset, cut=False,slice_size=512):
    batch_data = []
    labels = []
    
    for i,(eeg_batch, label) in enumerate(tqdm(dataset)):
        
        if i==0 and (eeg_batch.shape[1]!=slice_size) and cut==False:
            warnings.warn("The sample's length is not 1 second, which is {}, and cut is recommened.".format(eeg_batch.shape[1] // slice_size))
        if cut:
            if eeg_batch.shape[1]%slice_size!=0:
                raise ValueError("The length of the eeg data is not a multiple of slice_size")
            n_slices = eeg_batch.shape[1] // slice_size
            for i in range(n_slices):
                batch_data.append(eeg_batch[:, i*slice_size:(i+1)*slice_size])
                labels.extend([label])
        else:
            batch_data.append(eeg_batch)
            labels.append(label)
        
    batch_data_stacked = np.stack(batch_data, axis=0)
    labels_array = np.array(labels)
    assert batch_data_stacked.shape[0] == labels_array.shape[0],"The number of samples in batch_data and labels must match.Which is {} and {}".format(batch_data_stacked.shape[0],labels_array.shape[0])
    return batch_data_stacked, labels_array






def get_each_type_caption_feature(ds_info):
    text_model = CLIPTextModelWithProjection.from_pretrained('openai/clip-vit-large-patch14-336')
    text_model.to('cuda')
    text_tokenizer = AutoTokenizer.from_pretrained('openai/clip-vit-large-patch14-336')
    text_features = []
    for type in range(ds_info.id_min,ds_info.id_max+1):
        print(type,ds_info.get_caption(type))
        text_inputs = text_tokenizer(ds_info.get_caption(type), padding=True, return_tensors="pt").to('cuda')
        text_feature = text_model(**text_inputs).text_embeds
        # L2 normalize to unit hypersphere for CLIP alignment
        text_feature = F.normalize(text_feature, p=2, dim=-1)
        text_features.append(text_feature.detach().cpu().numpy())
        assert text_feature.shape[-1]==768

    del text_model,text_tokenizer
    return np.concatenate(text_features,axis=0)



def is_rank0():
    """Check if current process is rank 0 in distributed environment"""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True  # Single process environment


def safe_barrier(timeout_minutes=10, context="unknown"):
    """
    DDP-safe barrier with timeout and error handling.

    Prevents indefinite hangs by enforcing a timeout and logging helpful error messages.

    Args:
        timeout_minutes: Maximum time to wait for all ranks (default: 10 minutes)
        context: Description of where this barrier is called (for debugging)

    Raises:
        RuntimeError: If barrier times out or encounters an error
    """
    import sys

    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        try:
            torch.distributed.barrier()
        except Exception as e:
            # Log detailed error message to stderr for debugging
            error_msg = f"[Rank{rank}] Barrier timeout/error in context: {context}\n"
            error_msg += f"[Rank{rank}] Error details: {str(e)}\n"
            error_msg += f"[Rank{rank}] This usually means:\n"
            error_msg += f"  1. Another rank crashed or is still processing\n"
            error_msg += f"  2. Network issues between GPUs\n"
            error_msg += f"  3. Unbalanced workload across ranks\n"
            print(error_msg, file=sys.stderr, flush=True)


def get_wavemind_root() -> str:
    """
    自动检测 WaveMind 项目根目录。

    通过查找 pyproject.toml 哨兵文件确定项目根路径。
    若设置了 WaveMind_ROOT_PATH_ 环境变量，优先使用（向后兼容）。

    Returns:
        项目根目录的绝对路径。

    Raises:
        RuntimeError: 若找不到 pyproject.toml 且环境变量也未设置。
    """
    import os
    from pathlib import Path

    # 向后兼容：环境变量优先
    env_val = os.environ.get('WaveMind_ROOT_PATH_', None)
    if env_val:
        return env_val

    # 从 data/Utils.py 位置向上搜索
    current = Path(__file__).resolve().parent  # data/
    project_root = current.parent  # WaveMind/

    if (project_root / 'pyproject.toml').exists():
        return str(project_root)

    # 搜索更上层（最多10层）
    current = project_root
    for _ in range(10):
        if (current / 'pyproject.toml').exists():
            return str(current)
        parent = current.parent
        if parent == current:
            break
        current = parent

    raise RuntimeError(
        "Could not find WaveMind project root. "
        "Ensure pyproject.toml exists in the project root."
    )
