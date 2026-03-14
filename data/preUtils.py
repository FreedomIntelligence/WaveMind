import os
import warnings

# import autoreject
import h5py
import mne
# import mne_icalabel
import numpy as np
import pandas as pd
import scipy.io as sio
# from autoreject import get_rejection_threshold
from filelock import FileLock, Timeout
from mne import create_info
from mne.io import RawArray
from mne.preprocessing import ICA
import time

import os
os.environ['MNE_N_JOBS'] = '16'
def auto_scale_to_uv(raw):
    """
    Automatically detect and scale EEG data to microvolts (μV) using mean and standard deviation (std).

    Parameters:
        raw (mne.io.Raw): MNE-Python Raw object containing the EEG data.

    Returns:
        raw (mne.io.Raw): Scaled Raw object with data in μV.
    """
    # Get the data
    data = raw.get_data()

    # Calculate the mean and standard deviation
    std_val = np.std(data)

    print("Standard deviation:", std_val)


    # Determine the unit of the data based on std
    if std_val < 8e-3:
        # Data is likely in volts (V)
        raw._data *= 1e6  # Convert to microvolts
        warnings.warn("Data was in volts (V). Converted to microvolts (μV).")
    elif std_val < 1:
        # Data is likely in millivolts (mV)
        raw._data *= 1000  # Convert to microvolts
        warnings.warn("Data was in millivolts (mV). Converted to microvolts (μV).")
    elif std_val < 1000:
        # Data is likely already in microvolts (μV)
        print("Data is already in microvolts (μV). No scaling applied.")
    else:
        # Data is in an unexpected range; consider manual inspection
        warnings.warn("Data is in an unexpected range. Consider manual inspection.")

    return raw




def eeg_filter_notch(raw):
    raw=raw.load_data()
    raw=raw.notch_filter(50,phase='zero')
    raw=raw.notch_filter(60,phase='zero')
    return raw

def pick_eeg_channels(raw):
    eeg_picks = mne.pick_types(raw.info, eeg=True, meg=False, stim=False)
    return raw.pick(eeg_picks)

def ica_removal(raw):
    # Perform ICA decomposition
    # ica = ICA(n_components=0.9999, max_iter='auto', method='infomax', fit_params=dict(extended=True))
    ica = ICA(n_components=0.9999, max_iter='auto')
    try:
        ica.fit(raw)
    except:
        warnings.warn("ICA failed to fit the data. Skipping ALL raw", category=UserWarning)
        return None
    # Use mne_icalabel to label ICs
    ic_labels = mne_icalabel.label_components(raw, ica, method="iclabel")
    labels = ic_labels["labels"]

    # Initialize the set to store indices of components to exclude
    exclude_idx = set()

    # Check if any channels with 'eog' in their name exist
    if any('eog' in ch_name.lower() for ch_name in raw.info['ch_names']):
        print("EOG channels found in the dataset.")
        eog_inds, scores_eog = ica.find_bads_eog(raw)  # Find EOG-related ICs
        print("EOG-related ICs:", eog_inds)
        exclude_idx.update(eog_inds)  # Add EOG-related IC indices to exclude list

    # Further exclude components based on IC label emb_prediction from mne_icalabel
    for i in range(len(labels)):
        if labels[i] == 'brain' and ic_labels['y_pred_proba'][i] < 0.6:
            exclude_idx.add(i)
        elif labels[i] in ['eye blink', 'heart beat'] and ic_labels['y_pred_proba'][i] > 0.65:
            exclude_idx.add(i)
        elif labels[i] == 'other' and ic_labels['y_pred_proba'][i] > 0.85:
            exclude_idx.add(i)
        elif labels[i] == 'line noise':
            exclude_idx.add(i)

    # Convert the set of indices to a list and set the ICA exclude attribute
    ica.exclude = list(exclude_idx)

    # Apply ICA and remove the identified ICs
    ica.apply(raw)

    return raw

def create_epoches(raw,duration=1):
    events = mne.make_fixed_length_events(raw, start=0, stop=None, duration=duration)
    epochs = mne.Epochs(raw, events, tmin=0, tmax=2, baseline=(None,None), preload=False)
    return epochs
# def fix_epochs(epochs):
#     epochs.load_data()
#     ar = autoreject.AutoReject(n_interpolate=[4,16],
#                                n_jobs=64, verbose=False,cv=2)
#     ar.fit(epochs)
#     epochs_ar, reject_log = ar.transform(epochs, return_log=True)
#     if np.sum(reject_log.bad_epochs)/len(epochs)>0.5:
#         warnings.warn('Too many epochs rejected')
#     return epochs_ar,np.sum(reject_log.bad_epochs),len(epochs)
def fix_epochs2(epochs,dont_reject=False):
    if epochs.get_data().shape[0]<20:
        return epochs,0,len(epochs)
    reject = get_rejection_threshold(epochs, decim=1,cv=3)
    origin_samples=len(epochs)
    if dont_reject:
        return epochs,0,origin_samples
    epochs_ar=epochs.drop_bad(reject=reject)
    return epochs_ar,origin_samples-len(epochs_ar),origin_samples




def process_Temple_channel(raw):
    def remove_duplicate_values(d):
        reverse_map = {}
        unique_map = {}
        for key, value in d.items():
            if value not in reverse_map:
                reverse_map[value] = key
                unique_map[key] = value
        return unique_map

    def modify_channel_names(ch_names):
        new_ch_names = {}
        keep_channels=set()
        for name in ch_names:
            name_ori=name
            name=name.replace("EEG ", "").split("-")[0].strip()
            if name.isdigit():
                new_ch_names[name_ori]=name
                continue
            if name.startswith('DC'):
                new_ch_names[name_ori]=name
                continue
            if len(name) > 3:
                new_ch_names[name_ori]=name
                continue
            if 'EKG' in name.upper():
                new_ch_names[name_ori]=name
                continue
            if  name.upper() in ['ROC','LOC','EMG','IBI','PG1','PG2',"T1","T2","SP1","SP2","RLC",'LUC','X1','C3P','C4P']:
                new_ch_names[name_ori]=name
                continue
            new_ch_names[name_ori]=name[0].upper() + name[1:].lower()
            keep_channels.add(new_ch_names[name_ori])

        return remove_duplicate_values(new_ch_names),list(keep_channels)

    original_ch_names = raw.info['ch_names']
    new_ch_names,keep_channels = modify_channel_names(original_ch_names)
    raw.rename_channels(new_ch_names)
    raw.pick_channels(keep_channels)
    if len(raw.info['ch_names'])<15:
        warnings.warn("The number of channels is {} which is less than 15, please check the channel names".format(len(raw.info['ch_names'])), category=UserWarning)
    raw.set_montage('standard_1020')
    return raw

def z_score_normalize(data):
    """
    Apply Z-score normalization to the data for each channel.

    Parameters:
    - data: numpy array of shape (batch, channel, time)

    Returns:
    - normalized_data: numpy array of shape (batch, channel, time)
    """
    assert len(data.shape) == 3
    means = np.mean(data, axis=2, keepdims=True)
    stds = np.std(data, axis=2, keepdims=True)
    stds[stds == 0] = 1


    normalized_data = (data - means) / stds
    # print("After normalization general std:",np.mean(np.std(normalized_data, axis=2)))

    return normalized_data

def eeg_filter_all(raw,highpass,lowpass,notch=True):
    if highpass and lowpass:
        assert highpass < lowpass, "High-pass cutoff frequency must be lower than low-pass cutoff frequency."
    raw = raw.filter(highpass,lowpass,n_jobs='cuda',phase='zero')
    if notch:
        raw = eeg_filter_notch(raw)
    return raw

def processRaw(raw,ica_func,l_freq=0.5,h_freq=100):
    def filterInterpolateCh(raw):
        raw = pick_eeg_channels(raw)
        target_channels = [
            'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
            'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8',
            'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8',
            'POz', 'O1', 'Oz', 'O2', 'AFz', 'CPz', 'FCz'
        ]
        raw_ch_names_lower = [ch.lower() for ch in raw.ch_names]
        missing_channels = [ch for ch in target_channels if ch.lower() not in raw_ch_names_lower]
        # print("Total channels numbers:{} existing channels number:{} missing channels number:{}".format(len(raw.ch_names),len(existing_channels),len(missing_channels)))
        # raw=raw.pick_channels(existing_channels)
        if missing_channels:
            new_raw_info = create_info(ch_names=missing_channels, sfreq=raw.info['sfreq'], ch_types='eeg')
            new_raw = mne.io.RawArray(np.zeros((len(missing_channels), raw.n_times)), new_raw_info)
            new_raw = eeg_filter_all(new_raw,lowpass=raw.info['lowpass'],highpass=raw.info['highpass'])
            raw = raw.add_channels([new_raw])
            raw = raw.set_montage('standard_1020')
            raw.info['bads'].extend(missing_channels)
            assert len(raw.info['bads']) != 0
            assert len(raw.info['bads']) == len(missing_channels)
            raw = raw.interpolate_bads()
        raw = raw.pick_channels(target_channels)
        assert len(raw.ch_names) == len(target_channels)
        raw = raw.reorder_channels(target_channels)
        return raw
    if raw.info['bads']:
        raw = raw.interpolate_bads()
    raw = eeg_filter_all(raw,l_freq,h_freq,notch=True)
    if ica_func:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = ica_func(raw)
    if raw != None:
        raw = filterInterpolateCh(raw)
    return raw

def preProcessOneFile(loadFunc,eeg_file_name,ica_func=ica_removal,epoch_interval=1,dont_reject=False):
    raw=loadFunc(eeg_file_name)
    if raw != None:
        EEG_length=raw.n_times / raw.info['sfreq']
    if raw==None:
        print("Rejecting all raw")
        text = {
            "time": time.strftime("%m-%d %H:%M:%S", time.localtime()),
            "remain": 0,
            "total": 0,
            "drop": 0,
            "drop_rate": 100,
            "file": os.path.basename(eeg_file_name),
            "Raw duration": 0
        }
        df = pd.DataFrame(text, index=[0])
        return df,None
    raw=processRaw(raw,ica_func)
    if raw==None:
        print("Rejecting all raw")
        text = {
            "time": time.strftime("%m-%d %H:%M:%S", time.localtime()),
            "remain": 0,
            "total": 0,
            "drop": 0,
            "drop_rate": 100,
            "file": os.path.basename(eeg_file_name),
            "Raw duration": 0
        }
        df = pd.DataFrame(text, index=[0])
        return df,None
    epoches=create_epoches(raw,duration=epoch_interval).load_data()
    print("Epoch shape before:{}".format(epoches.get_data().shape))
    del raw
    epoches,drop_num,total_num=fix_epochs2(epoches,dont_reject=dont_reject)
    print("Epoch shape after:{}".format(epoches.get_data().shape))
    text = {
        "time": time.strftime("%m-%d %H:%M:%S", time.localtime()),
        "remain": len(epoches),
        "total": total_num,
        "drop": drop_num,
        "drop_rate": drop_num / total_num,
        "file": os.path.basename(eeg_file_name),
        "Raw duration":EEG_length
    }
    df=pd.DataFrame(text,index=[0])
    np_data=z_score_normalize(epoches.get_data())
    return df,np_data


def write2FileNPZ(df, np_data):
    if np_data is not None and np_data.size > 0:
        np_data=np_data.astype('float32')
        sample_number=np_data.shape[0]
        if not os.path.exists('data.npz'):
            np.savez_compressed('data.npz', np_data)
        else:
            data = np.load('data.npz')['arr_0']
            data = np.concatenate([data, np_data], axis=0)
            np.savez_compressed('data.npz', data)
            sample_number = data.shape[0]
        df['samples_number'] = sample_number
    writeFileinDF(df)

def process_flat_signal_new(raw):
    prev_raw_time=raw.n_times
    def process_flat_signal_epoches(epoches):
        reject_indices = []
        data = epoches.get_data()
        count=0
        for idx, epoch_data in enumerate(data):
            segment1 = np.std(epoch_data[5, :15])
            segment2 = np.std(epoch_data[0, -15:-5])
            if segment1 < 1e-8 or segment2 < 1e-8:
                reject_indices.append(idx)
            if idx==100:
                if count>90:
                    return None
                break
        # print(reject_indices)
        return epoches.drop(reject_indices)
    try:
        epoches=create_epoches(raw,duration=2)
        epoches=process_flat_signal_epoches(epoches)
    except:
        return None,100
    if epoches.get_data().shape[0]<=1 or epoches==None:
        return None,100
    data = epoches.get_data()
    n_epochs, n_channels, n_times = data.shape
    data_2d = data.transpose(1, 0, 2).reshape(n_channels, -1)
    info = epoches.info.copy()
    raw = mne.io.RawArray(data_2d, info)
    new_raw_time=raw.n_times
    drop_rate=(prev_raw_time-new_raw_time)/prev_raw_time
    return raw, drop_rate


import errno

def write2FileinHDF5(np_data, dataset_name):
    if np_data is None or np_data.size == 0:
        print("No data to write.")
        return 0

    if len(np_data.shape) != 3 or np_data.shape[1:] != (32, 513):
        raise ValueError(f"Input data must have shape (n, 32, 513), but got {np_data.shape}.")

    np_data = np_data.astype('float16')
    file_path = 'data.h5'
    chunks = (10, 32, 513)
    compression_opts = 9

    def try_open_hdf5(mode='a', max_retries=20, retry_delay=10):
        retries = 0
        while retries < max_retries:
            try:
                return h5py.File(file_path, mode)
            except OSError as e:
                if e.errno in [errno.EACCES, errno.EBUSY]:
                    print(f"File {file_path} is locked. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retries += 1
                else:
                    print(f"Failed to open file {file_path}: {e}")
                    raise
        raise Exception(f"Failed to open file {file_path} after {max_retries} retries.")

    with try_open_hdf5('a' if os.path.exists(file_path) else 'w') as hf:
        if dataset_name not in hf:
            hf.create_dataset(
                dataset_name,
                data=np_data,
                chunks=chunks,
                compression="gzip",
                compression_opts=compression_opts,
                maxshape=(None, 32, 513)
            )
        else:
            dset = hf[dataset_name]
            new_shape = (dset.shape[0] + np_data.shape[0], 32, 513)
            dset.resize(new_shape)
            dset[-np_data.shape[0]:] = np_data

        sample_number = hf[dataset_name].shape[0]
    return sample_number

def writeFileinDF(df):
    log_file_path = 'log.csv'
    try:
        with FileLock(log_file_path + '.lock', timeout=20):
            if not os.path.exists(log_file_path):
                df.to_csv(log_file_path, index=True)
            else:
                log_df = pd.read_csv(log_file_path, index_col=0)
                last_index = int(log_df.index[-1] if not log_df.empty else -1)
                df.index = range(last_index + 1, last_index + 1 + len(df))
                result_df = pd.concat([log_df, df])
                result_df.to_csv(log_file_path, index=True)
    except Timeout:
        print("Could not acquire lock on the log file. Please try again later.")

def write2FileHDF5(df, np_data,dataset_name):
    df['sample_number']=write2FileinHDF5(np_data,dataset_name)
    writeFileinDF(df)


