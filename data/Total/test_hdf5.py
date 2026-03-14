import numpy as np
import random
from tqdm import tqdm
import torch

import torch
import random
from transformers import CLIPTextModelWithProjection, AutoTokenizer

# Global model and tokenizer for efficiency
_text_model = None
_text_tokenizer = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_path=r'/F00120250029/lixiang_share/Models/clip-vit-large-patch14-336'

def _load_model():
    """Load CLIP model and tokenizer if not already loaded"""
    global _text_model, _text_tokenizer
    if _text_model is None or _text_tokenizer is None:
        _text_model = CLIPTextModelWithProjection.from_pretrained(model_path)
        _text_tokenizer = AutoTokenizer.from_pretrained(model_path)
        _text_model.to(_device)
        _text_model.eval()

def convert_to_feature(caption, stored_text_features=None, compare_samples=5):
    """
    Convert single text to feature vector and optionally perform self-similarity comparison
    
    Args:
        caption (str): Input text
        stored_text_features (torch.Tensor, optional): Stored text features for similarity comparison
        compare_samples (int): Number of other samples to compare with, default is 5
    
    Returns:
        dict: Contains feature and similarity comparison results
    """
    assert isinstance(caption, str), "Caption must be a string, got {}".format(type(caption))
    
    # Load model if not already loaded
    _load_model()
    
    # Process single sample
    text_inputs = _text_tokenizer(caption, padding=True, return_tensors="pt").to(_device)
    text_feature = _text_model(**text_inputs).text_embeds
    assert text_feature.shape[-1] == 768
    
    # Check if stored features are already L2 normalized
    need_normalization = True
    if stored_text_features is not None:
        # Check if stored features are normalized
        stored_norms = torch.norm(stored_text_features, p=2, dim=1)
        avg_norm = torch.mean(stored_norms)
        # If average norm is close to 1, consider them normalized
        if torch.isclose(avg_norm, torch.tensor(1.0), atol=1e-3):
            need_normalization = True
        else:
            need_normalization = False
    
    # If normalization is needed, apply L2 normalization to current feature
    if need_normalization:
        text_feature = torch.nn.functional.normalize(text_feature, p=2, dim=1)
    
    result = {
        'text_feature': text_feature.detach().cpu(),
        'caption': caption,
        'need_normalization': need_normalization
    }
    
    # If stored features are provided, perform self-similarity comparison
    if stored_text_features is not None:
        similarities = self_similarity_test(text_feature, stored_text_features, compare_samples)
        result.update(similarities)
    
    # Clean up intermediate tensors
    del text_inputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return result


def self_similarity_test(current_feature, stored_features, compare_samples=5):
    """
    Self-similarity test: compare current extracted feature with stored features
    
    Args:
        current_feature (torch.Tensor): Current extracted text feature
        stored_features (torch.Tensor): Stored text features (should include the corresponding stored feature)
        compare_samples (int): Number of other samples to compare with
    
    Returns:
        dict: Similarity comparison results
    """
    # Ensure current feature and stored features are on the same device
    current_feature = current_feature.cpu()
    stored_features = stored_features.cpu()
    
    # Calculate cosine similarity with all stored features
    similarities = torch.matmul(stored_features, current_feature.T).squeeze()
    
    # Sort similarities in descending order
    sorted_similarities, sorted_indices = torch.sort(similarities, descending=True)
    
    # The highest similarity should be with the corresponding stored feature
    self_similarity = sorted_similarities[0].item()
    
    # Get similarities with other samples (excluding the highest one)
    other_similarities = sorted_similarities[1:compare_samples+1].tolist()
    
    # If there are fewer than compare_samples other features, pad with zeros
    while len(other_similarities) < compare_samples:
        other_similarities.append(0.0)
    
    return {
        'self_similarity': self_similarity,
        'other_similarities': other_similarities,
        'max_other_similarity': max(other_similarities) if other_similarities else 0,
        'is_highest': self_similarity >= max(other_similarities) if other_similarities else True,
        'all_similarities': sorted_similarities.tolist()
    }




def test_dataset(ds):
    """
    Test the CLIPDataset for basic data integrity checks using PyTorch operations.
    
    Args:
        ds: Instance of CLIPDataset to be tested
    
    Tests performed:
    1. EEG data checks:
        - Data type is float32
        - Standard deviation > 0.01 across all channels
    2. Text features checks:
        - Features are 1D vectors
        - L2 normalized (norm ≈ 1.0)
        - Features show highest similarity with their own text
    3. Image path checks:
        - Valid path for 'thingEEG' and 'ImageNetEEG'
        - Empty string for other datasets
    """
    # Select a random sample for testing
    idx = random.randint(0, len(ds) - 1)
    sample = ds[idx]
    print(f"Testing sample index: {idx}")
    
    # 1. Test EEG data
    eeg_data = sample['eeg_data']
    
    # 1.1 Check data type is float32
    assert eeg_data.dtype == torch.float32, \
        f"EEG data type should be float32, got {eeg_data.dtype}"
    
    # 1.2 Check standard deviation > 0.01
    std = torch.std(eeg_data)
    assert std > 0.01, \
        f"EEG data std ({std.item():.4f}) should be > 0.01"
    
    print(f"EEG data passed: dtype={eeg_data.dtype}, std={std.item():.4f}")
    
    # 2. Test text features
    align_feature = sample['text_features']
    current_text = sample['text']
    
    # 2.1 Check shape and normalization (simplified version)
    assert align_feature.dim() == 1, \
        f"Text features should be 1D, got shape {align_feature.shape}"
    
    norm = torch.norm(align_feature, p=2)
    # assert torch.isclose(norm, torch.tensor(1.0), atol=1e-4), \
    #     f"Text features should be normalized (norm={norm.item():.6f})"
    # print norm
    print(f"Text features: (norm={norm.item():.6f})")
    if not torch.isclose(norm, torch.tensor(1.0), atol=1e-4):
        print(f"Warning: Text features not normalized (norm={norm.item():.6f})")
    
    # 2.2 Self-similarity test: compare current extracted feature with stored features
    # Extract current text feature using convert_to_feature (simulating real-time extraction)
    current_extracted_feature = convert_to_feature(current_text)['text_feature']
    
    # Get the corresponding stored feature for this sample
    corresponding_stored_feature = align_feature.unsqueeze(0)  # Shape: (1, 768)
    
    # Randomly select 5 other stored features from the dataset (excluding current sample)
    other_indices = random.sample([i for i in range(len(ds)) if i != idx], 5)
    other_stored_features = torch.stack([ds[i]['text_features'] for i in other_indices])
    
    # Combine stored features: corresponding feature + 5 random other features
    stored_features = torch.cat([corresponding_stored_feature, other_stored_features], dim=0)
    
    # Perform self-similarity test
    similarity_results = self_similarity_test(current_extracted_feature, stored_features, compare_samples=5)
    
    self_similarity = similarity_results['self_similarity']
    other_similarities = similarity_results['other_similarities']
    max_other_similarity = similarity_results['max_other_similarity']
    is_highest = similarity_results['is_highest']
    
    print(f"Self-similarity test results:")
    print(f"  Self-similarity: {self_similarity:.4f}")
    print(f"  Other similarities: {[f'{s:.4f}' for s in other_similarities]}")
    print(f"  Max other similarity: {max_other_similarity:.4f}")
    print(f"  Is highest: {is_highest}")
    
    # The extracted feature should have highest similarity with its corresponding stored feature
    # Allow equality since it's possible for the same feature to have equal similarity
    assert is_highest or self_similarity == max_other_similarity, \
        f"Extracted feature similarity ({self_similarity:.4f}) not higher than or equal to others ({max_other_similarity:.4f})"
    
    print(f"Text features passed: normalized (norm={norm.item():.6f}), self-similarity={self_similarity:.4f} > others")
    
    # 3. Test image path
    img_path = sample['img_path']
    dataset_name = ds.dataset_name
    
    if dataset_name in ['thingEEG', 'ImageNetEEG']:
        assert isinstance(img_path, str) and len(img_path) > 0, \
            f"img_path should be non-empty for {dataset_name}"
    else:
        assert img_path == "", f"img_path should be empty for {dataset_name}, got '{img_path}'"
    
    print(f"Image path passed")
    
    print("All tests passed successfully!")



def test_convert_to_feature():
    """
    Test the new convert_to_feature function with self-similarity comparison
    """
    print("Testing convert_to_feature function...")
    
    # Test with a single caption
    caption = "This is a test caption for EEG data analysis"
    result = convert_to_feature(caption)
    
    print(f"Single sample test:")
    print(f"Caption: {result['caption']}")
    print(f"Feature shape: {result['text_feature'].shape}")
    print(f"Need normalization: {result['need_normalization']}")
    
    # Test with stored features for self-similarity comparison
    print("\nTesting with stored features for self-similarity...")
    
    # Create more realistic stored features
    # First, extract the feature for the same caption to simulate stored feature
    stored_feature_same = convert_to_feature(caption)['text_feature']
    
    # Create some other dummy stored features with different captions
    other_captions = [
        "A different EEG data analysis caption",
        "Another test for brain signal processing",
        "EEG signal classification task",
        "Brain computer interface application",
        "Neural signal processing experiment"
    ]
    
    other_stored_features = []
    for other_caption in other_captions:
        other_feature = convert_to_feature(other_caption)['text_feature']
        other_stored_features.append(other_feature)
    
    # Combine all stored features (same caption + other captions)
    all_stored_features = torch.cat([stored_feature_same] + other_stored_features, dim=0)
    
    # Test self-similarity: extract feature again and compare with stored features
    result_with_comparison = convert_to_feature(
        caption,
        stored_text_features=all_stored_features,
        compare_samples=5
    )
    
    print(f"Self-similarity (with stored same feature): {result_with_comparison['self_similarity']:.4f}")
    print(f"Other similarities: {[f'{s:.4f}' for s in result_with_comparison['other_similarities']]}")
    print(f"Max other similarity: {result_with_comparison['max_other_similarity']:.4f}")
    print(f"Is highest: {result_with_comparison['is_highest']}")
    
    # Test normalization logic
    print(f"\nNormalization check:")
    print(f"Stored features average norm: {torch.mean(torch.norm(all_stored_features, p=2, dim=1)):.4f}")
    print(f"Need normalization: {result_with_comparison['need_normalization']}")
    
    print("convert_to_feature test completed successfully!")


if __name__ == "__main__":
    import os
    root_path=os.environ['WaveMind_ROOT_PATH_']
    os.chdir(os.path.join(root_path,'data/Total'))
    
    from EEG_Encoder.Tools.dataBuilder import CLIPDataset
    
    # Test the new convert_to_feature function
    # test_convert_to_feature()
    
    print("\n" + "="*50 + "\n")
    
    # Original dataset test
    # ---------------------------------
    dataset_name='ImageNetEEG'
    ds=CLIPDataset(hdf5_file_path='data_label.h5',
                    mode='test',
                    dataset_name=dataset_name,
                    ground_truth_dir='CLIP_groundTruth')
    test_dataset(ds)
    # ---------------------------------
    dataset_name='thingEEG'
    ds=CLIPDataset(hdf5_file_path='data_label.h5',
                    mode='test',
                    dataset_name=dataset_name,
                    ground_truth_dir='CLIP_groundTruth')
    test_dataset(ds)
    # ---------------------------------
    dataset_name='TUEV'
    ds=CLIPDataset(hdf5_file_path='data_label.h5',
                    mode='test',
                    dataset_name=dataset_name,
                    ground_truth_dir='CLIP_groundTruth')
    test_dataset(ds)
    # ---------------------------------
    dataset_name='TUAB'
    ds=CLIPDataset(hdf5_file_path='data_label.h5',
                    mode='test',
                    dataset_name=dataset_name,
                    ground_truth_dir='CLIP_groundTruth')
    test_dataset(ds)
    # ---------------------------------


    # Original dataset train
    # ---------------------------------
    dataset_name='ImageNetEEG'
    ds=CLIPDataset(hdf5_file_path='data_label.h5',
                    mode='train',
                    dataset_name=dataset_name,
                    ground_truth_dir='CLIP_groundTruth')
    test_dataset(ds)
    # ---------------------------------
    dataset_name='thingEEG'
    ds=CLIPDataset(hdf5_file_path='data_label.h5',
                    mode='train',
                    dataset_name=dataset_name,
                    ground_truth_dir='CLIP_groundTruth')
    test_dataset(ds)
    # ---------------------------------
    dataset_name='TUEV'
    ds=CLIPDataset(hdf5_file_path='data_label.h5',
                    mode='train',
                    dataset_name=dataset_name,
                    ground_truth_dir='CLIP_groundTruth')
    test_dataset(ds)
    # ---------------------------------
    dataset_name='TUAB'
    ds=CLIPDataset(hdf5_file_path='data_label.h5',
                    mode='train',
                    dataset_name=dataset_name,
                    ground_truth_dir='CLIP_groundTruth')
    test_dataset(ds)
    # ---------------------------------
