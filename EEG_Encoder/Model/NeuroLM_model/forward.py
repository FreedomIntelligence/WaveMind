"""
NeuroLM Model Forward Pass Implementation
by Wei-Bang Jiang
https://github.com/935963004/NeuroLM

This file implements the forward pass for the NeuroLM model, including:
- Model loading from checkpoint
- EEG and text processing
- Input/output shape documentation
"""

import torch
import sys
import os
from EEG_Encoder.Model.NeuroLM_model.model_neurolm import NeuroLM
from EEG_Encoder.Model.NeuroLM_model.model import GPTConfig

def load_neurolm(checkpoint_path, vq_checkpoint_path=None):
    """
    Load NeuroLM model from checkpoint with optional VQ tokenizer
    
    Args:
        checkpoint_path: Path to NeuroLM-B.pt checkpoint
        vq_checkpoint_path: Path to VQ.pt checkpoint (optional)
        
    Returns:
        Initialized NeuroLM model
    """
    # Load NeuroLM checkpoint
    checkpoint = torch.load(checkpoint_path,weights_only=False)
    checkpoint_model_args = checkpoint['model_args']
    
    # Create model config
    model_args = dict(
        n_layer=checkpoint_model_args['n_layer'],
        n_head=checkpoint_model_args['n_head'],
        n_embd=checkpoint_model_args['n_embd'],
        block_size=checkpoint_model_args['block_size'],
        bias=checkpoint_model_args['bias'],
        vocab_size=checkpoint_model_args['vocab_size'],
        dropout=checkpoint_model_args.get('dropout', 0.0)
    )
    
    # Initialize model with VQ tokenizer if provided
    gptconf = GPTConfig(**model_args)
    model = NeuroLM(gptconf,
                   tokenizer_ckpt_path=vq_checkpoint_path,
                   init_from='scratch')
    
    # Load NeuroLM state dict
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict,strict=True)
    print("NeuroLM model loaded from checkpoint")
    import torch.nn as nn
    model.GPT2.lm_head_2 = nn.Linear(gptconf.n_embd, 768, bias=False)
    device = next(model.GPT2.lm_head.parameters()).device
    model.GPT2.lm_head_2.to(device)
    model.train()
    model.tokenizer.train()

    return model



def neurolm_forward(model, x_eeg=None, x_text=None, input_chans=None, input_time=None, input_mask=None):
    """
    Forward pass for NeuroLM model
    
    Args:
        model: NeuroLM model instance
        x_eeg: EEG input tensor of shape [B, N1, T]
               B = batch size
               N1 = number of EEG channels
               T = time points
        x_text: Text input tensor of shape [B, N2]
                N2 = text sequence length
        input_chans: Channel indices tensor of shape [B, N1]
        input_time: Time indices tensor of shape [B, N1] 
        input_mask: Attention mask for EEG of shape [B, N1]
        
    Returns:
        logits: Output logits of shape [B, N1+N2, vocab_size]
        loss: Cross-entropy loss (if targets provided)
        accuracy: Prediction accuracy (if targets provided)
    """
    # return model.forward_custom(x_eeg)
    
    # input_chans should generate to be (batch,32)
    # input_chans=torch.arange(32).unsqueeze(0).repeat(x_eeg.size(0),1)
    # Process EEG inputs if provided
    
    # input_chans = torch.arange(N1, device=x_eeg.device, dtype=torch.long).repeat(B, 1)
    # input_time = torch.zeros(B, 1, device=x_eeg.device,dtype=torch.long)
    # input_mask = torch.ones(B, N1)
    
    # if x_eeg is not None:
    #     # Add batch dimension if needed
    #     if x_eeg.dim() == 2:
    #         x_eeg = x_eeg.unsqueeze(0)
            
    #     # Process through tokenizer
    #     input_mask = input_mask.unsqueeze(1).repeat(1, x_eeg.size(1), 1).unsqueeze(1)
    #     x_eeg = model.tokenizer(x_eeg, input_chans, input_time, input_mask, return_all_tokens=True)
    #     x_eeg = model.encode_transform_layer(x_eeg)
    #     x_eeg += model.pos_embed(input_chans)
        
    # # Process text inputs if provided
    # if x_text is not None:
    #     # Add batch dimension if needed
    #     if x_text.dim() == 1:
    #         x_text = x_text.unsqueeze(0)
    
    # # Create attention mask if both inputs provided
    # eeg_text_mask = None
    # if x_eeg is not None and x_text is not None:
    #     eeg_max_time = torch.max(input_time)
    #     eeg_text_mask = torch.ones(
    #         (x_eeg.size(0), 1, x_eeg.size(1)+x_text.size(1), x_eeg.size(1)+x_text.size(1)),
    #         device=x_eeg.device
    #     )
    
    
    # # Forward through GPT model
    # logits, loss, accuracy = model.GPT2(
    #     x_eeg=x_eeg,
    #     x_text=x_text,
    #     eeg_time_idx=input_time,
    #     eeg_text_mask=eeg_text_mask
    # )
    # logits=logits.squeeze(1)
    
    # return logits, loss, accuracy

def generate_text(model, x_eeg, input_chans, input_time, input_mask, max_new_tokens=10):
    """
    Generate text from EEG inputs
    
    Args:
        model: NeuroLM model instance
        x_eeg: EEG input tensor of shape [B, N1, T]
        input_chans: Channel indices tensor of shape [B, N1]
        input_time: Time indices tensor of shape [B, N1]
        input_mask: Attention mask for EEG of shape [B, N1]
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        Generated text tensor of shape [B, max_new_tokens]
    """
    # Initialize text with separator token
    sep_token = model.GPT2.config.vocab_size - 1
    x_text = torch.full((x_eeg.size(0), 1), sep_token, device=x_eeg.device)
    
    # Generate text
    return model.generate(
        x_eeg=x_eeg,
        x_text=x_text,
        input_chans=input_chans,
        input_time=input_time,
        input_mask=input_mask,
        eeg_text_mask=None,
        max_new_tokens=max_new_tokens
    )

# Example usage
if __name__ == '__main__':
    # Load model with both NeuroLM and VQ checkpoints
    checkpoint_path = f'{os.environ["WaveMind_ROOT_PATH_"]}]/EEG_Encoder/Resource/Checkpoint/ALL/NeuroLM-B.pt'
    vq_path = f'{os.environ["WaveMind_ROOT_PATH_"]}]/EEG_Encoder/Resource/Checkpoint/ALL/VQ.pt'
    model = load_neurolm(checkpoint_path, vq_checkpoint_path=vq_path)
    # model.eval()
    

    B, N1, T = 52, 32, 512
    x_eeg = torch.randn(B, N1, T)


    # print(model.forward(x_eeg))
    