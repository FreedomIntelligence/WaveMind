import os

import torch
from torch.nn import functional as F

import math
import numpy as np
import torch.nn as nn

from einops.layers.torch import Rearrange, Reduce

import sys



sys.path.append('/mnt/nvme2n1/zengziyi/Research')
sys.path.append('/223040001/ziyi/Research')


from EEG_Encoder.Model.CommonBlock import Config, iTransformer, Enc_eeg, Proj_eeg, iTransformer_Modify # type: ignore
#from EEG_Encoder.Model.CommonBlock import Config, iTransformer, Enc_eeg, Proj_eeg, iTransformer_Modify
# from ..loss import ClipLoss

from braindecode.models import *
import argparse


device= torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder_CLIP_training(nn.Module):
    def __init__(self, encoder, config=Config()):
        super(Encoder_CLIP_training, self).__init__()
        self.encoder=encoder
        self.text_logit_scale = nn.Parameter(torch.ones([]) * np.log(1/0.07))
        self.image_logit_scale = nn.Parameter(torch.ones([]) * np.log(1/0.07))
    def forward(self, x):
        res=self.encoder(x)

        return res



# -------------------

# class MOERouter(nn.Module):
#     def __init__(self, encoder1, encoder2, encoder3):
#         super().__init__()
#         self.experts = nn.ModuleList([encoder1, encoder2, encoder3])
#
#         self.router = nn.Sequential(
#             nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
#             nn.GELU(),
#             nn.AdaptiveAvgPool1d(16),
#             nn.Flatten(),
#             nn.SyncBatchNorm(64 * 16),
#             nn.Linear(64 * 16, 128),
#             nn.GELU(),
#             nn.Dropout(0.5),
#             nn.Linear(128, 3),
#             nn.Softmax(dim=1)
#         )
#
#     def forward(self, x):
#         routing_weights = self.router(x)
#
#         expert_outputs = torch.stack([e(x)['pooler_output'] for e in self.experts], dim=1)
#
#         weighted_output = torch.sum(routing_weights.unsqueeze(-1) * expert_outputs, dim=1)
#
#         return {'pooler_output':weighted_output,"last_hidden_state":None}


class MOERouter(nn.Module):
    def __init__(self, *encoders, temperature=1, hard=False, noise_scale=0.1):
        super().__init__()
        assert len(encoders) >= 2, "At least two experts required"
        self.experts = nn.ModuleList(encoders)
        self.temperature = temperature  # Gumbel softmax temperature
        self.hard = hard  # Use hard selection (enable for inference)
        self.noise_scale = noise_scale  # Logit noise for exploration

        # Routing network (outputs unnormalized logits)
        self.router = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.SyncBatchNorm(64 * 16),  # DDP-friendly batch norm
            nn.Linear(64 * 16, 128),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(128, len(encoders))  # Raw logits (no softmax)
        )

    def forward(self, x):
        # Calculate routing logits
        logits = self.router(x)  # (B, num_experts)

        # Add exploration noise during training
        if self.training:
            logits = logits + torch.randn_like(logits) * self.noise_scale

        # Gumbel-softmax sampling
        if not self.hard:
            # Differentiable approximation during training
            routing_weights = F.gumbel_softmax(
                logits, tau=self.temperature, hard=self.hard, dim=1
            )  # (B, num_experts)
        else:
            # Hard top-2 selection for inference
            _, topk_indices = torch.topk(logits, k=2, dim=1)
            routing_weights = torch.zeros_like(logits).scatter(
                dim=1, index=topk_indices, value=1.0
            )

        # Always select top-2 experts with normalized weights
        topk_values, topk_indices = torch.topk(routing_weights, k=2, dim=1)
        normalized_weights = topk_values / (topk_values.sum(dim=1, keepdim=True) + 1e-8)

        # --- Dynamic expert computation ---
        B = x.size(0)
        expert_outputs = torch.zeros(B, 2, 768, device=x.device)

        # Expand batch dimension for parallel processing
        x_expanded = x.unsqueeze(1).expand(-1, 2, -1, -1).reshape(B * 2, *x.shape[1:])
        expert_ids = topk_indices.view(-1)  # (B*2,)

        # Process unique experts in parallel
        unique_expert_ids = torch.unique(expert_ids)
        for expert_id in unique_expert_ids:
            mask = expert_ids == expert_id
            if not mask.any():
                continue

            # Compute expert outputs in batch
            expert_input = x_expanded[mask]
            expert_out = self.experts[expert_id](expert_input)['pooler_output']

            # DDP-safe type conversion and assignment
            expert_outputs.view(B * 2, -1)[mask] = expert_out.to(
                dtype=expert_outputs.dtype,
                device=expert_outputs.device
            )

        # Weighted combination of expert outputs
        weighted_output = (expert_outputs * normalized_weights.unsqueeze(-1)).sum(dim=1)
        return {'pooler_output': weighted_output}

    def extra_repr(self):
        return f"Gumbel: temp={self.temperature}, hard={self.hard}, noise={self.noise_scale}"






# -------------------

class MOERouter_Optimized(nn.Module):
    def __init__(self, experts, hidden_dim=32, capacity_factor=1.5,
                 expert_dropout=0.3, balance_alpha=0.005, k_threshold=0.8,regularization_alpha=1e-6):
        """Mixture of Experts with dynamic routing and capacity constraints

        Args:
            experts: List of expert modules
            hidden_dim: Feature dimension
            capacity_factor: Capacity multiplier for expert load balancing
            expert_dropout: Dropout rate for router
            balance_alpha: Weight for load balancing loss
            k_threshold: Cumulative probability threshold for dynamic expert selection
        """
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.num_experts = len(experts)
        self.capacity_factor = capacity_factor
        self.balance_alpha = balance_alpha
        self.regularization_alpha = regularization_alpha

        self.k_threshold = k_threshold

        # Routing network
        self.router = nn.Sequential(
            nn.Conv1d(hidden_dim, 64, 3, padding=1),  # Process temporal features
            nn.GELU(),                                # Nonlinear activation
            nn.AdaptiveAvgPool1d(16),                 # Temporal dimension pooling
            nn.Flatten(),                             # Prepare for linear layers
            nn.SyncBatchNorm(64 * 16),                  # Batch normalization
            nn.Linear(64 * 16, 128),                  # Feature projection
            nn.GELU(),                                # Nonlinear activation
            nn.Dropout(expert_dropout),               # Regularization
            nn.Linear(128, self.num_experts),         # Expert logits
            nn.Softmax(dim=1)                         # Probability distribution
        )
        self.expert_dropout = nn.Dropout(expert_dropout)
        self.register_buffer('expert_counts', torch.zeros(self.num_experts))

    def calculate_dynamic_k(self, weights):
        """Dynamically determine number of experts per sample

        Args:
            weights: Routing probabilities [B, E]

        Returns:
            k_indices: Number of experts per sample [B]
        """
        sorted_weights, _ = torch.sort(weights, dim=1, descending=True)
        cumulative = torch.cumsum(sorted_weights, dim=1)

        # Find first position where cumulative probability exceeds threshold
        threshold_mask = (cumulative > self.k_threshold).float()
        k_indices = torch.argmax(threshold_mask, dim=1) + 1

        # Handle samples where sum never exceeds threshold
        k_indices[torch.all(cumulative <= self.k_threshold, dim=1)] = self.num_experts

        return torch.clamp(k_indices, min=1, max=self.num_experts)

    def enforce_capacity(self, topk_indices, topk_weights, expert_capacity):

        modified_indices = topk_indices.clone()  # 创建副本
        batch_size, max_k = modified_indices.shape


        expert_usage = torch.zeros(self.num_experts,
                                   dtype=torch.int64,
                                   device=modified_indices.device)
        expert_usage.scatter_add_(0, modified_indices.flatten(),
                                  torch.ones_like(modified_indices.flatten()))


        over_capacity = (expert_usage > expert_capacity).nonzero().squeeze(1)

        for expert in over_capacity:
            mask = (modified_indices == expert)
            sample_indices, k_positions = torch.where(mask)
            weights = topk_weights[mask]

            if len(weights) > expert_capacity:
                sorted_idx = torch.argsort(weights, descending=True)[:expert_capacity]
                valid_mask = torch.zeros_like(sample_indices, dtype=torch.bool)
                valid_mask[sorted_idx] = True
                modified_indices[sample_indices[~valid_mask], k_positions[~valid_mask]] = -1

        return modified_indices

    def forward(self, x):
        """MOE forward pass with dynamic expert selection

        Args:
            x: Input tensor [B, C, L]

        Returns:
            Dictionary containing:
            - pooler_output: Weighted expert outputs
            - balance_loss: Load balancing regularization
            - expert_usage: Routing probability statistics
            - expert_counts: Actual expert selection counts
        """
        assert x.dim() == 3, "Input must be 3D (batch, channels, length)"
        batch_size = x.size(0)
        if torch.distributed.is_initialized():
            total_bs = torch.tensor([batch_size], device=x.device)
            torch.distributed.all_reduce(total_bs, op=torch.distributed.ReduceOp.SUM)
            total_bs = total_bs.item()
        else:
            total_bs = batch_size

        expert_capacity = max(1, int(batch_size * self.capacity_factor / self.num_experts))

        # 1. Compute routing probabilities
        routing_weights = self.router(x)  # [B, E]
        masked_weights = self.expert_dropout(routing_weights)

        # 2. Dynamic expert selection
        k_per_sample = self.calculate_dynamic_k(masked_weights)
        max_k = k_per_sample.max().item()

        # 3. Select top-k experts with padding
        topk_weights, topk_indices = torch.topk(masked_weights, max_k, dim=1)

        # 4. Enforce capacity constraints
        topk_indices = self.enforce_capacity(topk_indices, topk_weights, expert_capacity)

        # 5. Update expert selection statistics
        valid_indices = topk_indices[topk_indices != -1]
        counts = torch.bincount(valid_indices, minlength=self.num_experts).detach().to(self.expert_counts.dtype)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
        self.expert_counts += counts[:self.num_experts]

        # 6. Sparse expert computation
        unique_experts = torch.unique(valid_indices).cpu().tolist()
        unique_experts = [e for e in unique_experts if e != -1 and 0 <= e < self.num_experts]
        expert_outputs = {}

        for e in unique_experts:
            # Process inputs for each active expert
            mask = (topk_indices == e).any(dim=1)
            expert_input = x[mask]
            expert_outputs[int(e)] = self.experts[e](expert_input)['pooler_output']

        # 7. Assemble final outputs
        outputs = torch.zeros(batch_size, max_k, expert_outputs[next(iter(expert_outputs))].shape[-1],
                              device=x.device)

        for k in range(max_k):
            expert_ids = topk_indices[:, k]
            valid_mask = expert_ids != -1

            for e in unique_experts:
                current_mask = (expert_ids == e) & valid_mask
                mask_e = (topk_indices == e).any(dim=1)
                combined_mask = current_mask & mask_e

                indices = torch.where(combined_mask)[0]
                if indices.numel() == 0:
                    continue

                selected_indices_e = torch.where(mask_e)[0]
                pos_in_selected = torch.searchsorted(selected_indices_e, indices)

                outputs[indices, k] = expert_outputs[e][pos_in_selected]

        # 8. Weighted combination
        valid_mask = (topk_indices != -1).float()
        normalized_weights = F.softmax(topk_weights * valid_mask, dim=1)
        weighted_output = torch.sum(outputs * normalized_weights.unsqueeze(-1), dim=1)

        # 9. Load balancing regularization
        expert_usage = routing_weights.mean(0).detach()
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(expert_usage, op=torch.distributed.ReduceOp.SUM)
            expert_usage /= torch.distributed.get_world_size()

        # endsure gradient is backpropagatable for all experts
        regularization_loss = 0.0
        for expert in self.experts:
            for param in expert.parameters():
                regularization_loss += torch.norm(param, p=2)**2

        added_loss = self.balance_alpha * torch.var(expert_usage) + self.regularization_alpha * regularization_loss


        # weighted_output=torch.randn(weighted_output.shape).to(x.device)

        return {
            'pooler_output': weighted_output,
            'added_loss': added_loss,
            'expert_usage': expert_usage.detach(),
            'expert_counts': self.expert_counts.detach()
        }

    def reset_expert_counts(self):
        self.expert_counts.zero_()


# -------------------






def model_selection(model_name='ATMS', load_dir=None, method='clip',confirm_load=True):
    """Select and initialize a model with optional checkpoint loading.

    Args:
        model_name (str): Name of the model architecture to initialize
        load_dir (str, optional): Directory containing checkpoint files
        method (str): 'clip' to wrap with Encoder_CLIP_training, 'pure' for raw model

    Returns:
        Model instance, optionally wrapped with Encoder_CLIP_training
    """

    def extract_encoder_state_dict(checkpoint):
        """Extract encoder parameters from checkpoint by removing 'encoder.' prefix"""
        return {k[len("encoder."):]: v for k, v in checkpoint.items()
                if k.startswith("encoder.")}

    # Default configuration shared by most models
    default_config = Config(
        num_channels=32,
        d_model=512,
        seq_len=512,
        channel_wise_layer_number_first=12,
        embedding_dim=3520,
        e_layers=2,
        proj_dim=768,
        freqN=512
    )

    def wrap_model(model):
        """Apply Encoder_CLIP_training wrapper based on method parameter"""
        return Encoder_CLIP_training(model) if method == 'clip' else model

    def load_weights(model, ckpt_path):
        """Load weights from checkpoint if path provided"""
        if load_dir:
            state_dict = torch.load(ckpt_path, weights_only=False)
            model.load_state_dict(extract_encoder_state_dict(state_dict), assign=True)
            print(f"Loaded weights from {ckpt_path}")
        return model

    # Model initialization
    # if model_name == 'EEGsuper':
    #     config = Config(**{**default_config.__dict__, 'e_layers': 12})
    #     model = EEGSuper(config)

    if model_name == 'ATMSmodify':
        default_config.e_layers=8
        default_config.dropout=0.3
        model = ATMS_modify(default_config)

    elif model_name in ['ATMS', 'NICE', "channelNet", 'EEGITNet',
                        'ShallowFBCSPNet', 'ATCNet','EEGConformer','ATMS_modify']:
        # Map model names to their constructor functions
        model_map = {
            'ATMS': ATMS,
            'NICE': NICE,
            'EEGConformer': EEGConformer_Encoder,
            'EEGITNet': EEGITNet_Encoder,
            'ShallowFBCSPNet': ShallowFBCSPNet_Encoder,
            'ATCNet': ATCNet_Encoder,
            'channelNet':ChannelNet,
            'ATMS_modify':ATMS_modify
        }
        model = model_map[model_name](default_config)

    elif model_name == 'MLP':
        model = Projector(default_config)

    elif model_name == 'mix':
        # Initialize mixture of experts
        assert load_dir, "load_dir required for mix model"
        experts = [
            load_weights(ATMS(default_config), f'{load_dir}/ATMS.pth'),
            load_weights(EEGConformer_Encoder(default_config), f'{load_dir}/EEGConformer.pth'),
            load_weights(NICE(default_config), f'{load_dir}/NICE.pth'),
            load_weights(ChannelNet(default_config), f'{load_dir}/channelNet.pth')
        ]
        model = MOERouter(*experts)

    elif model_name == 'mix_resume':
        if load_dir and os.path.exists(f'{load_dir}/mix.ckpt'):
            print(f"Loading full mixture model from {load_dir}/mix.ckpt")
            experts = [
                ATMS(default_config),
                EEGConformer_Encoder(default_config),
                NICE(default_config),
                ChannelNet(default_config)
            ]
            from EEG_Encoder.Tools.lightingModule import LitModel_CLIP
            model=LitModel_CLIP.load_from_checkpoint(f'{load_dir}/mix.ckpt', EEGencoder=wrap_model(MOERouter(*experts)))
            return model
        else:
                experts = [
                    ATMS(default_config),
                    EEGConformer_Encoder(default_config),
                    NICE(default_config),
                    ChannelNet(default_config),
                ]
                model = wrap_model(MOERouter(*experts))
                if load_dir:
                    model.load_state_dict(torch.load(f'{load_dir}/mix.pth', weights_only=False))
                    print(f"Loaded full mixture model from {load_dir}/mix.pth")
                return model

    else:
        raise ValueError(f'Unknown model: {model_name}')

    # Load individual model checkpoint if specified (except for mix cases)

    if model_name not in ['mix', 'mix_resume','mixE','mixE_resume',] and load_dir and confirm_load:
        checkpoint_path=os.path.join(load_dir,model_name+'.pth')
       
        if os.path.exists(checkpoint_path):
            model = load_weights(model, checkpoint_path)
        else:
            raise FileNotFoundError(f"NeuroTower Checkpoint not found at {checkpoint_path}")
        


    return wrap_model(model)







#--------------------------------NICE-----------------------------------#

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x

class NICE(nn.Module):
    def __init__(self,config=Config()):
        super().__init__()
        self.enc_eeg = Enc_eeg(num_channels=config.num_channels)
        self.proj_eeg = Proj_eeg(config.embedding_dim,proj_dim=config.proj_dim)
    def forward(self, data):
        eeg_embedding = self.enc_eeg(data)
        out = self.proj_eeg(eeg_embedding)

        return {"pooler_output":out,"last_hidden_state":eeg_embedding}
#########################################################################


#-------------------------------EEGNetv4--------------------------------#
class EEGNetv4_Encoder(nn.Module):
    def __init__(self,config=Config()):
        super().__init__()
        self.eegnet = EEGNetv4(
            n_chans=config.num_channels,
            n_outputs=config.proj_dim,
            n_times=config.seq_len,
            final_conv_length='auto',
            pool_mode='mean',
            F1=8,
            D=20,
            F2=160,
            kernel_length=4,
            third_kernel_size=(4, 2),
            drop_prob=0.25,

        )
    def forward(self, data):
        data = data.unsqueeze(0)
        data = data.reshape(data.shape[1], data.shape[2], data.shape[3], data.shape[0])
        prediction = self.eegnet(data)
        return {"pooler_output":prediction,"last_hidden_state":None}
#########################################################################


#--------------------------EEGConformer_Encoder-------------------------#
class EEGConformer_Encoder(nn.Module):
    def __init__(self,config=Config()):
        super().__init__()
        self.eegConformer = EEGConformer(n_outputs=config.proj_dim,
                                         n_times=config.seq_len,
                                    n_chans=config.num_channels,
                                   n_filters_time=40, 
                                   filter_time_length=10, 
                                   pool_time_length=25, 
                                   pool_time_stride=5, 
                                   drop_prob=0.25, 
                                   att_depth=config.channel_wise_layer_number_first,
                                   att_heads=8,
                                   att_drop_prob=0.5, 
                                   final_fc_length=3840,
                                   return_features=False,
                                   chs_info=None,
                                    add_log_softmax=False      )
    def forward(self, data):
        # data = data.unsqueeze(0)
        # data = data.reshape(data.shape[1], data.shape[2], data.shape[3], data.shape[0])
        # print(data.shape)
        prediction = self.eegConformer(data)
        return {"pooler_output":prediction,"last_hidden_state":None}
#########################################################################


#-----------------------------EEGITNet_Encoder--------------------------#
class EEGITNet_Encoder(nn.Module):
    def __init__(self,config=Config()):
        super().__init__()
        self.shape = (32, 513)
        self.eegEEGITNet = EEGITNet(n_outputs=config.proj_dim,
                                  n_chans=config.num_channels,
                                  n_times=None, 
                                  drop_prob=0.4, 
                                  chs_info=None, 
                                  input_window_seconds=config.input_window_seconds,
                                  sfreq=config.freqN,
                                  input_window_samples=config.seq_len,
                                  add_log_softmax=False)

    def forward(self, data):
        prediction = self.eegEEGITNet(data)
        return {"pooler_output":prediction,"last_hidden_state":None}
#########################################################################


#--------------------------------MLP------------------------------------#
def make_block(h_c, h_l,dropout_rate=0.25):
    block = nn.Sequential(
        nn.LayerNorm(h_l),
        nn.Linear(h_l, h_l), 
        nn.GELU(),
        nn.Dropout(dropout_rate),  
        Rearrange('B C L->B L C'),
        nn.LayerNorm(h_c),
        nn.Linear(h_c, h_c), 
        nn.GELU(),
        nn.Dropout(dropout_rate),  
        Rearrange('B L C->B C L'),
    )
    return block

class Projector(nn.Module):

    def __init__(self, config:Config, h_dim=(64, 1024), n_hidden_layer=2,dropout_rate=0.25):
        # in_features: (c, l)
        super().__init__()

        c, l = config.num_channels, config.seq_len
        h_c, h_l = h_dim
        c_o, l_o = 1, config.proj_dim

        self.input_layer = nn.Sequential(
            nn.LayerNorm(l),
            nn.Linear(l, h_l), 
            nn.GELU(),
            nn.Dropout(dropout_rate),  
            Rearrange('B C L->B L C'),
            nn.LayerNorm(c),
            nn.Linear(c, h_c), 
            nn.GELU(),
            nn.Dropout(dropout_rate),  
            Rearrange('B L C->B C L'),
        )
        
        self.output_layer = nn.Sequential(
            nn.LayerNorm(h_l),
            nn.Linear(h_l, l_o), 
            nn.GELU(),
            nn.Dropout(dropout_rate),  
            Rearrange('B C L->B L C'),
            nn.LayerNorm(h_c),
            nn.Linear(h_c, c_o), 
            nn.GELU(),
            nn.Dropout(dropout_rate),  
            Rearrange('B L C->B (C L)'),
        )
        
        self.blocks = nn.Sequential(*[
            make_block(h_c, h_l) for _ in range(n_hidden_layer)
        ])
        
        self.projector = nn.Sequential(*[
            self.input_layer,
            self.blocks,
            self.output_layer,
        ])
    def forward(self, eeg_embeds):
        eeg_embeds = self.projector(eeg_embeds)
        # print("eeg_embeds")
        # print(eeg_embeds.shape)
        eeg_features = F.normalize(eeg_embeds, dim=-1)
        return {"pooler_output":eeg_features,"last_hidden_state":None}
#########################################################################


#-------------------------ShallowFBCSPNet_Encoder-----------------------#
class ShallowFBCSPNet_Encoder(nn.Module):
    def __init__(self,config=Config()):
        super().__init__()
        self.ShallowFBCSPNet = ShallowFBCSPNet(n_chans=config.num_channels,
                                         n_outputs=config.proj_dim,
                                         n_times=config.seq_len,
                                         n_filters_time=20, 
                                         filter_time_length=20,
                                         n_filters_spat=20,
                                         pool_time_length=25, 
                                         pool_time_stride=5, 
                                         final_conv_length='auto', 
                                         pool_mode='mean', 
                                         split_first_layer=True,
                                         batch_norm=True, 
                                         batch_norm_alpha=0.1, 
                                         drop_prob=0.5,
                                         chs_info=None, 
                                         input_window_seconds=config.input_window_seconds,
                                         sfreq=config.freqN,
                                         add_log_softmax=False)
    def forward(self, data):
        prediction = self.ShallowFBCSPNet(data)
        return {"pooler_output":prediction,"last_hidden_state":None}
#########################################################################


#---------------------------ATCNet_Encoder------------------------------#
class ATCNet_Encoder(nn.Module):
    def __init__(self,config=Config()):
        super().__init__()
        self.eegATCNet = ATCNet(n_chans=config.num_channels,
                                n_outputs=config.proj_dim,
                                input_window_seconds=config.input_window_seconds,
                                sfreq=config.freqN,
                                conv_block_n_filters=8,
                                conv_block_kernel_length_1=31,
                                conv_block_kernel_length_2=7,
                                conv_block_pool_size_1=4,
                                conv_block_pool_size_2=3,
                                conv_block_depth_mult=2,
                                conv_block_dropout=0.3,
                                n_windows=5,
                                att_head_dim=4,
                                att_num_heads=2,
                                att_dropout=0.5,
                                tcn_depth=2,
                                tcn_kernel_size=4,
                                tcn_n_filters=16,
                                tcn_dropout=0.3,
                                tcn_activation=nn.ELU(),
                                concat=False,
                                max_norm_const=0.25,
                                chs_info=None,
                                n_times=None,
                                n_channels=None,
                                n_classes=None,
                                input_size_s=None,
                                add_log_softmax=False)

    def forward(self, data):
        # print("data", data.shape)
        prediction = self.eegATCNet(data)
        return {"pooler_output":prediction,"last_hidden_state":None}
#########################################################################


#-------------------------------Meta------------------------------------#
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#
#         div_term = torch.exp(torch.arange(0, d_model + 1, 2).float() * (-math.log(10000.0) / d_model))
#
#         pe[:, 0::2] = torch.sin(position * div_term[:d_model // 2 + 1])
#         pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
#
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         pe = self.pe[:x.size(0), :].unsqueeze(1).repeat(1, x.size(1), 1)
#         x = x + pe
#         return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, batch_first=False):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)
        self.batch_first = batch_first

    def forward(self, x):
        if self.batch_first:
            pe = self.pe[:x.size(1), :].unsqueeze(0)  # (1, sequence_length, d_model)
            pe = pe.expand(x.size(0), -1, -1)  # (batch_size, sequence_length, d_model)
        else:
            pe = self.pe[:x.size(0), :].unsqueeze(1)  #  (sequence_length, 1, d_model)
            pe = pe.expand(-1, x.size(1), -1)  # (sequence_length, batch_size, d_model)

        x = x + pe
        return x


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, batch_first=True):
        super(LearnablePositionalEncoding, self).__init__()
        self.batch_first = batch_first
        pe = torch.zeros(max_len, d_model)
        self.pe = nn.Parameter(pe)

        nn.init.normal_(self.pe, mean=0, std=d_model ** -0.5)

    def forward(self, x):
        if self.batch_first:
            pe = self.pe[:x.size(1), :].unsqueeze(0)
            pe = pe.expand(x.size(0), -1, -1)
        else:
            pe = self.pe[:x.size(0), :].unsqueeze(1)
            pe = pe.expand(-1, x.size(1), -1)
        x = x + pe
        return x

class EEGAttention(nn.Module):
    def __init__(self, channel, d_model, nhead):
        super(EEGAttention, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model,batch_first=False)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.channel = channel
        self.d_model = d_model

    def forward(self, src):
        src = src.permute(2, 0, 1)  # Change shape to [time_length, batch_size, channel]
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output.permute(1, 2, 0)  # Change shape back to [batch_size, channel, time_length]

class MetaEEG(nn.Module):
    def __init__(self, num_channels, sequence_length, num_subjects=1, num_features=64, num_latents=1024, num_blocks=1):
        super(MetaEEG, self).__init__()
        self.attention_model = EEGAttention(num_channels, num_channels, nhead=1)               
        # self.subject_wise_linear = nn.ModuleList([nn.Linear(sequence_length, sequence_length) for _ in range(num_subjects)])
        self.conv_blocks = nn.Sequential(*[ConvBlock(num_channels, sequence_length) for _ in range(num_blocks)],
                                         Rearrange('B C L->B L C'))
        self.linear_projection = nn.Sequential(
                                            Rearrange('B L C->B C L'),
                                            nn.Linear(sequence_length, num_latents),
                                            Rearrange('B C L->B L C'))
        self.temporal_aggregation = nn.Linear(sequence_length, 1)
        self.clip_head = MLPHead(num_latents, num_latents)
        self.mse_head = MLPHead(num_latents, num_latents)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/0.01))
        # self.loss_func = ClipLoss()
        
    def forward(self, x, subject_id):
        # print(f'Input shape: {x.shape}')
        # attn_output, _ = self.attention(x, x, x)
       
        x = self.attention_model(x)
        # print(f'After attention shape: {x.shape}')
         
        x = self.subject_wise_linear[subject_id](x)
        # print(f'After subject-specific linear transformation shape: {x.shape}')
        
        x = self.conv_blocks(x)
        # print(f'After convolutional blocks shape: {x.shape}')
        
        # x = self.conv_blocks(x)
        # print(f'After convolutional blocks shape: {x.shape}')
        
        x = self.linear_projection(x)
        # print(f'After linear projection shape: {x.shape}')
        
        x = self.temporal_aggregation(x)
        # print(f'After temporal aggregation shape: {x.shape}')

        clip_out = self.clip_head(x)
        # print(f'Clip head output shape: {clip_out.shape}')
    
        mse_out = self.mse_head(x)
        # print(f'MSE head output shape: {mse_out.shape}')

        return clip_out, mse_out

class ConvBlock(nn.Module):
    def __init__(self, num_channels, num_features):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(num_channels, num_features, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(num_features, num_features, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(num_features, num_features, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.LayerNorm(num_features)
        self.norm2 = nn.LayerNorm(num_features)
        self.norm3 = nn.LayerNorm(num_features)
        self.residual_conv = nn.Conv1d(num_channels, num_features, kernel_size=1)

    def forward(self, x):
        # print(f'ConvBlock input shape: {x.shape}')
        residual = self.residual_conv(x)
        # residual = x
        # print(f'residual shape: {residual.shape}')
        
        x = F.gelu(self.conv1(x))
        x = self.norm1(x)
        # print(f'After first convolution shape: {x.shape}')
                
        x = F.gelu(self.conv2(x))
        x = self.norm2(x)
        # print(f'After second convolution shape: {x.shape}')
        
        x = F.gelu(self.conv3(x))
        x = self.norm3(x)
        # print(f'After third convolution shape: {x.shape}')
        
        x += residual
        # print(f'ConvBlock output shape: {x.shape}')
        return x

class MLPHead(nn.Module):
    def __init__(self, in_features, num_latents, dropout_rate=0.25):
        super(MLPHead, self).__init__()

        self.layer1 = nn.Sequential(
            Rearrange('B C L->B L C'),
            nn.LayerNorm(in_features),
            nn.Linear(in_features, num_latents),
            nn.GELU(),
            nn.Dropout(dropout_rate), 
            Rearrange('B L C->B (C L)'),
        )
    def forward(self, x):
        # print(f'MLPHead input shape: {x.shape}')
        x = self.layer1(x)
        # print(f'After first layer of MLPHead shape: {x.shape}')
        return x
#########################################################################

#-------------------------------ATMS------------------------------------#

#########################################################################
class ATMS(nn.Module):
    def __init__(self, config=Config()):
        super(ATMS, self).__init__()
        default_config = config
        self.encoder = iTransformer(default_config, joint_train=False)
        self.enc_eeg = Enc_eeg(num_channels=default_config.num_channels)
        self.proj_eeg = Proj_eeg(embedding_dim=config.embedding_dim,proj_dim=config.proj_dim)


    def forward(self, x, subject_ids=None):
        x = self.encoder(x, None, subject_ids)
        x = F.normalize(x,dim=-1)
        out = self.proj_eeg(self.enc_eeg(x))
        return {"pooler_output":out,"last_hidden_state":x}
#########################################################################

#--------------------------ATMS_modify----------------------------------#

#########################################################################
class ATMS_modify(nn.Module):
    def __init__(self, config=Config()):
        super(ATMS_modify, self).__init__()
        default_config = config
        self.encoder = iTransformer_Modify(default_config, joint_train=False)
        self.enc_eeg = Enc_eeg(num_channels=default_config.num_channels)
        self.proj_eeg = Proj_eeg(embedding_dim=config.embedding_dim,proj_dim=config.proj_dim)


    def forward(self, x, subject_ids=None):
        x = self.encoder(x, None, subject_ids)
        x = F.normalize(x,dim=-1)
        out = self.proj_eeg(self.enc_eeg(x))
        return {"pooler_output":out,"last_hidden_state":x}
#########################################################################



# Channelnet

class ConvLayer2D(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel, stride, padding, dilation):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(True))
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel,
                                          stride=stride, padding=padding, dilation=dilation, bias=True))
        self.add_module('drop', nn.Dropout2d(0.2))

    def forward(self, x):
        return super().forward(x)

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers, kernel_size, stride, dilation_list):
        super().__init__()
        if len(dilation_list) < n_layers:
            dilation_list = dilation_list + [dilation_list[-1]] * (n_layers - len(dilation_list))

        padding = []
        for dilation in dilation_list:
            kernel_width = kernel_size[1]
            dilation = dilation[1]
            padding_needed = (kernel_width - 1) * dilation // 2
            padding.append((0, padding_needed))

        self.layers = nn.ModuleList([
            ConvLayer2D(
                in_channels, out_channels, kernel_size,
                stride=stride,
                padding=padding[i],
                dilation=(1, dilation_list[i][1])
            ) for i in range(n_layers)
        ])

    def forward(self, x):
        return torch.cat([layer(x) for layer in self.layers], 1)

class SpatialBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_spatial_layers, stride, input_height):
        super().__init__()
        kernel_list = [(input_height // (i + 1), 1) for i in range(num_spatial_layers)]
        padding = [((kernel[0] - 1) // 2, 0) for kernel in kernel_list]

        self.layers = nn.ModuleList([
            ConvLayer2D(
                in_channels, out_channels, kernel_list[i], stride, padding[i], 1
            ) for i in range(num_spatial_layers)
        ])

    def forward(self, x):
        features = [layer(x) for layer in self.layers]
        return torch.cat(features, 1)

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class FeaturesExtractor(nn.Module):
    def __init__(self, in_channels, temp_channels, out_channels, input_height,
                 temporal_kernel=(1, 33), temporal_stride=(1, 2),
                 temporal_dilation_list=[(1, 1), (1, 2), (1, 4), (1, 8)],
                 num_temp_layers=4, num_spatial_layers=4, spatial_stride=(2, 1),
                 num_residual_blocks=4, down_kernel=3, down_stride=2):

        super().__init__()
        # (batch, 32, 512) -> (batch, 1, 32, 512)
        self.input_proj = nn.Conv2d(1, in_channels, kernel_size=(1, 1))

        self.temporal_block = TemporalBlock(
            in_channels, temp_channels, num_temp_layers,
            kernel_size=(1, temporal_kernel[1]),
            stride=temporal_stride,
            dilation_list=temporal_dilation_list
        )

        # Spatial Block（处理空间维度）
        self.spatial_block = SpatialBlock(
            temp_channels * num_temp_layers, out_channels,
            num_spatial_layers, spatial_stride, input_height
        )

        # Residual Blocks
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(
                    out_channels * num_spatial_layers,
                    out_channels * num_spatial_layers
                ),
                ConvLayer2D(
                    out_channels * num_spatial_layers,
                    out_channels * num_spatial_layers,
                    kernel=(down_kernel, 1),
                    stride=(down_stride, 1),
                    padding=((down_kernel-1)//2, 0),
                    dilation=1
                )
            ) for _ in range(num_residual_blocks)
        ])

        # Final Conv
        self.final_conv = ConvLayer2D(
            out_channels * num_spatial_layers, out_channels,
            kernel=(1, 1), stride=1, padding=0, dilation=1
        )

    def forward(self, x):
        # Input shape: (B, C, H, W) = (batch, 1, 32, 512)
        x = self.input_proj(x)
        x = self.temporal_block(x)
        x = self.spatial_block(x)

        for res_block in self.res_blocks:
            x = res_block(x)

        x = self.final_conv(x)
        return x

class ChannelNet(nn.Module):
    def __init__(self, config=Config(num_channels=32, d_model=512,seq_len=512,embedding_dim=3520,e_layers=2,proj_dim=768)):
        super().__init__()

        self.feature_extractor = FeaturesExtractor(
            in_channels=1,
            temp_channels=16,
            out_channels=32,
            input_height=32,
            temporal_kernel=(1, 33),
            temporal_stride=(1, 2),
            num_temp_layers=4,
            num_spatial_layers=4,
            num_residual_blocks=2
        )

        with torch.no_grad():
            dummy = torch.randn(2, 1, 32, 512)  # (batch, 1, 32, 512)
            out = self.feature_extractor(dummy)
            in_features = out.view(out.size(0), -1).shape[1]

        self.MLP = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, 768)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, 32, 512)
        features = self.feature_extractor(x)
        features = features.flatten(1)
        return {"pooler_output":self.MLP(features),"last_hidden_state":features}






#-------------------------------EEGSuper------------------------------------#

class EEGSuper(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.time_pos = nn.Parameter(torch.randn(1, 32, 768))
        self.space_pos = nn.Embedding(32, 768)
        self.pre_linear = nn.Linear(512, 768)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=768,
                nhead=8,
                dim_feedforward=768*2,
                dropout=0.3,
                batch_first=True,
                norm_first=True
            ),
            num_layers=config.e_layers
        )

        self.pooler = nn.Sequential(
            nn.Linear(768, 768*2),
            nn.GELU(),
            nn.LayerNorm(768*2),
            nn.Dropout(0.5),
            nn.Linear(768*2, config.proj_dim),
        )

    def forward(self, x):

        b, c, t = x.shape

        x = self.pre_linear(x)

        x = x + self.time_pos

        space_emb = self.space_pos(torch.arange(c, device=x.device))  # (32, 768)
        x = x + space_emb.unsqueeze(0)

        hidden_states = self.transformer(x)  # (b, 32, 768)

        query = hidden_states.mean(dim=1, keepdim=True)  # (b, 1, 768)
        attn_weights = torch.softmax(
            (hidden_states @ query.transpose(1,2)) / (768**0.5),
            dim=1
        )  # (b, 32, 1)
        pooled = (attn_weights * hidden_states).sum(dim=1)  # (b, 768)

        return {"pooler_output": self.pooler(pooled),'last_hidden_states':hidden_states}
#########################################################################




# class MaskToken(nn.Module):
#     def __init__(self, d_model, mask_ratio=0.15):
#         """
#         Args:
#             d_model (int): Embedding dimension of input tokens.
#             mask_ratio (float): Fraction of tokens to mask (0 < mask_ratio < 1).
#         """
#         super(MaskToken, self).__init__()
#         self.d_model = d_model
#         self.mask_ratio = mask_ratio
#         self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
#         nn.init.normal_(self.mask_token, std=.02)
#
#     def forward(self, x):
#         """
#         Args:
#             x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
#
#         Returns:
#             masked_x (torch.Tensor): Masked tensor with same shape as input.
#             mask_ids (torch.Tensor): Boolean mask of shape (batch_size, seq_len).
#         """
#         batch_size, seq_len, _ = x.shape
#         num_mask = max(1, int(seq_len * self.mask_ratio))  # Ensure at least 1 token is masked
#
#         # Generate mask indices
#         mask_indices = torch.rand(batch_size, seq_len, device=x.device).argsort(dim=-1)[:, :num_mask]
#
#         # Create mask IDs
#         mask_ids = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=x.device)
#         mask_ids.scatter_(1, mask_indices, True)
#
#         # Apply masking
#         masked_x = x.clone()  # Avoid modifying the original input
#         masked_x[mask_ids] = self.mask_token
#
#         return masked_x, mask_ids
#
#
#
# class EEGSuperEncoder(nn.Module):
#     def __init__(self, config=Config(),use_mamba=False,position_encoding='learnable'):
#         super(EEGSuperEncoder, self).__init__()
#         self.before_mlp=nn.Linear(config.d_model,config.d_model)
#         self.patch_layer=PatchEmbeddingES(config)
#         self.masking=MaskToken(config.d_model)
#         if position_encoding=='learnable':
#             self.positional_embedding=LearnablePositionalEncoding(config.d_model, max_len=64,batch_first=True)
#         elif position_encoding=='sinusoidal':
#             self.positional_embedding=PositionalEncoding(config.d_model, max_len=64,batch_first=True)
#         else:
#             raise ValueError('position_encoding should be either learnable or sinusoidal')
#
#         if use_mamba==False:
#             self.encoder_1=nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=config.d_model, nhead=config.n_heads,batch_first=True), num_layers=config.channel_wise_layer_number_first,)
#             self.encoder_2=nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=config.patch_size, nhead=config.n_heads,batch_first=True), num_layers=config.temp_wise_layer_number)
#             self.encoder_3=nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=config.d_model, nhead=config.n_heads,batch_first=True), num_layers=config.channel_wise_layer_number_second)
#         else:
#             self.encoder_1=Mamba(d_model=config.d_model, d_state=16, d_conv=4, expand=config.channel_wise_layer_number_first)
#             self.encoder_2=Mamba(d_model=config.patch_size, d_state=16, d_conv=4, expand=config.temp_wise_layer_number)
#             self.encoder_3=Mamba(d_model=config.d_model, d_state=16, d_conv=4, expand=config.temp_wise_layer_number)
#
#     def forward(self, x, mask=False):
#         x = self.before_mlp(x)
#         masks_id = None
#         if mask:
#             # print("activate and forward masking")
#             x, masks_id = self.masking(x)
#         # else:
#         #     print("forward without masking")
#         x = self.positional_embedding(x)
#         x = self.encoder_1(x)
#         x=self.patch_layer.patchify(x)
#         x = self.encoder_2(x.transpose(-2, -1))
#         x=self.patch_layer.unpatchify(x)
#         x = self.encoder_3(x.transpose(-2, -1))
#         # shape of x: (batch_size, 32, 512)
#         return x,masks_id
#
# class EEGSuperRouter(nn.Module):
#     def __init__(self, config=Config()):
#         super(EEGSuperRouter, self).__init__()
#         self.avg_pool=lambda x: torch.mean(x,dim=-2)
#         self.router=nn.Sequential(
#             nn.Linear(config.d_model,config.d_model),
#             nn.LeakyReLU(),
#             nn.Dropout(p=0.5),
#             nn.Linear(config.d_model,config.finetune_label_class),
#         )
#     def forward(self, x):
#         x=self.avg_pool(x)
#         x=self.router(x)
#         return x
#
# class EEGSuperLabelPredictor(nn.Module):
#     def __init__(self, config=Config()):
#         super(EEGSuperLabelPredictor, self).__init__()
#         self.avg_pool=lambda x: torch.mean(x,dim=-2)
#         self.router=nn.Sequential(
#             nn.Linear(config.d_model,config.pretrain_label_class),
#         )
#     def forward(self, x):
#         x=self.avg_pool(x)
#         x=self.router(x)
#         return x
#
# class PatchEmbeddingES(nn.Module):
#     def __init__(self,config=Config()):
#         super(PatchEmbeddingES, self).__init__()
#         self.cut_ratio = int(config.seq_len/config.patch_size)
#         self.seq_len = int(config.seq_len)
#         self.conv2d_layer = nn.Conv2d(in_channels=1, out_channels=self.cut_ratio, kernel_size=(1, self.cut_ratio), stride=(1, self.cut_ratio))
#
#     def forward(self, x):
#         pass
#
#     def patchify(self, x):
#         x_reshaped = x.unsqueeze(1)
#         output_conv2d = self.conv2d_layer(x_reshaped)
#         output = output_conv2d.view(output_conv2d.size(0), -1,  self.seq_len // self.cut_ratio)
#         return output
#
#     def unpatchify(self, x):
#         x = x.view(x.size(0), self.seq_len, -1 )
#         return x



#-------------------------------BIOT------------------------------------#
class PatchFrequencyEmbedding(nn.Module):
    def __init__(self, emb_size=256, n_freq=101):
        super().__init__()
        self.projection = nn.Linear(n_freq, emb_size)

    def forward(self, x):
        """
        x: (batch, freq, time)
        out: (batch, time, emb_size)
        """
        x = x.permute(0, 2, 1)
        x = self.projection(x)
        return x

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.clshead = nn.Sequential(
            nn.ELU(),
            nn.Linear(emb_size, n_classes),
        )

    def forward(self, x):
        out = self.clshead(x)
        return out

class PositionalEncodingBIOT(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super(PositionalEncodingBIOT, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            x: `embeddings`, shape (batch, max_len, d_model)
        Returns:
            `encoder input`, shape (batch, max_len, d_model)
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)

class BIOTEncoder(nn.Module):
    def __init__(
            self,
            emb_size=256,
            heads=8,
            depth=4,
            n_channels=16,
            n_fft=200,
            hop_length=100,
            method='fill',
            **kwargs
    ):
        super().__init__()
        from linear_attention_transformer import LinearAttentionTransformer

        self.n_fft = n_fft
        self.hop_length = hop_length

        self.patch_embedding = PatchFrequencyEmbedding(
            emb_size=emb_size, n_freq=self.n_fft // 2 + 1
        )
        self.transformer = LinearAttentionTransformer(
            dim=emb_size,
            heads=heads,
            depth=depth,
            max_seq_len=1024,
            attn_layer_dropout=0.2,
            attn_dropout=0.2,
        )
        self.positional_encoding = PositionalEncodingBIOT(emb_size)

        # channel token, N_channels >= your actual channels
        self.channel_tokens = nn.Embedding(n_channels, emb_size)
        self.index = nn.Parameter(
            torch.LongTensor(range(n_channels)), requires_grad=False
        )
        self.fill_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.fill_method = method

    def stft(self, sample):
        spectral = torch.stft(
            input=sample.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            center=False,
            onesided=True,
            return_complex=True,
            window=torch.hann_window(self.n_fft, device=sample.device)
        )
        return torch.abs(spectral)

    def forward(self, x, n_channel_offset=0, perturb=False):
        """
        x: [batch_size, channel, ts]
        output: [batch_size, emb_size], emb_seq, masks
        """
        emb_seq = []
        masks = []

        for i in range(x.shape[1]):
            channel_spec_emb = self.stft(x[:, i:i + 1, :])
            channel_spec_emb = self.patch_embedding(channel_spec_emb)
            batch_size, ts, _ = channel_spec_emb.shape
            channel_token_emb = (
                self.channel_tokens(self.index[i + n_channel_offset])
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(batch_size, ts, 1)
            )
            channel_emb = self.positional_encoding(channel_spec_emb + channel_token_emb)

            mask = None
            if perturb:
                # Randomly select two time steps to mask
                selected_ts = np.random.choice(range(ts), size=1, replace=False)
                selected_ts = np.sort(selected_ts)  # Sort to maintain order

                if self.fill_method == 'fill':
                    mask = torch.ones((batch_size, ts, 1), device=x.device)
                    mask[:, selected_ts] = 0  # Mask positions to be replaced with learnable tokens
                    # Create a tensor of learnable tokens matching the shape of the masked positions
                    learnable_tokens = self.fill_token.repeat(batch_size, ts, 1)
                    # Apply the mask to select original embeddings or learnable tokens
                    channel_emb = torch.where(mask == 1, channel_emb, learnable_tokens)
                elif self.fill_method == 'select':
                    # Select only the specified time steps
                    channel_emb = channel_emb[:, selected_ts]
                    mask = None  # Set mask to None as we're not using it in this case

            emb_seq.append(channel_emb)
            masks.append(mask)

        # Concatenate embeddings from all channels
        emb = torch.cat(emb_seq, dim=1)

        # Process masks
        final_mask = None
        if perturb and self.fill_method == 'fill':
            # Concatenate masks across channels, ensuring they match the concatenated embedding's shape
            final_mask = torch.cat([m for m in masks if m is not None], dim=1)

        # Pass through transformer and average across sequence length
        emb = self.transformer(emb)


        return emb.mean(dim=1), emb, final_mask

class BIOTClassifier(nn.Module):
    def __init__(self, emb_size=256, heads=8, depth=4, n_classes=6, **kwargs):
        super().__init__()
        self.biot = BIOTEncoder(emb_size=emb_size, heads=heads, depth=depth, **kwargs)
        self.classifier = ClassificationHead(emb_size, n_classes)

    def forward(self, x):
        x = self.biot(x)
        x = self.classifier(x)
        return x

class BIOTCLIPMapper(nn.Module):
    def __init__(self, emb_size=256, heads=8, depth=4, CLIPemb=1024, **kwargs):
        super().__init__()
        self.biot = BIOTEncoder(emb_size=emb_size, heads=heads, depth=depth, **kwargs)
        self.mapper = nn.Sequential(
            nn.Linear(emb_size, int(emb_size / 3)),
            nn.GELU(),
            nn.Linear(int(emb_size / 3), CLIPemb),
        )

    def forward(self, x):
        emb,_,_ = self.biot(x)
        x = self.mapper(emb)
        return x

class BIOTUnsupervisedPretrain(nn.Module):
    def __init__(self, emb_size=256, heads=8, depth=4, n_channels=32,method='fill'):
        super(BIOTUnsupervisedPretrain, self).__init__()
        self.biot = BIOTEncoder(emb_size, heads, depth, n_channels,method=method)
        self.T = 0.2
        self.prediction = nn.Sequential(
            nn.Linear(emb_size, int(emb_size/3)),
            nn.GELU(),
            nn.Linear(int(emb_size/3), emb_size),
        )

    def forward(self, x, n_channel_offset=0):
        emb_pert,emb_seq_pert,mask = self.biot(x, n_channel_offset, perturb=True)
        emb,emb_seq,_ = self.biot(x, n_channel_offset)

        emb_pert = self.prediction(emb_pert)
        loss,loss_log=self.compute_loss(emb,emb_pert,x)
        return emb,loss,loss_log


    def compute_loss(self,prest_samples_emb,prest_masked_emb,EEG):
        # L2 normalize
        prest_samples_emb = F.normalize(prest_samples_emb, dim=1, p=2)
        prest_masked_emb = F.normalize(prest_masked_emb, dim=1, p=2)
        N = EEG.shape[0]
        # representation similarity matrix, NxN
        logits = torch.mm(prest_samples_emb, prest_masked_emb.t()) / self.T
        labels = torch.arange(N).to(logits.device)
        contrastive_loss = F.cross_entropy(logits, labels, reduction="mean")
        loss = contrastive_loss
        loss_log = {"contrastive_loss": contrastive_loss}
        return loss,loss_log


# # -----------
# class EEFormerEncoder(nn.Module):
#     class PatchFrequencyEmbeddingEF(nn.Module):
#         def __init__(self, emb_size=256, input_size=957):
#             super().__init__()
#             self.projection = nn.Linear(input_size, emb_size)
#
#         def forward(self, x):
#             """
#             x: (batch, channel, freq, time)
#             out: (batch, time, emb_size)
#             """
#             x = x.flatten(2, 3).contiguous()
#             x = self.projection(x)
#             return x
#
#     def __init__(self, config=Config(), use_mamba=False, position_encoding='sinusoidal'):
#         super(EEFormerEncoder, self).__init__()
#         self.before_mlp_temporal = nn.Linear(config.d_model, config.d_model)
#         self.before_mlp_frq = nn.Linear(config.d_model, config.d_model)
#         self.after_mlp = nn.Linear(config.d_model * 2, 1024)
#         self.patch_layer = PatchEmbeddingES(config)
#
#         self.frq_learn_token = nn.Parameter(torch.randn(1, 1, config.d_model))
#         self.temp_learn_token = nn.Parameter(torch.randn(1, 1, config.d_model))
#
#         if position_encoding == 'learnable':
#             self.positional_embedding = LearnablePositionalEncoding(config.d_model, max_len=96, batch_first=True)
#         elif position_encoding == 'sinusoidal':
#             self.positional_embedding = PositionalEncoding(config.d_model, max_len=96, batch_first=True)
#         else:
#             raise ValueError('position_encoding should be either learnable or sinusoidal')
#
#         self.subject_embedding = nn.Embedding(32, config.d_model)
#
#         if use_mamba == False:
#             self.encoder_1 = nn.TransformerEncoder(
#                 nn.TransformerEncoderLayer(d_model=config.d_model, nhead=config.n_heads, batch_first=True,dropout=0.5),
#                 num_layers=config.channel_wise_layer_number_first,)
#         else:
#             self.encoder_1 = Mamba(d_model=config.d_model, d_state=16, d_conv=4,
#                                    expand=config.channel_wise_layer_number_first)
#
#         self.n_fft = 64
#
#         self.patch_frq_embedding = self.PatchFrequencyEmbeddingEF(
#             emb_size=512, input_size=957
#         )
#
#     def forward(self, x,subject_id=None):
#         if subject_id is not None:
#             subject_token = self.subject_embedding(subject_id)
#             subject_token = subject_token.unsqueeze(1)
#         else:
#             subject_token = torch.zeros(x.shape[0], 1, x.shape[2], device=x.device)
#
#         stft_result = self.stft(x)
#         frq_patches = self.patch_frq_embedding(stft_result)
#         frq_patches = self.before_mlp_frq(frq_patches)
#         time_patches = self.before_mlp_temporal(x)
#         # frq_token = self.frq_learn_token.repeat(x.shape[0], 1, 1)
#         # temp_token = self.temp_learn_token.repeat(x.shape[0], 1, 1)
#
#         x = torch.cat([time_patches, frq_patches,subject_token], dim=1)
#         x = self.positional_embedding(x)
#         x = self.encoder_1(x)
#         frq_mean = torch.mean(x[:,:32], dim=1)
#         temp_mean = torch.mean(x[:,32:], dim=1)
#         x=torch.cat([frq_mean,temp_mean],dim=1)
#         # x = torch.concat([x[:, 0, :], x[:, -2, :]], dim=1)
#         x = self.after_mlp(x)
#
#         return x
#
#     def stft(self, batch):
#         stft_result = []
#         for i in range(batch.shape[0]):
#             stft_result.append(torch.abs(torch.stft(
#                 input=batch[i],
#                 n_fft=self.n_fft,
#                 center=False,
#                 onesided=True,
#                 return_complex=True,
#                 window=torch.hann_window(self.n_fft, device=batch[i].device))
#             ))
#         return torch.stack(stft_result)



#-------------------------------Pretraining------------------------------------#

class AttentionUnsupervisedPretrain(nn.Module):
    def __init__(self, encoder,emb_size=1024,mask_ratio=0.6,seq_len=512):
        super(AttentionUnsupervisedPretrain, self).__init__()
        self.encoder = encoder
        self.T = 0.2
        self.emb_prediction = nn.Sequential(
            nn.Linear(emb_size, int(emb_size/4)),
            nn.GELU(),
            nn.Linear(int(emb_size/4), emb_size),
        )
        self.fft_prediction = nn.Sequential(
            nn.Linear(seq_len, int(emb_size/4)),
            nn.GELU(),
            nn.Linear(int(emb_size/4), seq_len // 2 + 1),
        )
        self.amp_prediction = nn.Sequential(
            nn.Linear(seq_len, int(emb_size/4)),
            nn.GELU(),
            nn.Linear(int(emb_size/4), seq_len),
        )
        self.mask_ratio = mask_ratio

    def forward(self, x):
        # x: (batch, channel, seq_len)
        batch_size, channel, seq_len = x.size()

        x_fft=self.compute_FFT(x)

        # Generate mask matrix (batch, channel) with 60% channels masked per sample
        k = int(channel * self.mask_ratio)
        # Randomly select k channels to mask for each sample
        indices = torch.rand(batch_size, channel, device=x.device).argsort(dim=1)
        mask = torch.zeros((batch_size, channel), dtype=torch.bool, device=x.device)
        mask.scatter_(1, indices[:, :k], 1)

        # Expand mask to match x dimensions (batch, channel, seq_len)
        mask = mask.unsqueeze(-1).expand_as(x)

        # Generate random noise with the same shape as x
        random_noise = torch.randn_like(x)
        random_noise = self.normalize_noise(x, random_noise, noise_scale=1.0)

        # Apply masking: replace masked channels with random noise
        x_masked = torch.where(mask, random_noise, x)

        # Get embeddings
        # self.encoder(x)[0]->(batch, emb_size) self.encoder(x)[1]->(batch, channel(token),emb_size)
        emb_predict = self.emb_prediction(self.encoder(x_masked)[0])  # Prediction from masked input
        emb,token_emb = self.encoder(x)                                # Original embeddings

        predict_fft = self.fft_prediction(token_emb)
        predict_amp = self.amp_prediction(token_emb)

        # Compute contrastive loss
        contrastive_loss= self.compute_loss_contrastive(emb, emb_predict, x)
        # Compute FFT and amplitude loss
        fft_loss, amp_loss = self.compute_FFT_amp_loss(x, x_fft, predict_amp, predict_fft)

        loss = contrastive_loss + fft_loss + amp_loss
        loss_log = {"contrastive_loss": contrastive_loss, "fft_loss": fft_loss, "amp_loss": amp_loss}

        return emb,loss, loss_log

    def normalize_noise(self,x, noise, noise_scale=1.0):

        # 计算原始信号的幅度（L2范数）
        x_norm = torch.norm(x, p=2, dim=-1, keepdim=True)  # (batch, channel, 1)

        # 计算噪声的幅度
        noise_norm = torch.norm(noise, p=2, dim=-1, keepdim=True)  # (batch, channel, 1)

        # 归一化噪声
        noise_normalized = noise / (noise_norm + 1e-8)  # 防止除零

        # 将噪声缩放到与原始信号相同的幅度
        noise_scaled = noise_normalized * x_norm * noise_scale

        return noise_scaled

    def compute_FFT(self, x):
        # x shape: (batch, channel, seq_len)
        x_fft = torch.fft.rfft(x, dim=-1, norm='ortho')
        x_fft = torch.abs(x_fft)
        return x_fft

    def compute_FFT_amp_loss(self, x, x_fft,predict_amp,predict_fft):
        fft_loss = F.mse_loss(
            predict_fft,
            x_fft.detach(),
            reduction="mean"
        )
        amp_loss = F.mse_loss(
            predict_amp,
            x.detach(),
            reduction="mean"
        )

        return fft_loss, amp_loss



    def compute_loss_contrastive(self,prest_samples_emb,prest_masked_emb,EEG):
        # L2 normalize
        prest_samples_emb = F.normalize(prest_samples_emb, dim=1, p=2)
        prest_masked_emb = F.normalize(prest_masked_emb, dim=1, p=2)
        N = EEG.shape[0]
        # representation similarity matrix, NxN
        logits = torch.mm(prest_samples_emb, prest_masked_emb.t()) / self.T
        labels = torch.arange(N).to(logits.device)
        contrastive_loss = F.cross_entropy(logits, labels, reduction="mean")
        return contrastive_loss







