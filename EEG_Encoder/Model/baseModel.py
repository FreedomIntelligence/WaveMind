import os

import torch
from torch.nn import functional as F

import math
import numpy as np
import torch.nn as nn

from EEG_Encoder.Model.NeuroLM_model.model_neurolm import GPTConfig, NeuroLM
from einops.layers.torch import Rearrange

from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_warn

from EEG_Encoder.Model.CommonBlock import Config, iTransformer, Enc_eeg, Proj_eeg, iTransformer_Modify

from EEG_Encoder.Model.CSBrain import create_csbrain_encoder

from braindecode.models import EEGNetv4, ATCNet, EEGConformer, EEGITNet, ShallowFBCSPNet



device= torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AlignmentEncoder(nn.Module):
    def __init__(self, encoder, config=Config()):
        super(AlignmentEncoder, self).__init__()
        self.encoder=encoder
        self.text_logit_scale = nn.Parameter(torch.ones([]) * np.log(1/0.07))
        self.image_logit_scale = nn.Parameter(torch.ones([]) * np.log(1/0.07))
    def forward(self, x):
        assert x.dim() == 3, f"Input must be a 3D tensor of shape (batch_size, num_channels, time),but get {x.shape}"
        res=self.encoder(x)
        return res




def model_selection(model_name='ATMS', load_dir=None, method='clip',confirm_load=True, checkpoint_name=None):
    """Select and initialize a model with optional checkpoint loading.

    Args:
        model_name (str): Name of the model architecture to initialize
        load_dir (str, optional): Directory containing checkpoint files
        method (str): 'clip' to wrap with AlignmentEncoder, 'pure' for raw model

    Returns:
        Model instance, optionally wrapped with AlignmentEncoder
    """


    sample_rate=512

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
        """Apply AlignmentEncoder wrapper based on method parameter"""
        if method == 'clip':
            return AlignmentEncoder(model)
        else:
            raise RuntimeError("Unexpected")

    def load_weights(model, ckpt_path):
        """Load weights from checkpoint if path provided"""
        if load_dir:
            state_dict = torch.load(ckpt_path, weights_only=False)
            encoder_state_dict = extract_encoder_state_dict(state_dict)
            
            # Load with strict mode and capture the result
            missing_keys, unexpected_keys = model.load_state_dict(encoder_state_dict, strict=False)

            if missing_keys:
                rank_zero_warn(f"Warning: Missing keys in checkpoint: {missing_keys}")
            if unexpected_keys:
                rank_zero_warn(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")

            # Count loaded parameters
            total_params = sum(p.numel() for p in model.parameters())
            loaded_params = sum(p.numel() for p in encoder_state_dict.values())
            rank_zero_info(f"[SUCCESS] Model checkpoint loaded successfully!")
            rank_zero_info(f"[SUCCESS] Checkpoint path: {ckpt_path}")
        else:
            rank_zero_warn(f"[DEBUG] load_dir is None, skipping checkpoint loading")
        return model

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


    elif 'NeuroLM' in model_name:
        sample_rate=200
        
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
        
        
        
        if model_name == 'NeuroLM-B':
            checkpoint_path_=os.path.join(load_dir,'NeuroLM-B.pt')
        elif model_name == 'NeuroLM-L':
            checkpoint_path_=os.path.join(load_dir,'NeuroLM-L.pt')
        else:
            raise ValueError(f'Unknown model: {model_name}.you should attach -B or -L')
            
        vq_checkpoint_path=os.path.join(load_dir,'VQ.pt')
        rank_zero_info(f"load pre-train checkpoint from {checkpoint_path_}, VQ checkpoint from {vq_checkpoint_path}")
        model = load_neurolm(checkpoint_path=checkpoint_path_, vq_checkpoint_path=vq_checkpoint_path) if load_dir else None

    elif 'CBraMod' in model_name:
        sample_rate=200
        from EEG_Encoder.Model.cbramod import CBraMod
        model = CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12,
                    nhead=8).to(device)
        checkpoint_path_=os.path.join(load_dir,'CbraMod.pth')
        rank_zero_info(f"load pre-train checkpoint from {checkpoint_path_}")
        model.load_state_dict(torch.load(checkpoint_path_,map_location=device),strict=False)

    elif model_name == 'CSBrain':
        sample_rate=200
        model = create_csbrain_encoder(
            num_channels=32,
            d_model=512,
            num_heads=8,
            num_layers=6,
            patch_size=20,
            proj_dim=default_config.proj_dim
        )
        
    else:
        raise ValueError(f'Unknown model: {model_name}')

    # Load individual model checkpoint if specified (except for mix cases)

    if load_dir and confirm_load:
        if checkpoint_name is None:
            checkpoint_path = f'{load_dir}/{model_name}_all.pth'
        elif os.path.isabs(checkpoint_name):
            # Absolute path, use as-is
            checkpoint_path = checkpoint_name
        elif checkpoint_name.startswith('./') or checkpoint_name.startswith('../'):
            # Relative path from current directory - convert to absolute path
            checkpoint_path = os.path.abspath(os.path.normpath(checkpoint_name))
        else:
            # Plain filename - join with load_dir
            checkpoint_path = os.path.normpath(os.path.join(load_dir, checkpoint_name))
        if os.path.exists(checkpoint_path):
            rank_zero_info(f"Found checkpoint at: {checkpoint_path}")
            model = load_weights(model, checkpoint_path)
            rank_zero_info(f"Successfully loaded checkpoint from {checkpoint_path}")
        else:
            # If checkpoint_name is specified, raise error when checkpoint cannot be found
            if checkpoint_name is not None:
                raise FileNotFoundError(f"Specified checkpoint {checkpoint_path} does not exist. Program cannot continue.")
            else:
                rank_zero_info(f"Checkpoint {checkpoint_path} does not exist. Skipping loading.")

    return wrap_model(model=model),sample_rate







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
        assert data.shape[-1] == 512, f"EEG shape error: {data.shape}"
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
        assert x.shape[-2] == 32, "The channel should be 32"
        assert x.shape[-1] == 512, "The last dimension of x must be 512"
        
        # x = F.normalize(x,dim=-1)

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







