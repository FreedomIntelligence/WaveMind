import torch
import sys



from einops.layers.torch import Rearrange
from torch import nn, Tensor
import sys
from EEG_Encoder.Model.Embed import DataEmbedding





class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40,num_channels=63):
        super().__init__()
        # Revised from ShallowNet
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), stride=(1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (num_channels, 1), stride=(1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        # b, _, _, _ = x.shape
        x = x.unsqueeze(1)
        # print("x", x.shape)
        x = self.tsconv(x)
        # print("tsconv", x.shape)
        x = self.projection(x)
        # print("projection", x.shape)
        return x

class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x

class Enc_eeg(nn.Sequential):
    def __init__(self, emb_size=40, num_channels=63):
        super().__init__(
            PatchEmbedding(emb_size, num_channels),
            FlattenHead(),
        )


class iTransformer(nn.Module):
    def __init__(self, configs, joint_train=False, num_subjects=10):
        from Transformer_EncDec import Encoder, EncoderLayer
        from SelfAttention_Family import FullAttention, AttentionLayer
        from Embed import DataEmbedding
        super(iTransformer, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.num_channels = configs.num_channels
        # Embedding
        self.enc_embedding = DataEmbedding(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout, joint_train=joint_train, num_subjects=num_subjects)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

    def forward(self, x_enc, x_mark_enc, subject_ids=None):
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc, subject_ids)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        enc_out = enc_out[:, :self.num_channels, :]
        return enc_out

class iTransformer_Modify(nn.Module):
    def __init__(self, configs, joint_train=False, num_subjects=10):
        super(iTransformer_Modify, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.num_channels = configs.num_channels
        # Embedding
        self.enc_embedding = DataEmbedding(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout, joint_train=joint_train, num_subjects=num_subjects)
        # Encoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
            configs.d_model,
            configs.n_heads,
            dim_feedforward=configs.d_model*6,
            batch_first=True,
            dropout=configs.dropout,
            ),configs.e_layers
        )
    def forward(self, x_enc, x_mark_enc, subject_ids=None):
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc, subject_ids)
        # enc_out = x_enc
        enc_out = self.encoder(enc_out)

        enc_out = enc_out[:, :self.num_channels, :]

        return enc_out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=1440, proj_dim=1024, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
            nn.Linear(proj_dim, proj_dim),
        )



class Config:
    def __init__(self, task_name='classification', seq_len=512, pred_len=250,
                 output_attention=False, d_model=256, embed='timeF',
                 freq='h', dropout=0.25, factor=1, n_heads=8,
                 e_layers=1, d_ff=256, activation='gelu',
                 enc_in=63, num_channels=32,embedding_dim=1440,freqN=512,channel_wise_layer_number_first=6,temp_wise_layer_number=6,channel_wise_layer_number_second=6,patch_size=128
                 , pretrain_label_class=4, finetune_label_class=2, proj_dim=1024
                 ):
        self.freqN = freqN
        self.task_name = task_name
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.output_attention = output_attention
        self.d_model = d_model
        self.embed = embed
        self.freq = freq
        self.dropout = dropout
        self.factor = factor
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_ff = d_ff
        self.activation = activation
        self.enc_in = enc_in
        self.num_channels = num_channels
        self.embedding_dim = embedding_dim
        self.shape = (seq_len, num_channels)
        self.input_window_seconds = self.seq_len / self.freqN
        self.channel_wise_layer_number_first = channel_wise_layer_number_first
        self.temp_wise_layer_number = temp_wise_layer_number
        self.channel_wise_layer_number_second = channel_wise_layer_number_second
        self.patch_size = patch_size
        self.pretrain_label_class = pretrain_label_class
        self.finetune_label_class = finetune_label_class
        self.proj_dim = proj_dim