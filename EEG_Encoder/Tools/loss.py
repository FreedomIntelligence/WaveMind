
import logging

import numpy as np
import torch
import sys
import torch.distributed.nn
from einops import rearrange
from sympy.physics.vector.tests.test_printing import alpha
from torch import distributed as dist, nn as nn
from torch.nn import functional as F

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_warn



class CLIPEEGLoss_Simple_(nn.Module):
    """CLIP-style contrastive loss with built-in feature normalization.

    Features:
    - Automatic L2 normalization of input features
    - Logit scaling with safety clamping
    - Symmetric contrastive loss
    - Numerical stability checks
    """

    def __init__(self, logit_scale_max=100.0):
        super().__init__()
        self.logit_scale_max = logit_scale_max

    def forward(self, eeg_features, alignment_features, logit_scale):
        """
        Compute contrastive loss with automatic feature normalization.

        Args:
            eeg_features: (N, D) raw EEG embeddings
            alignment_features: (N, D) raw alignment embeddings
            logit_scale: log-domain scale parameter (nn.Parameter)

        Returns:
            Scalar loss value
        """
        eeg_features =eeg_features if eeg_features.dim() == 2 else eeg_features.squeeze()
        alignment_features =alignment_features if alignment_features.dim() == 2 else alignment_features.squeeze()

        self.validate_inputs(eeg_features, alignment_features, logit_scale)
        safe_scale = self.clamp_logit_scale(logit_scale)
        return self.contrastive_loss(eeg_features, alignment_features, safe_scale)

    def calculate_logits(self, eeg_features, alignment_features, logit_scale):
        """
        Compute normalized similarity matrix for inference.

        Args:
            Same as forward()

        Returns:
            (N, N) similarity matrix
        """

        eeg_features=eeg_features if eeg_features.dim() == 2 else eeg_features.squeeze()
        alignment_features=alignment_features if alignment_features.dim() == 2 else alignment_features.squeeze()


        self.validate_inputs(eeg_features, alignment_features, logit_scale)
        safe_scale = self.clamp_logit_scale(logit_scale)
        eeg_norm = F.normalize(eeg_features, p=2, dim=-1)
        align_norm = F.normalize(alignment_features, p=2, dim=-1)

        # make same type
        safe_scale = safe_scale.to(eeg_norm.dtype)
        align_norm=align_norm.to(eeg_norm.dtype)

        return safe_scale * eeg_norm @ align_norm.t()

    def contrastive_loss(self, query, key, logit_scale):
        """Compute symmetric loss with numerical stability."""
        # Normalize features
        query_norm = F.normalize(query, p=2, dim=-1)
        key_norm = F.normalize(key, p=2, dim=-1)

        # make same type
        logit_scale = logit_scale.to(query_norm.dtype)
        key_norm = key_norm.to(query_norm.dtype)

        # Calculate logits
        logits = logit_scale * query_norm @ key_norm.t()
        labels = torch.arange(len(query), device=query.device)

        # Stable cross-entropy
        loss_q = F.cross_entropy(logits, labels)
        loss_k = F.cross_entropy(logits.t(), labels)
        return (loss_q + loss_k) / 2

    def clamp_logit_scale(self, logit_scale):
        """Prevent logit scale explosion."""
        return logit_scale.exp().clamp(max=self.logit_scale_max)

    def validate_inputs(self, eeg, alignment, logit_scale):
        if eeg.dim() != 2 or alignment.dim() != 2:
            raise ValueError(f"Features must be 2D. Got shapes {eeg.shape} and {alignment.shape}")

        if eeg.size(-1) != alignment.size(-1):
            raise ValueError(f"Feature dim mismatch: {eeg.size(-1)} vs {alignment.size(-1)}")

        if torch.isnan(eeg).any() or torch.isinf(eeg).any():
            raise ValueError("NaN or Inf values detected in EEG features")
        if torch.isnan(alignment).any() or torch.isinf(alignment).any():
            raise ValueError("NaN or Inf values detected in alignment features")

        if logit_scale is None:
            raise ValueError("logit_scale must be provided as nn.Parameter")



class PretrainLoss(nn.Module):
    def __init__(self, method='temp_mlp',mse_loss=True):
        super(PretrainLoss, self).__init__()
        rank_zero_info("Using method: ", method)
        self.loss_fn = nn.MSELoss(reduction='mean') if mse_loss else nn.SmoothL1Loss(reduction='mean')
        self.cls_loss = nn.CrossEntropyLoss()
        self.method = method

    def _get_stft(self,x,dims):
        n_fft = 64
        hop_length = n_fft // 4
        win_length = n_fft
        window = torch.hann_window(win_length).to(x.device)
        stft_results = []
        if dims==3:
            for channel in x:
                stft_result = torch.stft(channel, n_fft=n_fft, hop_length=hop_length,
                                         win_length=win_length, window=window, return_complex=True)
                stft_results.append(torch.abs(stft_result))
            stft_tensor = torch.stack(stft_results)
        elif dims==2:
            stft_result = torch.stft(x, n_fft=n_fft, hop_length=hop_length,
                                     win_length=win_length, window=window, return_complex=True)
            stft_tensor = torch.abs(stft_result)
        else:
            raise ValueError("dims should be 2 or 3")
        return stft_tensor

    def calculate_rec_loss(self, rec, target):
        if len(target.shape) == 4:
            target = rearrange(target, 'b n a c -> b n (a c)')
        rec_loss = self.loss_fn(rec, target)
        return rec_loss


    def std_norm(self, x):
        mean = torch.mean(x, dim=tuple(range(1, x.dim())), keepdim=True)
        std = torch.std(x, dim=tuple(range(1, x.dim())), keepdim=True)
        eps = 1e-8
        x = (x - mean) / (std + eps)
        return x

    def calculate_label_loss(self, pred_label, true_label):
        if pred_label is not None and true_label is not None:
            assert pred_label.dtype == torch.float32 or pred_label.dtype == torch.float64
            assert true_label.dtype == torch.long
            cls_loss = self.cls_loss(pred_label, true_label)
        else:
            cls_loss = torch.tensor(0.0)
        return cls_loss


    def total_loss_rebuild_all(self, x, recs_x, true_label=None,pred_label=None):
        rec_stft,rec_amp,rec_tmp,rec_angle=recs_x
        true_stft=self._get_stft(x,dims=3)
        true_fft = torch.fft.fft(x, dim=-1)
        true_amplitude = torch.abs(true_fft)
        true_angle = torch.angle(true_fft)

        rec_loss_stft = self.calculate_rec_loss(rec_stft, true_stft).to(x.device)
        rec_loss_amp = self.calculate_rec_loss(rec_amp, true_amplitude).to(x.device)
        rec_loss_tmp = self.calculate_rec_loss(rec_tmp, x).to(x.device)
        rec_loss_angle = self.calculate_rec_loss(rec_angle, true_angle).to(x.device)
        cls_loss = self.calculate_label_loss(pred_label, true_label).to(x.device)
        return self.sum_up_loss(rec_loss_stft, rec_loss_amp, rec_loss_tmp, rec_loss_angle, cls_loss)

    def total_loss_rebuild_except_temporal(self, x, recs_x, true_label=None, pred_label=None):
        rec_stft, rec_amp, _, rec_angle = recs_x
        true_stft = self._get_stft(x,dims=3)
        true_fft = torch.fft.fft(x, dim=-1)
        true_amplitude = self.torch.abs(true_fft)
        true_angle = self.torch.angle(true_fft)


        rec_tmp = torch.fft.ifft(rec_amp * torch.exp(1j * rec_angle), dim=-1).real
        rec_tmp_1=torch.fft.ifft(true_amplitude * torch.exp(1j * true_angle), dim=-1).real


        rec_loss_stft = self.calculate_rec_loss(rec_stft, true_stft)
        rec_loss_amp = self.calculate_rec_loss(rec_amp, true_amplitude)
        rec_loss_angle = self.calculate_rec_loss(rec_angle, true_angle)


        cls_loss = self.calculate_label_loss(pred_label, true_label)

        rec_loss_tmp = self.calculate_rec_loss(rec_tmp_1, x)

        return self.sum_up_loss(rec_loss_stft, rec_loss_amp, rec_loss_tmp, rec_loss_angle, cls_loss)

    def total_loss_rebuild_except_temporal_mask(self, x, recs_x, true_label=None, pred_label=None,mask_tokens_id=None):
        def _extract_mask_tokens(recs_x, mask_tokens_id):
            if len(recs_x[0].shape) == 4:
                recs_x[0] = recs_x[0].view(recs_x[0].size(0), recs_x[0].size(1), -1)
            rec_stft_mask = recs_x[0][mask_tokens_id].view(mask_tokens_id.size(0), -1, recs_x[0].size(-1))
            rec_amp_mask = recs_x[1][mask_tokens_id].view(mask_tokens_id.size(0), -1, recs_x[1].size(-1))
            rec_angle_mask = recs_x[3][mask_tokens_id].view(mask_tokens_id.size(0), -1, recs_x[3].size(-1))
            rec_tmp_mask = recs_x[2][mask_tokens_id].view(mask_tokens_id.size(0), -1, recs_x[2].size(-1))
            return rec_stft_mask, rec_amp_mask, rec_tmp_mask,rec_angle_mask

        if mask_tokens_id is None:
            raise ValueError("mask_token is None")

        true_stft = self._get_stft(x, dims=3)
        true_fft = torch.fft.fft(x, dim=-1)
        true_amplitude = torch.abs(true_fft)
        true_angle = torch.angle(true_fft)
        rec_tmp = torch.fft.ifft(true_fft, dim=-1).real

        rec_tmp_mask = rec_tmp[mask_tokens_id].view(mask_tokens_id.size(0), -1, rec_tmp.size(-1))
        true_stft_mask, true_amplitude_mask, x_mask, true_angle_mask = _extract_mask_tokens([true_stft, true_amplitude, x, true_angle], mask_tokens_id)


        rec_stft_mask, rec_amp_mask, _, rec_angle_mask = _extract_mask_tokens(recs_x, mask_tokens_id)

        rec_loss_stft = self.calculate_rec_loss(rec_stft_mask, true_stft_mask)
        rec_loss_amp = self.calculate_rec_loss(rec_amp_mask, true_amplitude_mask)
        rec_loss_angle = self.calculate_rec_loss(rec_angle_mask, true_angle_mask)
        cls_loss = self.calculate_label_loss(pred_label, true_label)
        rec_loss_tmp = self.calculate_rec_loss(rec_tmp_mask, x_mask)


        return self.sum_up_loss(rec_loss_stft, rec_loss_amp, rec_loss_tmp, rec_loss_angle, cls_loss)


    def sum_up_loss(self, rec_loss_stft, rec_loss_amp, rec_loss_tmp, rec_loss_angle, cls_loss):
        rec_loss_tmp = rec_loss_tmp
        rec_loss_angle=0
        rec_loss_stft=0

        total_loss = 0.1* rec_loss_stft + rec_loss_amp + rec_loss_tmp + rec_loss_angle + cls_loss
        log = {}
        log["rec_loss_stft"] = rec_loss_stft
        log["rec_loss_amp"] = rec_loss_amp
        log["rec_loss_tmp"] = rec_loss_tmp
        log["rec_loss_angle"] = rec_loss_angle
        log["cls_loss"] = cls_loss
        return total_loss, log



    def forward(self, x, recs_x, true_label=None, pred_label=None,masks_id=None):
        if self.method == 'temp_mlp':
            return self.total_loss_rebuild_all(x, recs_x, true_label, pred_label)
        elif self.method == 'temp_by_fft':
            return self.total_loss_rebuild_except_temporal(x, recs_x, true_label, pred_label)
        elif self.method == 'temp_by_fft_mask':
            return self.total_loss_rebuild_except_temporal_mask(x, recs_x, true_label, pred_label,masks_id)

