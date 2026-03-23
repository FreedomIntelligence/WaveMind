import random
import warnings
from collections import defaultdict

import numpy as np
import torch
import torchmetrics
from torchmetrics import MetricCollection, Accuracy, AUROC, F1Score
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup

from EEG_Encoder.Model.baseModel import BIOTUnsupervisedPretrain, AttentionUnsupervisedPretrain
import lightning as L
import torch.nn as nn
import torch.nn.functional as F
from EEG_Encoder.Tools.loss import *
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_warn


class LitModel_CLIP(L.LightningModule):
    def __init__(self, EEGencoder, lr=1e-4, batch_size=32, eeg_feat_dim=768,
                 w_cls_compute=False, lambda_clip=1.0, classifier_config=None):
        super().__init__()
        self.lr = lr
        self.EEGencoder = EEGencoder
        self.batch_size = batch_size
        self.loss_func = CLIPEEGLoss_Simple_()
        self.validation_step_outputs = defaultdict(list)
        self.test_step_outputs = defaultdict(list)
        self.training_datasets = []

        # Classifier configuration
        self.w_cls_compute = w_cls_compute
        self.lambda_clip = lambda_clip  # Loss weighting parameter

        # Default classifier config
        if classifier_config is None:
            classifier_config = {
                'hidden_dim': 512,
                'dropout': 0.5,
                'use_simple_head': True,
                'label_smoothing': 0.2,
                'dataset_n_class': {
                    'TUAB': 2,
                    'TUEV': 6,
                    'SEED': 3,
                    'SEED-IV': 4,
                    'BCICIV2a': 4,
                    'FACE': 9,
                    'thingEEG': 1654,
                    'ImageNetEEG': 40
                }
            }
        self.classifier_config = classifier_config
        self.dataset_n_class = classifier_config['dataset_n_class']

        # Build classifier heads if enabled
        if self.w_cls_compute:
            self.cls_heads = self._build_classifier_heads(
                eeg_feat_dim,
                classifier_config['hidden_dim'],
                classifier_config['dropout'],
                classifier_config['use_simple_head']
            )

        self.save_hyperparameters(ignore='EEGencoder')

    def _build_classifier_heads(self, input_dim, hidden_dim, dropout, use_simple):
        """Build per-dataset classifier heads with configurable architecture"""
        heads = nn.ModuleDict()
        for dataset, n_class in self.dataset_n_class.items():
            if use_simple:
                # Simple: LayerNorm + Linear (current implementation)
                heads[dataset] = nn.Sequential(
                    nn.LayerNorm(input_dim),
                    nn.Linear(input_dim, n_class)
                )
            else:
                # MLP: LayerNorm + Linear + ReLU + Dropout + Linear
                heads[dataset] = nn.Sequential(
                    nn.LayerNorm(input_dim),
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, n_class)
                )
        return heads


    def training_step(self, batch, batch_idx):
        forward_result = self._forward(batch, batch_idx)
        loss = forward_result['loss']

        # Log total loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True,
                 sync_dist=True, batch_size=self.batch_size)

        # Log lambda parameter (once per epoch)
        if batch_idx == 0:
            self.log("lambda_clip", self.lambda_clip, on_step=False, on_epoch=True,
                     prog_bar=False, sync_dist=True)

        # Log individual loss components if classifier enabled
        if self.w_cls_compute:
            self.log("train_clip_loss", forward_result['CLIP_loss'], on_step=True,
                     on_epoch=True, prog_bar=True, sync_dist=True, batch_size=self.batch_size)
            self.log("train_cls_loss", forward_result['CLS_loss'], on_step=True,
                     on_epoch=True, prog_bar=True, sync_dist=True, batch_size=self.batch_size)

            # Log weighted contributions
            weighted_clip = self.lambda_clip * forward_result['CLIP_loss']
            weighted_cls = (1 - self.lambda_clip) * forward_result['CLS_loss']
            self.log("train_weighted_clip", weighted_clip, on_step=False, on_epoch=True,
                     prog_bar=False, sync_dist=True, batch_size=self.batch_size)
            self.log("train_weighted_cls", weighted_cls, on_step=False, on_epoch=True,
                     prog_bar=False, sync_dist=True, batch_size=self.batch_size)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.EEGencoder.parameters(), lr=self.lr)
        scheduler= get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=500,num_training_steps=self.trainer.max_steps)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }


    def _forward(self, batch, batch_idx):
        EEG = batch['eeg_data']
        label = batch['label']
        align_feature = batch['text_features']
        dataset_name = batch['dataset_name']
        
        
        # Forward pass through EEG encoder
        eeg_feature = self.EEGencoder(EEG)['pooler_output']
        
        # Prepare features for loss calculation
        align_feature = align_feature.unsqueeze(0) if align_feature.dim() == 1 else align_feature
        eeg_feature = eeg_feature.unsqueeze(0) if eeg_feature.dim() == 1 else eeg_feature
        
        # Comprehensive tensor validation for both EEG and text features
        has_issues, issue_info = self.detect_zero_samples(
            eeg_feature, align_feature,
            tensor_names=['eeg_feature', 'align_feature'],
            raise_error=True,
            verbose=False,
            check_nan_inf=True
        )
        if has_issues:
            rank_zero_warn(f"[CRITICAL] Tensor validation failed! Batch index: {batch_idx}")
            for name, info in issue_info.items():
                if info['has_issues']:
                    details = info['details']
                    if details['zero_vectors']:
                        rank_zero_warn(f"{name} zero vectors: {len(details['zero_indices'])} samples")
                    if details['has_nan']:
                        rank_zero_warn(f"{name} NaN values: {len(details['nan_indices'])} samples")
                    if details['has_inf']:
                        rank_zero_warn(f"{name} Inf values: {len(details['inf_indices'])} samples")
            raise RuntimeError("Tensor validation failed - training aborted for safety.")
        
        # Calculate CLIP loss
        loss_other = self.loss_func(
            eeg_features=eeg_feature,
            alignment_features=align_feature,
            logit_scale=self.EEGencoder.image_logit_scale
        )
        CLIP_loss = loss_other
        
        # Calculate classification loss if needed
        cls_loss = torch.tensor(0.0, device=EEG.device)
        if self.w_cls_compute:
            unique_datasets = list({name.split('_')[0] for name in dataset_name})
            
            for ds in unique_datasets:
                mask = torch.tensor([name.split('_')[0] == ds for name in dataset_name], device=EEG.device).bool()
                if mask.sum() == 0 or ds not in self.cls_heads:
                    warnings.warn(f"No classifier for dataset {ds}")
                    continue
                classifier = self.cls_heads[ds]
                logits = classifier(eeg_feature[mask])
                labels = label[mask].long()
                cls_loss += F.cross_entropy(logits, labels,
                                           label_smoothing=self.classifier_config['label_smoothing'])

        # Apply lambda weighting to combine losses
        # L = lambda_clip * L_CLIP + (1 - lambda_clip) * L_cls
        total_loss = self.lambda_clip * CLIP_loss + (1 - self.lambda_clip) * cls_loss
        
        return {
            "loss": total_loss,
            'CLIP_loss': CLIP_loss,
            'CLS_loss': cls_loss,
            "eeg_features": eeg_feature,
            "labels": label,
            "dataset_name": dataset_name
        }

    def _log_metrics(self, prefix, forward_res):
        self.log(f"{prefix}_loss", forward_res['loss'],
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 sync_dist=True,
                 batch_size=self.batch_size)
        if self.w_cls_compute:
            self.log(f"{prefix}_clip_loss", forward_res['CLIP_loss'],sync_dist=True,batch_size=self.batch_size)
            self.log(f"{prefix}_cls_loss", forward_res['CLS_loss'],batch_size=self.batch_size,sync_dist=True)
            # self.log(f"{prefix}_added_loss", forward_res['added_loss'],batch_size=self.batch_size,sync_dist=True)

    def _collect_outputs(self, outputs_dict, forward_res):
        dataset_names = forward_res["dataset_name"]
        eeg_features = forward_res["eeg_features"]
        labels = forward_res["labels"]

        dataset_indices = defaultdict(list)
        for idx, name in enumerate(dataset_names):
            name=name.split('_')[0]
            dataset_indices[name].append(idx)

        for name, indices in dataset_indices.items():
            batch_eeg = eeg_features[indices]
            batch_labels = labels[indices]

            outputs_dict[name].append({
                "eeg_features": batch_eeg.detach().cpu(),
                "labels": batch_labels.detach().cpu()
            })
        return forward_res['loss']

    def validation_step(self, batch, batch_idx):
        val_forward_res = self._forward(batch, batch_idx)
        self._log_metrics("val", val_forward_res)
        return self._collect_outputs(self.validation_step_outputs, val_forward_res)

    def test_step(self, batch, batch_idx):
        test_forward_res = self._forward(batch, batch_idx)
        self._log_metrics("test", test_forward_res)
        return self._collect_outputs(self.test_step_outputs, test_forward_res)

    def on_train_epoch_end(self):
        rank_zero_info(f"Current epoch: {self.current_epoch}, Train Loss: {self.trainer.callback_metrics['train_loss']}")

    def _epoch_start(self):
        self.training_datasets=self.trainer.datamodule.dataset_names
        self.feature_all_tests = self.trainer.datamodule.feature_all_test
        assert self.feature_all_tests is not None
        for ds_name in self.feature_all_tests.keys():
            self.feature_all_tests[ds_name][0]=self.feature_all_tests[ds_name][0]

    def on_validation_epoch_start(self):
        self._epoch_start()

    def on_test_epoch_start(self):
        self._epoch_start()

    def _epoch_end(self, outputs_dict, prefix):
        # First, concatenate local outputs from this rank
        concatenated_outputs = {}
        for dataset_name, batch_records in outputs_dict.items():
            all_eeg = torch.cat([record["eeg_features"] for record in batch_records], dim=0)
            all_labels = torch.cat([record["labels"] for record in batch_records], dim=0)
            concatenated_outputs[dataset_name] = {
                "eeg_features": all_eeg,
                "labels": all_labels
            }

        # CRITICAL FIX: Gather outputs from all ranks in DDP mode
        # Previously each rank computed metrics only on its local data shard
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()

            # Gather data from all ranks for each dataset
            gathered_outputs = {}
            for dataset_name, data in concatenated_outputs.items():
                local_eeg = data["eeg_features"].cuda()
                local_labels = data["labels"].cuda()

                # All-gather EEG features from all ranks
                gathered_eeg_list = [torch.zeros_like(local_eeg) for _ in range(world_size)]
                torch.distributed.all_gather(gathered_eeg_list, local_eeg)
                all_eeg_gathered = torch.cat(gathered_eeg_list, dim=0).cpu()

                # All-gather labels from all ranks
                gathered_labels_list = [torch.zeros_like(local_labels) for _ in range(world_size)]
                torch.distributed.all_gather(gathered_labels_list, local_labels)
                all_labels_gathered = torch.cat(gathered_labels_list, dim=0).cpu()

                gathered_outputs[dataset_name] = {
                    "eeg_features": all_eeg_gathered,
                    "labels": all_labels_gathered
                }

            # Replace local outputs with gathered outputs
            concatenated_outputs = gathered_outputs
            rank_zero_info(f"[Rank{rank}] Gathered {prefix} data from all {world_size} ranks")

        outputs_dict.clear()
        metrics = {}

       
        for ds_name in tqdm(self.training_datasets, desc=f"Computing {prefix} metrics"):
            # Skip if dataset not present in outputs (e.g., due to sampling strategy)
            if ds_name not in concatenated_outputs:
                rank_zero_info(f"Skipping dataset {ds_name} as it was not present in the {prefix} step outputs")
                continue

            all_eeg_features = concatenated_outputs[ds_name]["eeg_features"].to(self.device)
            all_labels = concatenated_outputs[ds_name]["labels"].to(self.device)
            groud_truth_tensor=self.feature_all_tests[ds_name][0].to(self.device)
            k_value=self.feature_all_tests[ds_name][1]

            # Ensure k_value is a scalar integer
            if isinstance(k_value, torch.Tensor):
                k_value = k_value.item()  # Convert tensor to scalar

            if ds_name=='thingEEG' or ds_name=='ImageNetEEG':
                k_values = [2, 4, 10, 50, 100, 200] if ds_name=='thingEEG' else [2, 4, 10, 40]
                for k in k_values:
                    acc, top5_acc,standard_metrics = self._compute_accuracy(all_eeg_features, all_labels, k, groud_truth_tensor,use_balanced_acc=True,prefix=f"{ds_name}_{prefix}")
                    metrics[f"{ds_name}_{prefix}_acc_k{k}"] = acc
                    if k in [50, 100, 200]:
                        metrics[f"{ds_name}_{prefix}_top5_acc_k{k}"] = top5_acc
            else:
                acc, _,standard_metrics = self._compute_accuracy(all_eeg_features, all_labels, k_value, groud_truth_tensor,use_balanced_acc=True,prefix=f"{ds_name}_{prefix}")
                metrics[f"{ds_name}_{prefix}_acc_k{k_value}"] = acc



            metrics.update(standard_metrics)

            # Add classifier-based metrics if classifier is enabled
            if self.w_cls_compute:
                cls_metrics = self._compute_classifier_accuracy(
                    all_eeg_features, all_labels, ds_name, prefix
                )
                metrics.update(cls_metrics)

            groud_truth_tensor.to('cpu').detach()
            all_eeg_features.to('cpu').detach()
            all_labels.to('cpu').detach()

        self.log_dict(metrics, sync_dist=True,on_epoch=True,prog_bar=True)

        if self.trainer.is_global_zero:
            rank_zero_info(f"\n===== {prefix.capitalize()} Metrics =====")
            for key, value in metrics.items():
                rank_zero_info(f"{key}: {value:.4f}")
            rank_zero_info("=============================")

        outputs_dict.clear()
        del concatenated_outputs
        torch.cuda.empty_cache()

    def on_validation_epoch_end(self):
        self._epoch_end(self.validation_step_outputs, "val")

    def on_test_epoch_end(self):
        self._epoch_end(self.test_step_outputs, "test")
        
  

    def _compute_accuracy(self, eeg_features, labels, k, ground_truth_tensor,
                          use_weight_acc=False, use_balanced_acc=False,prefix=None):
        # Parameter validation
        if use_weight_acc and use_balanced_acc:
            raise ValueError("use_weight_acc and use_balanced_acc cannot be True simultaneously.")

        # Ensure k is a scalar integer
        if isinstance(k, torch.Tensor):
            k = k.item()  # Convert tensor to scalar

        num_classes = len(ground_truth_tensor)
        calc_metrics = (k == num_classes)  # Flag to calculate additional metrics
        is_binary = (num_classes == 2)  # Check if binary classification
        
        total = 0
        correct = 0.0
        top5_correct = 0.0
        all_logits = []  # For storing logits when k == num_classes
        all_labels = []  # For storing labels when k == num_classes

        if use_balanced_acc:
            from collections import defaultdict
            class_correct = defaultdict(lambda: [0, 0])  # [top1_correct, top5_correct]
            class_total = defaultdict(int)
        else:
            class_weights = None

        if use_weight_acc:
            labels_np = labels.cpu().numpy()
            unique_classes, class_counts = np.unique(labels_np, return_counts=True)
            total_samples = len(labels_np)
            class_weights = {cls: (total_samples / (num_classes * count)) 
                            for cls, count in zip(unique_classes, class_counts)}
        else:
            class_weights = None

        for i in tqdm(range(len(eeg_features)), disable=not self.trainer.is_global_zero,
                      desc=f"{prefix}_(k={k})"):
            eeg_feature = eeg_features[i]
            label = labels[i].item()
            assert label in range(len(ground_truth_tensor))

            # Randomly select k-1 negative classes + true class
            possible_classes = list(set(range(len(ground_truth_tensor))) - {label})
            selected_classes = torch.tensor(
                random.sample(possible_classes, k - 1) + [label],
                device=self.device
            )

            alignment_features = ground_truth_tensor[selected_classes].type(torch.float32)
            eeg_feature = eeg_feature.unsqueeze(0).type(torch.float32)

            # Calculate similarity logits
            logits = self.loss_func.calculate_logits(
                eeg_features=eeg_feature.to(self.device),
                alignment_features=alignment_features.to(self.device),
                logit_scale=self.EEGencoder.image_logit_scale
            ).squeeze()

            # Collect logits and labels for additional metrics when k == num_classes
            if calc_metrics:
                # Create full logits tensor with original class order
                full_logits = torch.full((num_classes,), -1e9, device=self.device, dtype=torch.float32)
                full_logits[selected_classes] = logits
                all_logits.append(full_logits.detach().cpu())
                all_labels.append(label)

            # Top1 prediction
            pred = selected_classes[logits.argmax()]
            weight = class_weights[label] if use_weight_acc else 1.0

            if use_balanced_acc:
                class_total[label] += 1

                # Track top1 correct
                if pred == label:
                    class_correct[label][0] += 1

                # Track top5 correct if applicable
                if k >= 5:
                    top5_values, top5_indices = logits.topk(5)
                    top5_preds = selected_classes[top5_indices]
                    if (top5_preds == label).any():
                        class_correct[label][1] += 1

            else:
                if pred == label:
                    correct += weight

                if k >= 5:
                    top5_values, top5_indices = logits.topk(5)
                    top5_preds = selected_classes[top5_indices]
                    if (top5_preds == label).any():
                        top5_correct += weight

            total += 1

        # Calculate metrics when k equals number of classes
        metrics_dict = None
        if calc_metrics and len(all_logits) > 0:
            all_logits = torch.stack(all_logits)
            all_labels = torch.tensor(all_labels, dtype=torch.long)
            
            # Calculate standard classification metrics
            try:
                if is_binary:
                    # For binary classification, use specialized functions
                    # Convert to probability scores for positive class
                    prob_scores = torch.softmax(all_logits, dim=1)[:, 1]
                    
                    # AUC ROC
                    auroc = torchmetrics.functional.classification.binary_auroc(
                        prob_scores, all_labels
                    )
                    
                    # F1 score
                    f1 = torchmetrics.functional.classification.binary_f1_score(
                        prob_scores, all_labels
                    )
                    
                    # Sensitivity (recall)
                    sensitivity = torchmetrics.functional.classification.binary_recall(
                        prob_scores, all_labels
                    )
                    
                    # Specificity
                    specificity = torchmetrics.functional.classification.binary_specificity(
                        prob_scores, all_labels
                    )
                else:
                    # For multiclass classification
                    # AUC ROC (macro averaged)
                    auroc = torchmetrics.functional.classification.multiclass_auroc(
                        all_logits, all_labels, num_classes=num_classes, average='macro'
                    )
                    
                    # F1 score (macro averaged)
                    f1 = torchmetrics.functional.classification.multiclass_f1_score(
                        all_logits, all_labels, num_classes=num_classes, average='macro'
                    )
                    
                    # Sensitivity (recall) (macro averaged)
                    sensitivity = torchmetrics.functional.classification.multiclass_recall(
                        all_logits, all_labels, num_classes=num_classes, average='macro'
                    )
                    
                    # Specificity (macro averaged)
                    specificity = torchmetrics.functional.classification.multiclass_specificity(
                        all_logits, all_labels, num_classes=num_classes, average='macro'
                    )
                
                metrics_dict = {
                    f'{prefix if prefix else ""}_auroc': auroc.item(),
                    f'{prefix if prefix else ""}_f1': f1.item(),
                    f'{prefix if prefix else ""}_sensitivity': sensitivity.item(),
                    f'{prefix if prefix else ""}_specificity': specificity.item()
                }
            except Exception as e:
                print(f"Error calculating metrics: {e}")
                metrics_dict = None

        if use_balanced_acc:
            # Calculate balanced accuracy for top1 and top5
            per_class_top1 = []
            per_class_top5 = []

            # Ensure all classes with samples are properly initialized in class_correct
            all_classes = set(class_total.keys())
            for cls in all_classes:
                if cls not in class_correct:
                    # Initialize with zeros if class has samples but no correct predictions
                    class_correct[cls] = [0, 0]  # [top1_correct, top5_correct]
            
            for cls in all_classes:
                if class_total[cls] > 0:
                    top1_acc = class_correct[cls][0] / class_total[cls]
                    per_class_top1.append(top1_acc)
                    
                    if k >= 5:
                        top5_acc = class_correct[cls][1] / class_total[cls]
                        per_class_top5.append(top5_acc)

            balanced_top1 = sum(per_class_top1) / len(per_class_top1) if per_class_top1 else 0
            balanced_top5 = sum(per_class_top5) / len(per_class_top5) if per_class_top5 else 0
            
            return balanced_top1, balanced_top5, metrics_dict

        else:
            accuracy = correct / total if total > 0 else 0
            top5_accuracy = top5_correct / total if k >= 5 and total > 0 else 0
            return accuracy, top5_accuracy, metrics_dict

    def _compute_classifier_accuracy(self, eeg_features, labels, dataset_name, prefix):
        """
        Compute classifier-based accuracy using learned classifier heads

        This differs from k-way retrieval accuracy:
        - k-way: Uses cosine similarity to ground truth features
        - Classifier: Uses learned linear/MLP classifier head

        Args:
            eeg_features: EEG embeddings (N, 768)
            labels: Ground truth labels (N,)
            dataset_name: Dataset identifier (e.g., 'TUEV_test')
            prefix: Logging prefix (e.g., 'val', 'test')

        Returns:
            dict: Metrics including accuracy, AUROC, F1
        """
        if not self.w_cls_compute:
            return {}

        base_name = dataset_name.split('_')[0]
        if base_name not in self.cls_heads:
            return {}

        classifier = self.cls_heads[base_name]
        with torch.no_grad():
            logits = classifier(eeg_features)
            preds = logits.argmax(dim=-1)

            # Compute accuracy
            correct = (preds == labels).sum().item()
            total = labels.size(0)
            accuracy = correct / total if total > 0 else 0.0

            # Compute AUROC and F1
            num_classes = self.dataset_n_class[base_name]
            is_binary = (num_classes == 2)

            try:
                if is_binary:
                    prob_scores = torch.softmax(logits, dim=1)[:, 1]
                    auroc = torchmetrics.functional.classification.binary_auroc(prob_scores, labels)
                    f1 = torchmetrics.functional.classification.binary_f1_score(prob_scores, labels)
                else:
                    auroc = torchmetrics.functional.classification.multiclass_auroc(
                        logits, labels, num_classes=num_classes, average='macro'
                    )
                    f1 = torchmetrics.functional.classification.multiclass_f1_score(
                        logits, labels, num_classes=num_classes, average='macro'
                    )

                return {
                    f'{prefix}_cls_acc_{base_name}': accuracy,
                    f'{prefix}_cls_auroc_{base_name}': auroc.item(),
                    f'{prefix}_cls_f1_{base_name}': f1.item()
                }
            except Exception as e:
                rank_zero_warn(f"Error computing classifier metrics for {base_name}: {e}")
                return {f'{prefix}_cls_acc_{base_name}': accuracy}



    def detect_zero_samples(self, *tensors, tensor_names=None, threshold=1e-8, raise_error=True, verbose=True, check_nan_inf=True):
        """
        Comprehensive tensor validation for multiple tensors including zero vectors, NaN, and Inf detection.
        
        Args:
            *tensors: Variable number of tensors to validate
            tensor_names: List of names for tensors (optional, for better error reporting)
            threshold: Threshold for detecting zero vectors
            raise_error: Whether to raise error when issues are found
            verbose: Whether to print detailed diagnostic information
            check_nan_inf: Whether to check for NaN and Inf values
            
        Returns:
            tuple: (has_issues, comprehensive_issue_info_dict)
        """
        if tensor_names is None:
            tensor_names = [f"tensor_{i}" for i in range(len(tensors))]
        
        if len(tensors) != len(tensor_names):
            raise ValueError(f"Number of tensors ({len(tensors)}) must match number of names ({len(tensor_names)})")
        
        comprehensive_issue_info = {}
        has_overall_issues = False
        
        for i, (tensor, name) in enumerate(zip(tensors, tensor_names)):
            issue_info = {
                'zero_vectors': False,
                'has_nan': False,
                'has_inf': False,
                'zero_indices': [],
                'nan_indices': [],
                'inf_indices': []
            }
            
            # Check for NaN and Inf values first (most critical)
            if check_nan_inf:
                nan_mask = torch.isnan(tensor)
                inf_mask = torch.isinf(tensor)
                
                if nan_mask.any():
                    issue_info['has_nan'] = True
                    issue_info['nan_indices'] = torch.where(nan_mask.any(dim=-1))[0].tolist()
                    
                if inf_mask.any():
                    issue_info['has_inf'] = True
                    issue_info['inf_indices'] = torch.where(inf_mask.any(dim=-1))[0].tolist()
            
            # Check for zero vectors using efficient norm calculation
            norms = tensor.norm(dim=-1)  # Compute norms once for efficiency
            zero_mask = norms < threshold
            
            if zero_mask.any():
                issue_info['zero_vectors'] = True
                issue_info['zero_indices'] = torch.where(zero_mask)[0].tolist()
            
            # Determine if any issues were found for this tensor
            tensor_has_issues = (issue_info['has_nan'] or issue_info['has_inf'] or issue_info['zero_vectors'])
            has_overall_issues = has_overall_issues or tensor_has_issues
            
            comprehensive_issue_info[name] = {
                'has_issues': tensor_has_issues,
                'details': issue_info
            }
            
            # Print detailed diagnostic information if requested
            if verbose and tensor_has_issues:
                print(f"[VALIDATION] Detected issues in {name}:")
                
                if issue_info['has_nan']:
                    print(f"  - NaN values: {len(issue_info['nan_indices'])} samples")
                    print(f"    Indices: {issue_info['nan_indices']}")
                    
                if issue_info['has_inf']:
                    print(f"  - Inf values: {len(issue_info['inf_indices'])} samples")
                    print(f"    Indices: {issue_info['inf_indices']}")
                    
                if issue_info['zero_vectors']:
                    print(f"  - Zero vectors: {len(issue_info['zero_indices'])} samples")
                    print(f"    Indices: {issue_info['zero_indices']}")
                    print(f"    Sample norms: {norms[issue_info['zero_indices']].cpu().numpy()}")
        
        # Raise comprehensive error if requested and issues found
        if raise_error and has_overall_issues:
            error_parts = []
            for name, info in comprehensive_issue_info.items():
                if info['has_issues']:
                    details = info['details']
                    tensor_errors = []
                    if details['has_nan']:
                        tensor_errors.append(f"{len(details['nan_indices'])} samples with NaN")
                    if details['has_inf']:
                        tensor_errors.append(f"{len(details['inf_indices'])} samples with Inf")
                    if details['zero_vectors']:
                        tensor_errors.append(f"{len(details['zero_indices'])} zero vectors")
                    
                    if tensor_errors:
                        error_parts.append(f"{name}: {', '.join(tensor_errors)}")
            
            error_msg = f"Tensor validation failed - {', '.join(error_parts)}"
            raise RuntimeError(error_msg)
        
        return has_overall_issues, comprehensive_issue_info
   



class LitModel_Classifier(L.LightningModule):
    def __init__(self, EEGencoder, n_class, lr=1e-4, model_name="model",verbose=False,use_subject_layers=False,batch_size=32,thing_test_class_indices=None):
        super().__init__()
        self.lr = lr
        self.EEGencoder = EEGencoder
        self.n_class = n_class
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, n_class)
        )
        self.save_hyperparameters(ignore="EEGencoder")
        self.model_name = model_name
        self.verbose = verbose
        self.use_subject_layers = use_subject_layers
        self.batch_size = batch_size
        self.thing_test_indicators = thing_test_class_indices

        if self.thing_test_indicators:
            n_test_classes = len(thing_test_class_indices)
        else:
            n_test_classes = n_class

        # Metrics collection for validation
        self.val_metrics = MetricCollection({
            "accuracy": Accuracy(task="multiclass", average='micro',num_classes=n_test_classes),
            "auroc": AUROC(num_classes=n_test_classes, task="multiclass"),
            "f1": F1Score(num_classes=n_test_classes, average='micro',task="multiclass"),
        }, prefix="val_")
        self.test_metrics = MetricCollection({
            "accuracy": Accuracy(task="multiclass", average='micro',num_classes=n_test_classes),
            "auroc": AUROC(num_classes=n_test_classes, task="multiclass"),
            "f1": F1Score(num_classes=n_test_classes, average='micro',task="multiclass"),
        }, prefix="test_")
        self.train_metrics = MetricCollection({
            "accuracy": Accuracy(task="multiclass", average='micro',num_classes=n_class),
            "auroc": AUROC(num_classes=n_class, task="multiclass"),
            "f1": F1Score(num_classes=n_class, average='micro',task="multiclass"),
        }, prefix="train_")

    def training_step(self, batch, batch_idx):
        logits, label = self._forward_model(batch)

        loss = F.cross_entropy(logits, label,label_smoothing=0.2)

        probs = logits.softmax(dim=-1)
        self.train_metrics.update(probs, label)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True,batch_size=self.batch_size)
        return loss

    def _forward_model(self, batch):
        if isinstance(batch, dict):
            EEG, label, text, text_feature, img_path, img_feature = batch['eeg_data'], batch['label'], batch['text'], batch[
                'text_features'], batch['img_path'], batch['img_features']
        else:
            EEG, label = batch
        assert EEG.size(1) == 32 and EEG.size(2) == 512, f"EEG shape is {EEG.size()}, expected (B, 32, 512)"

        assert not torch.isnan(EEG).any(), "EEG contains NaN"
        assert not torch.isinf(EEG).any(), "EEG contains inf"
        assert (label >= 0).all() and (label < self.n_class).all(), "label out of class range"

        logits = self.forward(EEG)
        label = label.long()

        assert not torch.isnan(logits).any(), "logits contains NaN"
        assert not torch.isinf(logits).any(), "logits contains inf"

        return logits,label
    
    def forward(self, batch):
        EEG=batch
        EEG=EEG.to(torch.float32)
        self.EEGencoder = self.EEGencoder.to(torch.float32)
        eeg_feature = self.EEGencoder(EEG)['pooler_output']
        eeg_feature = F.normalize(eeg_feature, dim=-1)
        logits = self.classifier(eeg_feature)
        return logits


    def validation_step(self, batch, batch_idx):
        logits,label = self._forward_model(batch)

        if self.thing_test_indicators is not None:
            logits = logits[:, self.thing_test_indicators]

        loss = F.cross_entropy(logits, label, label_smoothing=0.2)

        # Update metrics
        probs = logits.softmax(dim=-1)


        self.val_metrics.update(probs, label)


        # Log loss
        self.log("val_loss", loss, on_step=True, prog_bar=True, sync_dist=True,batch_size=self.batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        logits, label = self._forward_model(batch)

        if self.thing_test_indicators is not None:
            logits = logits[:, self.thing_test_indicators]

        loss = F.cross_entropy(logits, label, label_smoothing=0.2)

        # Update metrics
        probs = logits.softmax(dim=-1)

        self.test_metrics.update(probs, label)

        # Log loss
        self.log("test_loss", loss, on_step=True, prog_bar=True, sync_dist=True,batch_size=self.batch_size)
        return loss

    def on_train_epoch_end(self):
        # Compute metrics
        train_metrics = self.train_metrics.compute()
        self.train_metrics.reset()
        self.log_dict(train_metrics, sync_dist=True)
        if self.verbose:
            rank_zero_info(f"\nCurrent epoch: {self.current_epoch}, Train Loss: {self.trainer.callback_metrics['train_loss']}, Train Acc: {train_metrics['train_accuracy']}")

    def on_validation_epoch_end(self):
        # Compute metrics
        val_metrics = self.val_metrics.compute()
        self.val_metrics.reset()
        self.log_dict(val_metrics, sync_dist=True)
        if self.verbose:
            rank_zero_info(f"\nCurrent epoch: {self.current_epoch}, Val Loss: {self.trainer.callback_metrics['val_loss']}, Val Acc: {val_metrics['val_accuracy']}")

    def on_test_epoch_end(self):
        # Compute metrics
        test_metrics = self.test_metrics.compute()
        self.test_metrics.reset()
        self.log_dict(test_metrics, sync_dist=True)


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [{"params": self.EEGencoder.parameters()}, {"params": self.classifier.parameters()}],
            lr=self.lr
        )
        total_steps = self.trainer.estimated_stepping_batches
        # scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=20, num_training_steps=total_steps)
        scheduler=get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=20, num_training_steps=total_steps,num_cycles=10)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }





class LitModel_Pretrain(L.LightningModule):
    def __init__(self, lr=1e-4, emb_size=1024, heads=8, depth=4, n_channels=32,method="atten",attenEncoder=None,atten_mask_ratio=0.6):
        super().__init__()
        self.lr = lr
        if method == "BIOT":
            self.model = BIOTUnsupervisedPretrain(emb_size=emb_size, heads=heads, depth=depth, n_channels=n_channels)
        elif method == "atten":
            self.model = AttentionUnsupervisedPretrain(encoder=attenEncoder, emb_size=emb_size,mask_ratio=atten_mask_ratio)
        else:
            raise ValueError(f"Unknown method: {method}")

    def training_step(self, batch, batch_idx):
        EEG, true_label = batch
        _,loss,loss_log=self.model(EEG)
        self.log("train_loss", loss,sync_dist=True, on_step=True, on_epoch=True)
        for key in loss_log:
            self.log("train_"+key, loss_log[key], sync_dist=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        EEG, true_label = batch
        _, loss, loss_log = self.model(EEG)
        self.log("val_loss", loss, sync_dist=True, on_step=True, on_epoch=True)
        for key in loss_log:
            self.log("val_" + key, loss_log[key], sync_dist=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=self.trainer.max_steps)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }
