import inspect
import multiprocessing
import os
import shutil
import sys
import gc
import warnings
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from time import time, sleep
from typing import Tuple, Union, List

import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.nnunet.random_binary_operator import ApplyRandomBinaryOperatorTransform
from batchgeneratorsv2.transforms.nnunet.remove_connected_components import \
    RemoveRandomConnectedComponentFromOneHotEncodingTransform
from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import MoveSegAsOneHotToDataTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform
from torch import autocast, nn
from torch import distributed as dist
from torch._dynamo import OptimizedModule
from torch.cuda import device_count
from torch import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from nnunetv2.configuration import ANISO_THRESHOLD, default_num_processes
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnunetv2.inference.export_prediction import export_prediction_from_logits, resample_and_save
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.logging.nnunet_logger import nnUNetLogger
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.crossval_split import generate_crossval_split
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

class nnUNetTrainer(object):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        self.is_ddp = dist.is_available() and dist.is_initialized()
        self.local_rank = 0 if not self.is_ddp else dist.get_rank()
        self.device = device
        if self.is_ddp:
            print(f"I am local rank {self.local_rank}. {device_count()} GPUs are available. The world size is "
                  f"{dist.get_world_size()}."
                  f"Setting device to {self.device}")
            self.device = torch.device(type='cuda', index=self.local_rank)
        else:
            if self.device.type == 'cuda':
                self.device = torch.device(type='cuda', index=0)
            print(f"Using device: {self.device}")
        self.my_init_kwargs = {}
        for k in inspect.signature(self.__init__).parameters.keys():
            self.my_init_kwargs[k] = locals()[k]
        self.plans_manager = PlansManager(plans)
        self.configuration_manager = self.plans_manager.get_configuration(configuration)
        self.configuration_name = configuration
        self.dataset_json = dataset_json
        self.fold = fold
        self.preprocessed_dataset_folder_base = join(nnUNet_preprocessed, self.plans_manager.dataset_name) \
            if nnUNet_preprocessed is not None else None
        self.output_folder_base = join(nnUNet_results, self.plans_manager.dataset_name,
                                       self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" + configuration) \
            if nnUNet_results is not None else None
        self.output_folder = join(self.output_folder_base, f'fold_{fold}')
        self.preprocessed_dataset_folder = join(self.preprocessed_dataset_folder_base,
                                                self.configuration_manager.data_identifier)
        self.dataset_class = None
        self.is_cascaded = self.configuration_manager.previous_stage_name is not None
        self.folder_with_segs_from_previous_stage = \
            join(nnUNet_results, self.plans_manager.dataset_name,
                 self.__class__.__name__ + '__' + self.plans_manager.plans_name + "__" +
                 self.configuration_manager.previous_stage_name, 'predicted_next_stage', self.configuration_name) \
                if self.is_cascaded else None
        self.was_initialized = False
        self.initial_lr = 1e-2
        self.weight_decay = 3e-5
        self.oversample_foreground_percent = 0.33
        self.probabilistic_oversampling = False
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self.num_epochs = 500
        self.current_epoch = 0
        self.enable_deep_supervision = False
        self.label_manager = self.plans_manager.get_label_manager(dataset_json)
        self.num_input_channels = None
        self.network = None
        self.optimizer = self.lr_scheduler = None
        self.grad_scaler = GradScaler("cuda") if self.device.type == 'cuda' else None
        self.loss = None
        self.uncertainty_loss_weight = 1.0
        timestamp = datetime.now()
        maybe_mkdir_p(self.output_folder)
        self.log_file = join(self.output_folder, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
                             (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
                              timestamp.second))
        self.logger = nnUNetLogger()
        self.dataloader_train = self.dataloader_val = None
        self._best_ema = None
        self.inference_allowed_mirroring_axes = None
        self.save_every = 25
        self.disable_checkpointing = False
        self.error_stable_dict = defaultdict(lambda: None)
        self.epoch_counts = defaultdict(int)
        self.max_buffer_size = 30
        self.smoothing_alpha = 1.0 / self.max_buffer_size
        self.cancer_labels = [54]
        self.processed_val_cases = set()  # Track processed validation cases
        self.max_val_cases_to_save = 20  # Limit to first 20 validation cases

        self.print_to_log_file("\n#######################################################################\n"
                               "Please cite the following paper when using nnU-Net:\n"
                               "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
                               "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
                               "Nature methods, 18(2), 203-211.\n"
                               "#######################################################################\n",
                               also_print_to_console=True, add_timestamp=False)

    def initialize(self):
        if not self.was_initialized:
            self._set_batch_size_and_oversample()
            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)
            self.network = self.build_network_architecture(
                self.configuration_manager.network_arch_class_name,
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                self.enable_deep_supervision
            ).to(self.device)
            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)
            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])
            self.loss = self._build_loss()
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized.")

    def _do_i_compile(self):
        if self.device == torch.device('mps'):
            if 'nnUNet_compile' in os.environ.keys() and os.environ['nnUNet_compile'].lower() in ('true', '1', 't'):
                self.print_to_log_file("INFO: torch.compile disabled because of unsupported mps device")
            return False
        if self.device == torch.device('cpu'):
            if 'nnUNet_compile' in os.environ.keys() and os.environ['nnUNet_compile'].lower() in ('true', '1', 't'):
                self.print_to_log_file("INFO: torch.compile disabled because device is CPU")
            return False
        if os.name == 'nt':
            if 'nnUNet_compile' in os.environ.keys() and os.environ['nnUNet_compile'].lower() in ('true', '1', 't'):
                self.print_to_log_file("INFO: torch.compile disabled because Windows is not natively supported.")
            return False
        if 'nnUNet_compile' not in os.environ.keys():
            return True
        else:
            return os.environ['nnUNet_compile'].lower() in ('true', '1', 't')

    def _save_debug_information(self):
        if self.local_rank == 0:
            dct = {}
            for k in self.__dir__():
                if not k.startswith("__"):
                    if not callable(getattr(self, k)) or k in ['loss']:
                        dct[k] = str(getattr(self, k))
                    elif k in ['network']:
                        dct[k] = str(getattr(self, k).__class__.__name__)
                    elif k in ['dataloader_train', 'dataloader_val']:
                        if hasattr(getattr(self, k), 'generator'):
                            dct[k + '.generator'] = str(getattr(self, k).generator)
                        if hasattr(getattr(self, k), 'num_processes'):
                            dct[k + '.num_processes'] = str(getattr(self, k).num_processes)
                        if hasattr(getattr(self, k), 'transform'):
                            dct[k + '.transform'] = str(getattr(self, k).transform)
            import subprocess
            hostname = subprocess.getoutput(['hostname'])
            dct['hostname'] = hostname
            torch_version = torch.__version__
            if self.device.type == 'cuda':
                gpu_name = torch.cuda.get_device_name()
                dct['gpu_name'] = gpu_name
                cudnn_version = torch.backends.cudnn.version()
            else:
                cudnn_version = 'None'
            dct['device'] = str(self.device)
            dct['torch_version'] = torch_version
            dct['cudnn_version'] = cudnn_version
            save_json(dct, join(self.output_folder, "debug.json"))

    def build_network_architecture(self, architecture_class_name: str,
                                  arch_init_kwargs: dict,
                                  arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                  num_input_channels: int,
                                  enable_deep_supervision: bool = False) -> nn.Module:
        
        return get_network_from_plans(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            55,
            allow_init=True,
            deep_supervision=enable_deep_supervision)


    def compute_prediction_failure(self, seg_output: torch.Tensor, target: torch.Tensor, case_ids: list = None) -> torch.Tensor:
        """Compute failure map for foreground voxels (labels 1-54, indices 0-53).
        Args:
            seg_output: [B, 54, ...], segmentation logits for 54 foreground classes (indices 0-53)
            target: [B, C, ...] or [B, 1, ...], ground truth labels (one-hot with 54 channels or label map with values 0-53)
            case_ids: List of case identifiers (optional)
        Returns:
            failure: [B, 1, ...], failure map (1 where predicted foreground class != ground truth foreground class, 0 elsewhere)
        """
        with autocast(self.device.type, enabled=True):
            expected_shape = (seg_output.shape[0], target.shape[1] if target.shape[1] > 1 else 1, *seg_output.shape[2:])
            if target.shape != expected_shape:
                self.print_to_log_file(f"Shape mismatch in compute_prediction_failure: Resizing target from {target.shape} to {expected_shape}")
                target = F.interpolate(target.float(), size=seg_output.shape[2:], mode='nearest').long()
            
            probs = torch.softmax(seg_output, dim=1)  # [B, 54, ...]
            pred_class = torch.argmax(probs, dim=1, keepdim=True)  # [B, 1, ...], indices 0-53
            
            if target.shape[1] > 1:
                target_class = torch.argmax(target, dim=1, keepdim=True)
                foreground_mask = torch.any(target > 0, dim=1, keepdim=True).float()
            else:
                target_class = target - 1
                foreground_mask = (target > 0).float()
                target_class = torch.where(foreground_mask.bool(), target_class, -1)
            
            failure = ((pred_class != target_class) & (foreground_mask > 0)).float()
            
            self.print_to_log_file(f"pred_class unique: {torch.unique(pred_class).tolist()}")
            self.print_to_log_file(f"target_class unique: {torch.unique(target_class).tolist()}")
            if failure.sum() == 0 and foreground_mask.sum() > 0:
                self.print_to_log_file(f"WARNING: No failure detected in foreground voxels for case IDs: {case_ids}")
        
        del probs, pred_class, target_class, foreground_mask
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        
        return failure

    def compute_prediction_error_probability(self, seg_output: torch.Tensor, target: torch.Tensor, 
                                            case_identifiers: list) -> torch.Tensor:
        """Compute EMA-averaged normalized error probability for foreground voxels (labels 1-54, indices 0-53).
        Args:
            seg_output: [B, 54, ...], segmentation logits for 54 foreground classes (indices 0-53)
            target: [B, C, ...] or [B, 1, ...], ground truth labels (one-hot with 54 channels or label map with values 0-53)
            case_identifiers: List of image IDs for the batch
        Returns:
            error_stable: [B, 1, ...], normalized log-transformed failure probability per voxel, 0 for non-foreground
        """
        with autocast(self.device.type, enabled=True):
            expected_shape = (seg_output.shape[0], target.shape[1] if target.shape[1] > 1 else 1, *seg_output.shape[2:])
            if target.shape != expected_shape:
                self.print_to_log_file(f"Shape mismatch in compute_prediction_error_probability: Resizing target from {target.shape} to {expected_shape}")
                target = F.interpolate(target.float(), size=seg_output.shape[2:], mode='nearest').long()
            
            failure = self.compute_prediction_failure(seg_output, target, case_identifiers)
            foreground_mask, total_foreground_voxels = self.get_image_mask(target, seg_output)
            
            if total_foreground_voxels == 0:
                self.print_to_log_file(f"WARNING: No foreground voxels (labels 1-54) found in batch for case IDs: {case_identifiers}")
                error_stable = torch.zeros(target.shape[0], 1, *target.shape[2:], dtype=torch.float32, device=target.device)
                del failure, foreground_mask, total_foreground_voxels
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                gc.collect()
                return error_stable
            
            self.print_to_log_file(f"Total foreground voxels: {total_foreground_voxels.item():.1f}")
            failure.mul_(foreground_mask)
            self.print_to_log_file(f"failure mean (foreground only): {failure[foreground_mask > 0].mean().item()}")
            
            error_stable = torch.zeros_like(failure, dtype=torch.float32)
            failure_spatial = F.avg_pool3d(failure, kernel_size=3, stride=1, padding=1)
            
            for b in range(failure.shape[0]):
                case_id = case_identifiers[b]
                self.epoch_counts[case_id] = self.epoch_counts.get(case_id, 0) + 1
                error_key = f"error_{case_id}"
                if error_key not in self.error_stable_dict:
                    self.error_stable_dict[error_key] = failure_spatial[b].clone().half()
                else:
                    stored_error = self.error_stable_dict[error_key]
                    if stored_error.shape != failure_spatial.shape[1:]:
                        stored_error = F.interpolate(stored_error[None, ...], size=failure_spatial.shape[1:], 
                                                    mode='trilinear', align_corners=False)[0]
                        self.error_stable_dict[error_key] = stored_error.half()
                    alpha = min(1.0 / self.epoch_counts[case_id], self.smoothing_alpha)
                    self.error_stable_dict[error_key].mul_(1 - alpha).add_(alpha * failure_spatial[b].half())
                error_stable[b] = self.error_stable_dict[error_key].float()
                del self.error_stable_dict[error_key]
            
            error_stable.clamp_(min=1e-3)
            foreground_error = error_stable[foreground_mask > 0]
            if foreground_error.numel() > 0:
                error_stable.sub_(foreground_error.mean()).sigmoid_()
            error_stable.mul_(foreground_mask)
            
            self.print_to_log_file(f"error_stable normalized mean (foreground only): {error_stable[foreground_mask > 0].mean().item()}")
        
        del failure, foreground_mask, total_foreground_voxels, foreground_error, failure_spatial
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        
        return error_stable
        
    def compute_prediction_variance(self, seg_output: torch.Tensor, target: torch.Tensor, case_identifiers: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute variance of prediction probabilities with EMA and non-EMA (var1shot), normalized using non-cancer voxels, without spatial smoothing.
        Args:
            seg_output: [B, 3, ...], segmentation logits for 3 regions
            target: [B, 1, ...], ground truth labels
            case_identifiers: List of image IDs for the batch
        Returns:
            variance_stable: [B, 1, ...], EMA-averaged normalized variance-based metric per voxel, 0 for non-cancer
            var1shot: [B, 1, ...], non-EMA normalized variance-based metric per voxel, 0 for non-cancer
        """
        # probs = torch.sigmoid(seg_output)
        # variance = torch.var(probs, dim=1, keepdim=True)
        # cancer_mask, _ = self.get_image_mask(target, seg_output)
        # variance = variance * cancer_mask
        # self.print_to_log_file(f"variance mean (cancer only): {variance[cancer_mask > 0].mean().item() if cancer_mask.sum() > 0 else 0}")
        # variance_stable = torch.zeros_like(variance)
        # epsilon = 1e-3
        # for b in range(variance.shape[0]):
        #     case_id = case_identifiers[b]
        #     variance_key = f"variance_{case_id}"
        #     self.epoch_counts[case_id] += 1
        #     # Apply spatial smoothing per epoch
        #     variance_spatial = F.avg_pool3d(variance[b][None, ...], kernel_size=3, stride=1, padding=1)[0]
        #     if self.error_stable_dict[variance_key] is None:
        #         self.error_stable_dict[variance_key] = variance_spatial.clone().half()
        #     else:
        #         stored_variance = self.error_stable_dict[variance_key]
        #         if stored_variance.shape != variance_spatial.shape:
        #             stored_variance = F.interpolate(stored_variance[None, ...], size=variance_spatial.shape[1:], 
        #                                         mode='trilinear', align_corners=False)[0]
        #             self.error_stable_dict[variance_key] = stored_variance.half()
        #         alpha = min(1.0 / self.epoch_counts[case_id], self.smoothing_alpha)
        #         self.error_stable_dict[variance_key] = (
        #             alpha * variance_spatial.half() + (1 - alpha) * self.error_stable_dict[variance_key]
        #         )
        #     variance_stable[b] = self.error_stable_dict[variance_key].float()
        # variance_stable = variance_stable * cancer_mask
        # self.print_to_log_file(f"variance_stable pre-log mean (cancer only): {variance_stable[cancer_mask > 0].mean().item() if cancer_mask.sum() > 0 else 0}")
        # variance_stable = torch.clamp(variance_stable, min=epsilon)
        # cancer_variance = variance_stable[cancer_mask > 0]
        # if cancer_variance.numel() > 0:
        #     variance_stable = torch.sigmoid(variance_stable - cancer_variance.mean())
        # else:
        #     variance_stable = torch.zeros_like(variance_stable)
        # variance_stable = variance_stable * cancer_mask
        # self.print_to_log_file(f"variance_stable normalized mean (cancer only): {variance_stable[cancer_mask > 0].mean().item() if cancer_mask.sum() > 0 else 0}")
        # # Compute var1shot (non-EMA variance)
        # var1shot = variance * cancer_mask
        # self.print_to_log_file(f"var1shot mean (cancer only): {var1shot[cancer_mask > 0].mean().item() if cancer_mask.sum() > 0 else 0}")
        # var1shot = torch.clamp(var1shot, min=epsilon)
        # cancer_var1shot = var1shot[cancer_mask > 0]
        # if cancer_var1shot.numel() > 0:
        #     var1shot = torch.sigmoid(var1shot - cancer_var1shot.mean())
        # else:
        #     var1shot = torch.zeros_like(var1shot)
        # var1shot = F.avg_pool3d(var1shot, kernel_size=3, stride=1, padding=1)
        # var1shot = var1shot * cancer_mask
        # self.print_to_log_file(f"var1shot normalized mean (cancer only): {var1shot[cancer_mask > 0].mean().item() if cancer_mask.sum() > 0 else 0}")
        return torch.zeros_like(target).float(), torch.zeros_like(target).float()

    def compute_rmsd(self, pred: torch.Tensor, err: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Compute Root Mean Square Deviation (RMSD) between predicted values and error, using cancer voxels only.
        Args:
            pred: [B, 1, ...], predicted map (unc or variance)
            err: [B, 1, ...], normalized log-error
            weights: [B, 1, ...], weights for voxels (cancer mask)
        Returns:
            rmsd: [B], RMSD per sample
        """
        if not (pred.shape == err.shape == weights.shape):
            raise ValueError("Input tensors 'pred', 'err', and 'weights' must have the same shape")
        weights = weights.clamp(0.0, 1.0)
        spatial_dims = list(range(2, len(pred.shape)))
        squared_diff = (pred - err) ** 2 * weights
        weights_sum = weights.sum(dim=spatial_dims)
        mean_squared_diff = squared_diff.sum(dim=spatial_dims) / (weights_sum + 1e-5)
        rmsd = torch.sqrt(mean_squared_diff)
        rmsd = torch.where(weights_sum > 0, rmsd, torch.zeros_like(rmsd))
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        return rmsd.clamp(0.0, float('inf'))

    def get_image_mask(self, target: torch.Tensor, seg_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a mask covering foreground voxels (labels 1-54, indices 0-53) and total number of foreground voxels.
        Args:
            target: [B, C, ...] or [B, 1, ...], ground truth labels (one-hot with 54 channels or label map with values 0-53)
            seg_output: [B, 54, ...], segmentation logits for 54 foreground classes (indices 0-53)
        Returns:
            foreground_mask: [B, 1, ...], 1 for foreground voxels (labels 1-54), 0 elsewhere
            total_foreground_voxels: torch.Tensor, total number of foreground voxels across batch
        """
        with autocast(self.device.type, enabled=True):
            # Validate target shape
            expected_shape = (seg_output.shape[0], target.shape[1] if target.shape[1] > 1 else 1, *seg_output.shape[2:])
            if target.shape != expected_shape:
                self.print_to_log_file(f"Shape mismatch in get_image_mask: Resizing target from {target.shape} to {expected_shape}")
                target = F.interpolate(target.float(), size=seg_output.shape[2:], mode='nearest').long()
            
            if target.dtype == torch.bool:
                target = target.float()
            
            if target.shape[1] > 1:
                foreground_mask = torch.sum(target, dim=1, keepdim=True).clamp_(0, 1)
            else:
                foreground_mask = (target > 0).float()
            
            foreground_voxels = foreground_mask.sum(dim=list(range(2, foreground_mask.ndim)))  # [B]
            total_foreground_voxels = foreground_voxels.sum()
            
            self.print_to_log_file(f"get_image_mask: foreground_voxels_per_batch={foreground_voxels.tolist()}")
            if total_foreground_voxels == 0:
                self.print_to_log_file(f"WARNING: No foreground voxels (labels 1-54) found in batch")
        
        del foreground_voxels
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        
        return foreground_mask, total_foreground_voxels


    def compute_rmsd_score(self, uncertainty: torch.Tensor, error: torch.Tensor, target: torch.Tensor, seg_output: torch.Tensor) -> torch.Tensor:
        """Compute RMSD scores for uncertainty and error using cancer voxels.
        Args:
            uncertainty: [B, 1, ...], predicted uncertainty
            error: [B, 1, ...], log(prediction error)
            target: [B, C, ...] or [B, 1, ...], ground truth labels (one-hot or label map)
            seg_output: [B, N, ...], segmentation logits for N foreground classes (e.g., N=3 or N=54)
        Returns:
            rmsd_cancer: [B], RMSD for cancer
        """
        cancer_mask, _ = self.get_image_mask(target, seg_output)
        unc = uncertainty * cancer_mask
        err = error + 1e-5
        rmsd_cancer = self.compute_rmsd(unc, err, cancer_mask)
        self.print_to_log_file(f"compute_rmsd_score: rmsd_cancer mean {rmsd_cancer.mean().item()}")
        
        del cancer_mask, unc, err
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return rmsd_cancer

    def compute_correlation(self, unc_tensor: torch.Tensor, error: torch.Tensor, target: torch.Tensor, seg_output: torch.Tensor) -> torch.Tensor:
        """Compute weighted Pearson correlation between uncertainty and error using cancer voxels.
        Args:
            unc_tensor: [B, 1, ...], predicted uncertainty
            error: [B, 1, ...], error map
            target: [B, C, ...] or [B, 1, ...], ground truth labels (one-hot or label map)
            seg_output: [B, N, ...], segmentation logits for N foreground classes (e.g., N=3 or N=54)
        Returns:
            corr: [B], weighted Pearson correlation per batch element
        """
        batch_size = unc_tensor.shape[0]
        corr = torch.zeros(batch_size, device=unc_tensor.device)
        cancer_mask, _ = self.get_image_mask(target, seg_output)
        
        for b in range(batch_size):
            u = unc_tensor[b].flatten()
            e = error[b].flatten()
            w = cancer_mask[b].flatten()
            if w.sum() < 1e-5:
                self.print_to_log_file(f"No cancer voxels in batch {b}")
                corr[b] = 0.0
                continue
            u_mean = (w * u).sum() / (w.sum() + 1e-5)
            e_mean = (w * e).sum() / (w.sum() + 1e-5)
            u_std = torch.sqrt(((w * (u - u_mean) ** 2).sum() / (w.sum() + 1e-5)).clamp(min=1e-8))
            e_std = torch.sqrt(((w * (e - e_mean) ** 2).sum() / (w.sum() + 1e-5)).clamp(min=1e-8))
            if u_std > 1e-8 and e_std > 1e-8:
                corr[b] = ((w * (u - u_mean) * (e - e_mean)).sum() / (w.sum() + 1e-5)) / (u_std * e_std)
            else:
                corr[b] = 0.0
            self.print_to_log_file(f"unc_tensor variance (cancer only, batch {b}): {u[w > 0].var().item() if w.sum() > 0 else 0}")
            self.print_to_log_file(f"error variance (cancer only, batch {b}): {e[w > 0].var().item() if w.sum() > 0 else 0}")
        
        del cancer_mask, u, e, w
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return corr

    def _get_deep_supervision_scales(self):
        if self.enable_deep_supervision:
            deep_supervision_scales = list(list(i) for i in 1 / np.cumprod(np.vstack(
                self.configuration_manager.pool_op_kernel_sizes), axis=0))[:-1]
        else:
            deep_supervision_scales = None
        return deep_supervision_scales

    def _set_batch_size_and_oversample(self):
        if not self.is_ddp:
            self.batch_size = self.configuration_manager.batch_size
        else:
            world_size = dist.get_world_size()
            my_rank = dist.get_rank()
            global_batch_size = self.configuration_manager.batch_size
            assert global_batch_size >= world_size, 'Cannot run DDP if the batch size is smaller than the number of GPUs'
            batch_size_per_GPU = [global_batch_size // world_size] * world_size
            batch_size_per_GPU = [batch_size_per_GPU[i] + 1
                                  if (batch_size_per_GPU[i] * world_size + i) < global_batch_size
                                  else batch_size_per_GPU[i]
                                  for i in range(len(batch_size_per_GPU))]
            assert sum(batch_size_per_GPU) == global_batch_size
            sample_id_low = 0 if my_rank == 0 else np.sum(batch_size_per_GPU[:my_rank])
            sample_id_high = np.sum(batch_size_per_GPU[:my_rank + 1])
            oversample = [True if not i < round(global_batch_size * (1 - self.oversample_foreground_percent)) else False
                          for i in range(global_batch_size)]
            if sample_id_high / global_batch_size < (1 - self.oversample_foreground_percent):
                oversample_percent = 0.0
            elif sample_id_low / global_batch_size > (1 - self.oversample_foreground_percent):
                oversample_percent = 1.0
            else:
                oversample_percent = sum(oversample[sample_id_low:sample_id_high]) / batch_size_per_GPU[my_rank]
            print("worker", my_rank, "oversample", oversample_percent)
            print("worker", my_rank, "batch_size", batch_size_per_GPU[my_rank])
            self.batch_size = batch_size_per_GPU[my_rank]
            self.oversample_foreground_percent = oversample_percent

    def _build_loss(self):
        class_weights = torch.ones(55, device=self.device)  # Equal weights for all foreground classes
        class_weights[0] = 0.1
        class_weights[54] = 5.0  # Increase weight for cancer class (label 54, index 53)

        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({'weight': class_weights},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
                                   use_ignore_label=self.label_manager.ignore_label is not None,
                                   dice_class=MemoryEfficientSoftDiceLoss)
        else:
            loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {'weight': class_weights}, weight_ce=1, weight_dice=1,
                                  ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                # Anywho, the simple fix is to set a very low weight to this.
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)

        return loss
    

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)
        if dim == 2:
            do_dummy_2d_data_aug = False
            if max(patch_size) / min(patch_size) > 1.5:
                rotation_for_DA = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            else:
                rotation_for_DA = (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi)
            mirror_axes = (0, 1)
        elif dim == 3:
            do_dummy_2d_data_aug = (max(patch_size) / patch_size[0]) > ANISO_THRESHOLD
            if do_dummy_2d_data_aug:
                rotation_for_DA = (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi)
            else:
                rotation_for_DA = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            mirror_axes = (0, 1, 2)
        else:
            raise RuntimeError()
        initial_patch_size = get_patch_size(patch_size[-dim:],
                                            rotation_for_DA,
                                            rotation_for_DA,
                                            rotation_for_DA,
                                            (0.85, 1.25))
        if do_dummy_2d_data_aug:
            initial_patch_size[0] = patch_size[0]
        self.print_to_log_file(f'do_dummy_2d_data_aug: {do_dummy_2d_data_aug}')
        self.inference_allowed_mirroring_axes = mirror_axes
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

    def print_to_log_file(self, *args, also_print_to_console=True, add_timestamp=True):
        if self.local_rank == 0:
            timestamp = time()
            dt_object = datetime.fromtimestamp(timestamp)
            if add_timestamp:
                args = (f"{dt_object}:", *args)
            successful = False
            max_attempts = 5
            ctr = 0
            while not successful and ctr < max_attempts:
                try:
                    with open(self.log_file, 'a+') as f:
                        for a in args:
                            f.write(str(a))
                            f.write(" ")
                        f.write("\n")
                    successful = True
                except IOError:
                    print(f"{datetime.fromtimestamp(timestamp)}: failed to log: ", sys.exc_info())
                    sleep(0.5)
                    ctr += 1
            if also_print_to_console:
                print(*args)
        elif also_print_to_console:
            print(*args)

    def print_plans(self):
        if self.local_rank == 0:
            dct = deepcopy(self.plans_manager.plans)
            del dct['configurations']
            self.print_to_log_file(f"\nThis is the configuration used by this "
                                   f"training:\nConfiguration name: {self.configuration_name}\n",
                                   self.configuration_manager, '\n', add_timestamp=False)
            self.print_to_log_file('These are the global plan.json settings:\n', dct, '\n', add_timestamp=False)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler

    def plot_network_architecture(self):
        if self._do_i_compile():
            self.print_to_log_file("Unable to plot network architecture: nnUNet_compile is enabled!")
            return
        if self.local_rank == 0:
            try:
                import hiddenlayer as hl
                g = hl.build_graph(self.network,
                                   torch.rand((1, self.num_input_channels,
                                               *self.configuration_manager.patch_size),
                                              device=self.device),
                                   transforms=None)
                g.save(join(self.output_folder, "network_architecture.pdf"))
                del g
            except Exception as e:
                self.print_to_log_file("Unable to plot network architecture:")
                self.print_to_log_file(e)
            finally:
                empty_cache(self.device)

    def do_split(self):
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
        if self.fold == "all":
            case_identifiers = self.dataset_class.get_identifiers(self.preprocessed_dataset_folder)
            tr_keys = case_identifiers
            val_keys = tr_keys
        else:
            splits_file = join(self.preprocessed_dataset_folder_base, "splits_final.json")
            dataset = self.dataset_class(self.preprocessed_dataset_folder,
                                         identifiers=None,
                                         folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                all_keys_sorted = list(np.sort(list(dataset.identifiers)))
                splits = generate_crossval_split(all_keys_sorted, seed=12345, n_splits=5)
                save_json(splits, splits_file)
            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_json(splits_file)
                self.print_to_log_file(f"The split file contains {len(splits)} splits.")
            self.print_to_log_file("Desired fold for training: %d" % self.fold)
            if self.fold < len(splits):
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['val']
                self.print_to_log_file("This split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            else:
                self.print_to_log_file("INFO: You requested fold %d for training but splits "
                                       "contain only %d folds. I am now creating a "
                                       "random (but seeded) 80:20 split!" % (self.fold, len(splits)))
                rnd = np.random.RandomState(seed=12345 + self.fold)
                keys = np.sort(list(dataset.identifiers))
                idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                tr_keys = [keys[i] for i in idx_tr]
                val_keys = [keys[i] for i in idx_val]
                self.print_to_log_file("This random 80:20 split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            if any([i in val_keys for i in tr_keys]):
                self.print_to_log_file('WARNING: Some validation cases are also in the training set. Please check the '
                                       'splits.json or ignore if this is intentional.')
        return tr_keys, val_keys

    def get_tr_and_val_datasets(self):
        tr_keys, val_keys = self.do_split()
        dataset_tr = self.dataset_class(self.preprocessed_dataset_folder, tr_keys,
                                        folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
        dataset_val = self.dataset_class(self.preprocessed_dataset_folder, val_keys,
                                         folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
        return dataset_tr, dataset_val

    def get_dataloaders(self):
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
        patch_size = self.configuration_manager.patch_size
        deep_supervision_scales = self._get_deep_supervision_scales()
        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)
        val_transforms = self.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()
        dl_tr = nnUNetDataLoader(dataset_tr, self.batch_size,
                                 initial_patch_size,
                                 self.configuration_manager.patch_size,
                                 self.label_manager,
                                 oversample_foreground_percent=self.oversample_foreground_percent,
                                 sampling_probabilities=None, pad_sides=None, transforms=tr_transforms,
                                 probabilistic_oversampling=self.probabilistic_oversampling)
        dl_val = nnUNetDataLoader(dataset_val, self.batch_size,
                                  self.configuration_manager.patch_size,
                                  self.configuration_manager.patch_size,
                                  self.label_manager,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
                                  probabilistic_oversampling=self.probabilistic_oversampling)
        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(data_loader=dl_tr, transform=None,
                                                        num_processes=allowed_num_processes,
                                                        num_cached=max(6, allowed_num_processes // 2), seeds=None,
                                                        pin_memory=self.device.type == 'cuda', wait_time=0.002)
            mt_gen_val = NonDetMultiThreadedAugmenter(data_loader=dl_val,
                                                      transform=None, num_processes=max(1, allowed_num_processes // 2),
                                                      num_cached=max(3, allowed_num_processes // 4), seeds=None,
                                                      pin_memory=self.device.type == 'cuda',
                                                      wait_time=0.002)
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val

    def get_training_transforms(
            self,
            patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA: RandomScalar,
            deep_supervision_scales: Union[List, Tuple, None],
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            use_mask_for_norm: List[bool] = None,
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> BasicTransform:
        transforms = []
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None
        transforms.append(
            SpatialTransform(
                patch_size_spatial, patch_center_dist_from_border=0, random_crop=False, p_elastic_deform=0,
                p_rotation=0.2,
                rotation=rotation_for_DA, p_scaling=0.2, scaling=(0.7, 1.4), p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False
            )
        )
        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())
        transforms.append(RandomTransform(
            GaussianNoiseTransform(
                noise_variance=(0, 0.1),
                p_per_channel=1,
                synchronize_channels=True
            ), apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            GaussianBlurTransform(
                blur_sigma=(0.5, 1.),
                synchronize_channels=False,
                synchronize_axes=False,
                p_per_channel=0.5, benchmark=True
            ), apply_probability=0.2
        ))
        transforms.append(RandomTransform(
            MultiplicativeBrightnessTransform(
                multiplier_range=BGContrast((0.75, 1.25)),
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            ContrastTransform(
                contrast_range=BGContrast((0.75, 1.25)),
                preserve_range=True,
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            SimulateLowResolutionTransform(
                scale=(0.5, 1),
                synchronize_channels=False,
                synchronize_axes=True,
                ignore_axes=ignore_axes,
                allowed_channels=None,
                p_per_channel=0.5
            ), apply_probability=0.25
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=1,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=0,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.3
        ))
        if mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(
                MirrorTransform(
                    allowed_axes=mirror_axes
                )
            )
        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(MaskImageTransform(
                apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                channel_idx_in_seg=0,
                set_outside_to=0,
            ))
        transforms.append(
            RemoveLabelTansform(-1, 0)
        )
        if is_cascaded:
            assert foreground_labels is not None, 'We need foreground_labels for cascade augmentations'
            transforms.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1,
                    all_labels=foreground_labels,
                    remove_channel_from_source=True
                )
            )
            transforms.append(
                RandomTransform(
                    ApplyRandomBinaryOperatorTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        strel_size=(1, 8),
                        p_per_label=1
                    ), apply_probability=0.4
                )
            )
            transforms.append(
                RandomTransform(
                    RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        fill_with_other_class_p=0,
                        dont_do_if_covers_more_than_x_percent=0.15,
                        p_per_label=1
                    ), apply_probability=0.2
                )
            )
        if regions is not None:
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0
                )
            )
        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))
        return ComposeTransforms(transforms)

    @staticmethod
    def get_validation_transforms(
            deep_supervision_scales: Union[List, Tuple, None],
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> BasicTransform:
        transforms = []
        transforms.append(
            RemoveLabelTansform(-1, 0)
        )
        if is_cascaded:
            transforms.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1,
                    all_labels=foreground_labels,
                    remove_channel_from_source=True
                )
            )
        if regions is not None:
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0
                )
            )
        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))
        return ComposeTransforms(transforms)

    def set_deep_supervision_enabled(self, enabled: bool):
        if self.is_ddp:
            mod = self.network.module
        else:
            mod = self.network
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod
        mod.decoder.deep_supervision = enabled

    def on_train_start(self):
        if not self.was_initialized:
            self.initialize()
        self.dataloader_train, self.dataloader_val = self.get_dataloaders()
        maybe_mkdir_p(self.output_folder)
        self.set_deep_supervision_enabled(self.enable_deep_supervision)
        self.print_plans()
        empty_cache(self.device)
        if self.local_rank == 0:
            self.dataset_class.unpack_dataset(
                self.preprocessed_dataset_folder,
                overwrite_existing=False,
                num_processes=max(1, round(get_allowed_n_proc_DA() // 2)),
                verify=True)
        if self.is_ddp:
            dist.barrier()
        save_json(self.plans_manager.plans, join(self.output_folder_base, 'plans.json'), sort_keys=False)
        save_json(self.dataset_json, join(self.output_folder_base, 'dataset.json'), sort_keys=False)
        shutil.copy(join(self.preprocessed_dataset_folder_base, 'dataset_fingerprint.json'),
                    join(self.output_folder_base, 'dataset_fingerprint.json'))
        self.plot_network_architecture()
        self._save_debug_information()

    def on_train_end(self):
        self.current_epoch -= 1
        self.save_checkpoint(join(self.output_folder, "checkpoint_final.pth"))
        self.current_epoch += 1
        if self.local_rank == 0 and isfile(join(self.output_folder, "checkpoint_latest.pth")):
            os.remove(join(self.output_folder, "checkpoint_latest.pth"))
        old_stdout = sys.stdout
        with open(os.devnull, 'w') as f:
            sys.stdout = f
            if self.dataloader_train is not None and \
                    isinstance(self.dataloader_train, (NonDetMultiThreadedAugmenter, MultiThreadedAugmenter)):
                self.dataloader_train._finish()
            if self.dataloader_val is not None and \
                    isinstance(self.dataloader_val, (NonDetMultiThreadedAugmenter, MultiThreadedAugmenter)):
                self.dataloader_val._finish()
            sys.stdout = old_stdout
        empty_cache(self.device)
        self.print_to_log_file("Training done.")

    def on_train_epoch_start(self):
        self.network.train()
        self.lr_scheduler.step(self.current_epoch)
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch}')
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}")
        self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        case_identifiers = batch.get('case_identifiers', [f"case_{i}" for i in range(data.shape[0])])
        data = data.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)
        self.print_to_log_file(f"train target shape: {target.shape}, dtype: {target.dtype}, unique: {torch.unique(target).tolist()}")
        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            seg_output = output[:, :54, ...]
            unc_output = output[:, 54:55, ...]
            target_for_seg = target.long()  # [B, D, H, W]
            valid_mask = (target_for_seg > 0)
            target_for_seg = target_for_seg - 1
            target_for_seg = torch.where(valid_mask, target_for_seg, -100)
            error = self.compute_prediction_error_probability(seg_output, target, case_identifiers)
            variance_stable, var1shot = self.compute_prediction_variance(seg_output, target, case_identifiers)
            l_seg = self.loss(seg_output, target_for_seg)
            cancer_mask, _ = self.get_image_mask(target, seg_output)
            l_unc = F.mse_loss(torch.sigmoid(unc_output) * cancer_mask, error * cancer_mask)
            rmsd_cancer = self.compute_rmsd_score(torch.sigmoid(unc_output), error, target, seg_output)
            corr = self.compute_correlation(unc_output, error, target, seg_output)
            probs = torch.sigmoid(seg_output)
            weights = torch.sum(probs, dim=1, keepdim=True).clamp(0, 1)
            rmsd_variance = self.compute_rmsd_score(variance_stable, error, target, seg_output)
            corr_variance = self.compute_correlation(variance_stable, error, target, seg_output)
            rmsd_var1shot = self.compute_rmsd_score(var1shot, error, target, seg_output)
            corr_var1shot = self.compute_correlation(var1shot, error, target, seg_output)
            cancer_error = error[cancer_mask > 0]
            cancer_variance = variance_stable[cancer_mask > 0]
            cancer_var1shot = var1shot[cancer_mask > 0]
            self.print_to_log_file(f"unc_output mean: {unc_output.mean().item()}, std: {unc_output.std().item()}")
            self.print_to_log_file(f"variance_stable mean: {variance_stable.mean().item()}, std: {variance_stable.std().item()}")
            self.print_to_log_file(f"var1shot mean: {var1shot.mean().item()}, std: {var1shot.std().item()}")
            self.print_to_log_file(f"cancer_error variance: {cancer_error.var().item() if cancer_error.numel() > 0 else 0}")
            self.print_to_log_file(f"cancer_variance variance: {cancer_variance.var().item() if cancer_variance.numel() > 0 else 0}")
            self.print_to_log_file(f"cancer_var1shot variance: {cancer_var1shot.var().item() if cancer_var1shot.numel() > 0 else 0}")
            ramp_epochs = 100
            unc_weight = .1
            rmsd_weight_cancer = 0.5
            corr_weight = 0.05
            l = l_seg + unc_weight * l_unc + rmsd_weight_cancer * rmsd_cancer.mean() + corr_weight * (1.0 - corr.mean())
            self.print_to_log_file(f"weights: {unc_weight} {rmsd_weight_cancer} {corr_weight}")
            self.print_to_log_file(f"Loss components: seg={l_seg.item()}, unc={l_unc.item()}, rmsd_cancer={rmsd_cancer.mean().item()}, corr={corr.mean().item()}, rmsd_variance={rmsd_variance.mean().item()}, corr_variance={corr_variance.mean().item()}, rmsd_var1shot={rmsd_var1shot.mean().item()}, corr_var1shot={corr_var1shot.mean().item()}")
        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        return {
            'loss': l.detach().cpu().numpy(),
            'correlation_coefficient': corr.mean().detach().cpu().numpy(),
            'rmsd_cancer': rmsd_cancer.mean().detach().cpu().numpy(),
            'rmsd_variance': rmsd_variance.mean().detach().cpu().numpy(),
            'correlation_variance': corr_variance.mean().detach().cpu().numpy(),
            'rmsd_var1shot': rmsd_var1shot.mean().detach().cpu().numpy(),
            'corr_var1shot': corr_var1shot.mean().detach().cpu().numpy(),
        }

    def validation_step(self, batch: dict) -> dict:
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        data = batch['data']
        target = batch['target']
        case_identifiers = batch.get('case_identifiers', [f"case_{i}" for i in range(data.shape[0])])
        properties = batch.get('properties', [{} for _ in range(data.shape[0])])
        data = data.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)
        self.print_to_log_file(f"val target shape: {target.shape}, dtype: {target.dtype}, unique: {torch.unique(target).tolist()}")
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            seg_output = output[:, :54, ...]
            unc_output = output[:, 54:55, ...].float()
            target_for_seg = target.long()  # [B, D, H, W]
            valid_mask = (target_for_seg > 0)
            target_for_seg = target_for_seg - 1
            target_for_seg = torch.where(valid_mask, target_for_seg, -100)
            error = self.compute_prediction_error_probability(seg_output, target, case_identifiers)
            variance_stable, var1shot = self.compute_prediction_variance(seg_output, target, case_identifiers)
            l = self.loss(seg_output, target_for_seg)
            cancer_mask, _ = self.get_image_mask(target, seg_output)
            rmsd_cancer = self.compute_rmsd_score(torch.sigmoid(unc_output), error, target, seg_output)
            corr = self.compute_correlation(unc_output, error, target, seg_output)
            probs = torch.sigmoid(seg_output)
            weights = cancer_mask
            rmsd_variance = self.compute_rmsd_score(variance_stable, error, target, seg_output)
            corr_variance = self.compute_correlation(variance_stable, error, target, seg_output)
            rmsd_var1shot = self.compute_rmsd_score(var1shot, error, target, seg_output)
            corr_var1shot = self.compute_correlation(var1shot, error, target, seg_output)
            axes = [0] + list(range(2, seg_output.ndim))
            predicted_segmentation_onehot = (torch.sigmoid(seg_output) > 0.5).float()
            if self.label_manager.has_ignore_label:
                mask = (target_for_seg != self.label_manager.ignore_label).float()
                target_for_seg[target_for_seg == self.label_manager.ignore_label] = 0
            else:
                mask = None
            tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target_for_seg, axes=axes, mask=mask)
            tp_hard = tp.detach().cpu().numpy()
            fp_hard = fp.detach().cpu().numpy()
            fn_hard = fn.detach().cpu().numpy()

            if self.current_epoch % self.save_every == 0 and self.local_rank == 0:
                validation_maps_folder = join(self.output_folder, f'maps')
                maybe_mkdir_p(validation_maps_folder)
                for b in range(data.shape[0]):
                    case_id = case_identifiers[b]
                    if self.current_epoch == 0:
                        if case_id not in self.processed_val_cases and len(self.processed_val_cases) < self.max_val_cases_to_save:
                            self.processed_val_cases.add(case_id)
                    if case_id in self.processed_val_cases:
                        target_np = target_for_seg[b].detach().cpu().numpy()[0].astype(np.float32)
                        seg_np = torch.argmax(seg_output[b], dim=1).detach().cpu().numpy().astype(np.float32)
                        cancer_mask_np = cancer_mask[b].detach().cpu().numpy().squeeze().astype(np.float32)
                        error_np = error[b].detach().cpu().numpy().squeeze().astype(np.float32)
                        unc_np = torch.sigmoid(unc_output[b]).detach().cpu().numpy().squeeze().astype(np.float32)
                        variance_np = variance_stable[b].detach().cpu().numpy().squeeze().astype(np.float32)
                        var1shot_np = var1shot[b].detach().cpu().numpy().squeeze().astype(np.float32)
                        error_np = error_np * cancer_mask_np
                        unc_np = unc_np * cancer_mask_np
                        variance_np = variance_np * cancer_mask_np
                        var1shot_np = var1shot_np * cancer_mask_np
                        self.print_to_log_file(f"unc_np non-cancer mean: {unc_np[cancer_mask_np == 0].mean()}")
                        self.print_to_log_file(f"unc_np non-cancer max: {unc_np[cancer_mask_np == 0].max()}")
                        affine = properties[b].get('affine', np.eye(4))
                        header = properties[b].get('header', None)
                        target_img = nib.Nifti1Image(target_np, affine=affine, header=header)
                        target_filename = join(validation_maps_folder, f"{case_id}_{self.current_epoch}_target_stable.nii.gz")
                        nib.save(target_img, target_filename)
                        seg_img = nib.Nifti1Image(seg_np, affine=affine, header=header)
                        seg_filename = join(validation_maps_folder, f"{case_id}_{self.current_epoch}_seg_stable.nii.gz")
                        nib.save(seg_img, seg_filename)
                        self.print_to_log_file(f"save_step - target_for_seg shape: {target_for_seg.shape}, target_np shape: {target_np.shape}")
                        self.print_to_log_file(f"save_step - seg_output shape: {seg_output.shape}, seg_np shape: {seg_np.shape}")
                        error_img = nib.Nifti1Image(error_np, affine=affine, header=header)
                        error_filename = join(validation_maps_folder, f"{case_id}_{self.current_epoch}_error_stable.nii.gz")
                        nib.save(error_img, error_filename)
                        self.print_to_log_file(f"Saved error_stable for {case_id} at {error_filename}")
                        unc_img = nib.Nifti1Image(unc_np, affine=affine, header=header)
                        unc_filename = join(validation_maps_folder, f"{case_id}_{self.current_epoch}_uncertainty.nii.gz")
                        nib.save(unc_img, unc_filename)
                        self.print_to_log_file(f"Saved uncertainty for {case_id} at {unc_filename}")
                        variance_img = nib.Nifti1Image(variance_np, affine=affine, header=header)
                        variance_filename = join(validation_maps_folder, f"{case_id}_{self.current_epoch}_variance.nii.gz")
                        nib.save(variance_img, variance_filename)
                        self.print_to_log_file(f"Saved variance for {case_id} at {variance_filename}")
                        var1shot_img = nib.Nifti1Image(var1shot_np, affine=affine, header=header)
                        var1shot_filename = join(validation_maps_folder, f"{case_id}_{self.current_epoch}_var1shot.nii.gz")
                        nib.save(var1shot_img, var1shot_filename)
                        self.print_to_log_file(f"Saved var1shot for {case_id} at {var1shot_filename}")
                self.print_to_log_file(f"Processed validation cases so far: {len(self.processed_val_cases)}")
        return {
            'loss': l.detach().cpu().numpy(),
            'tp_hard': tp_hard,
            'fp_hard': fp_hard,
            'fn_hard': fn_hard,
            'correlation_coefficient': corr.mean().detach().cpu().numpy(),
            'rmsd_cancer': rmsd_cancer.mean().detach().cpu().numpy(),
            'rmsd_variance': rmsd_variance.mean().detach().cpu().numpy(),
            'correlation_variance': corr_variance.mean().detach().cpu().numpy(),
            'rmsd_var1shot': rmsd_var1shot.mean().detach().cpu().numpy(),
            'corr_var1shot': corr_var1shot.mean().detach().cpu().numpy(),
        }

    def on_train_epoch_end(self, train_outputs: List[dict]):
        outputs = collate_outputs(train_outputs)
        if self.is_ddp:
            losses_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_tr, outputs['loss'])
            loss_here = np.vstack(losses_tr).mean()
            corrs_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(corrs_tr, outputs['correlation_coefficient'])
            corr_here = np.vstack(corrs_tr).mean()
            rmsd_cancer_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(rmsd_cancer_tr, outputs['rmsd_cancer'])
            rmsd_cancer_here = np.vstack(rmsd_cancer_tr).mean()
            rmsd_var_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(rmsd_var_tr, outputs['rmsd_variance'])
            rmsd_var_here = np.vstack(rmsd_var_tr).mean()
            corr_var_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(corr_var_tr, outputs['correlation_variance'])
            corr_var_here = np.vstack(corr_var_tr).mean()
            rmsd_var1shot_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(rmsd_var1shot_tr, outputs['rmsd_var1shot'])
            rmsd_var1shot_here = np.vstack(rmsd_var1shot_tr).mean()
            corr_var1shot_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(corr_var1shot_tr, outputs['corr_var1shot'])
            corr_var1shot_here = np.vstack(corr_var1shot_tr).mean()
        else:
            loss_here = np.mean(outputs['loss'])
            corr_here = np.mean(outputs['correlation_coefficient'])
            rmsd_cancer_here = np.mean(outputs['rmsd_cancer'])
            rmsd_var_here = np.mean(outputs['rmsd_variance'])
            corr_var_here = np.mean(outputs['correlation_variance'])
            rmsd_var1shot_here = np.mean(outputs['rmsd_var1shot'])
            corr_var1shot_here = np.mean(outputs['corr_var1shot'])
        self.logger.log('train_losses', loss_here, self.current_epoch)
        self.logger.log('train_correlation_coefficient', corr_here, self.current_epoch)
        self.logger.log('train_rmsd_cancer', rmsd_cancer_here, self.current_epoch)
        self.logger.log('train_rmsd_variance', rmsd_var_here, self.current_epoch)
        self.logger.log('train_correlation_variance', corr_var_here, self.current_epoch)
        self.logger.log('train_rmsd_var1shot', rmsd_var1shot_here, self.current_epoch)
        self.logger.log('train_corr_var1shot', corr_var1shot_here, self.current_epoch)
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def on_validation_epoch_start(self):
        self.network.eval()

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)
        tp = np.sum(outputs_collated['tp_hard'], 0)
        fp = np.sum(outputs_collated['fp_hard'], 0)
        fn = np.sum(outputs_collated['fn_hard'], 0)
        if self.is_ddp:
            world_size = dist.get_world_size()
            tps = [None for _ in range(world_size)]
            dist.all_gather_object(tps, tp)
            tp = np.vstack([i[None] for i in tps]).sum(0)
            fps = [None for _ in range(world_size)]
            dist.all_gather_object(fps, fp)
            fp = np.vstack([i[None] for i in fps]).sum(0)
            fns = [None for _ in range(world_size)]
            dist.all_gather_object(fns, fn)
            fn = np.vstack([i[None] for i in fns]).sum(0)
            losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val, outputs_collated['loss'])
            loss_here = np.vstack(losses_val).mean()
            corrs_val = [None for _ in range(world_size)]
            dist.all_gather_object(corrs_val, outputs_collated['correlation_coefficient'])
            corr_here = np.vstack(corrs_val).mean()
            rmsd_cancer_val = [None for _ in range(world_size)]
            dist.all_gather_object(rmsd_cancer_val, outputs_collated['rmsd_cancer'])
            rmsd_cancer_here = np.vstack(rmsd_cancer_val).mean()
            rmsd_var_val = [None for _ in range(world_size)]
            dist.all_gather_object(rmsd_var_val, outputs_collated['rmsd_variance'])
            rmsd_var_here = np.vstack(rmsd_var_val).mean()
            corr_var_val = [None for _ in range(world_size)]
            dist.all_gather_object(corr_var_val, outputs_collated['correlation_variance'])
            corr_var_here = np.vstack(corr_var_val).mean()
            rmsd_var1shot_val = [None for _ in range(world_size)]
            dist.all_gather_object(rmsd_var1shot_val, outputs_collated['rmsd_var1shot'])
            rmsd_var1shot_here = np.vstack(rmsd_var1shot_val).mean()
            corr_var1shot_val = [None for _ in range(world_size)]
            dist.all_gather_object(corr_var1shot_val, outputs_collated['corr_var1shot'])
            corr_var1shot_here = np.vstack(corr_var1shot_val).mean()
        else:
            loss_here = np.mean(outputs_collated['loss'])
            corr_here = np.mean(outputs_collated['correlation_coefficient'])
            rmsd_cancer_here = np.mean(outputs_collated['rmsd_cancer'])
            rmsd_var_here = np.mean(outputs_collated['rmsd_variance'])
            corr_var_here = np.mean(outputs_collated['correlation_variance'])
            rmsd_var1shot_here = np.mean(outputs_collated['rmsd_var1shot'])
            corr_var1shot_here = np.mean(outputs_collated['corr_var1shot'])
        global_dc_per_class = [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)]
        mean_fg_dice = np.nanmean(global_dc_per_class)
        self.logger.log('mean_fg_dice', mean_fg_dice, self.current_epoch)
        self.logger.log('dice_per_class_or_region', global_dc_per_class, self.current_epoch)
        self.logger.log('val_losses', loss_here, self.current_epoch)
        self.logger.log('val_correlation_coefficient', corr_here, self.current_epoch)
        self.logger.log('val_rmsd_cancer', rmsd_cancer_here, self.current_epoch)
        self.logger.log('val_rmsd_variance', rmsd_var_here, self.current_epoch)
        self.logger.log('val_correlation_variance', corr_var_here, self.current_epoch)
        self.logger.log('val_rmsd_var1shot', rmsd_var1shot_here, self.current_epoch)
        self.logger.log('val_corr_var1shot', corr_var1shot_here, self.current_epoch)
        self.print_to_log_file('val_correlation_coefficient', np.round(corr_here, decimals=4))
        self.print_to_log_file('val_rmsd_cancer', np.round(rmsd_cancer_here, decimals=4))
        self.print_to_log_file('val_rmsd_variance', np.round(rmsd_var_here, decimals=4))
        self.print_to_log_file('val_correlation_variance', np.round(corr_var_here, decimals=4))
        self.print_to_log_file('val_rmsd_var1shot', np.round(rmsd_var1shot_here, decimals=4))
        self.print_to_log_file('val_corr_var1shot', np.round(corr_var1shot_here, decimals=4))
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def on_epoch_start(self):
        self.logger.log('epoch_start_timestamps', time(), self.current_epoch)

    def on_epoch_end(self):
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)
        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        self.print_to_log_file('Pseudo dice', [np.round(i, decimals=4) for i in
                                               self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]])
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))
        if self._best_ema is None or self.logger.my_fantastic_logging['ema_fg_dice'][-1] > self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
            self.print_to_log_file(f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))
        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)
        self.current_epoch += 1

    def on_epoch_end(self):
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)
        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        self.print_to_log_file('Pseudo dice', [np.round(i, decimals=4) for i in
                                               self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]])
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))
        if self._best_ema is None or self.logger.my_fantastic_logging['ema_fg_dice'][-1] > self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
            self.print_to_log_file(f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))
        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)
        self.current_epoch += 1

    def save_checkpoint(self, filename: str) -> None:
        if self.local_rank == 0:
            if not self.disable_checkpointing:
                if self.is_ddp:
                    mod = self.network.module
                else:
                    mod = self.network
                if isinstance(mod, OptimizedModule):
                    mod = mod._orig_mod
                checkpoint = {
                    'network_weights': mod.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'grad_scaler_state': self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
                    'logging': self.logger.get_checkpoint(),
                    '_best_ema': self._best_ema,
                    'current_epoch': self.current_epoch + 1,
                    'init_args': self.my_init_kwargs,
                    'trainer_name': self.__class__.__name__,
                    'inference_allowed_mirroring_axes': self.inference_allowed_mirroring_axes,
                    'processed_val_cases': list(self.processed_val_cases),
                }
                torch.save(checkpoint, filename)
            else:
                self.print_to_log_file('No checkpoint written, checkpointing is disabled')

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        if not self.was_initialized:
            self.initialize()
        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device, weights_only=False)
        new_state_dict = {}
        for k, value in checkpoint['network_weights'].items():
            key = k
            if key not in self.network.state_dict().keys() and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value
        self.my_init_kwargs = checkpoint['init_args']
        self.current_epoch = checkpoint['current_epoch']
        self.logger.load_checkpoint(checkpoint['logging'])
        self._best_ema = checkpoint['_best_ema']
        self.inference_allowed_mirroring_axes = checkpoint[
            'inference_allowed_mirroring_axes'] if 'inference_allowed_mirroring_axes' in checkpoint.keys() else self.inference_allowed_mirroring_axes
        self.processed_val_cases = set(checkpoint.get('processed_val_cases', []))
        if self.is_ddp:
            if isinstance(self.network.module, OptimizedModule):
                self.network.module._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.module.load_state_dict(new_state_dict)
        else:
            if isinstance(self.network, OptimizedModule):
                self.network._orig_mod.load_state_dict(new_state_dict)
            else:
                self.network.load_state_dict(new_state_dict)
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if self.grad_scaler is not None:
            if checkpoint['grad_scaler_state'] is not None:
                self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])

    def perform_actual_validation(self, save_probabilities: bool = False):
        self.set_deep_supervision_enabled(False)
        self.network.eval()
        if self.is_ddp and self.batch_size == 1 and self.enable_deep_supervision and self._do_i_compile():
            self.print_to_log_file("WARNING! batch size is 1 during training and torch.compile is enabled.")
        predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                    perform_everything_on_device=True, device=self.device, verbose=False,
                                    verbose_preprocessing=False, allow_tqdm=False)
        predictor.manual_initialization(self.network, self.plans_manager, self.configuration_manager, None,
                                        self.dataset_json, self.__class__.__name__,
                                        self.inference_allowed_mirroring_axes)
        with multiprocessing.get_context("spawn").Pool(default_num_processes) as segmentation_export_pool:
            worker_list = [i for i in segmentation_export_pool._pool]
            validation_output_folder = join(self.output_folder, 'validation')
            maybe_mkdir_p(validation_output_folder)
            _, val_keys = self.do_split()
            if self.is_ddp:
                last_barrier_at_idx = len(val_keys) // dist.get_world_size() - 1
                val_keys = val_keys[self.local_rank:: dist.get_world_size()]
            dataset_val = self.dataset_class(self.preprocessed_dataset_folder, val_keys,
                                             folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
            next_stages = self.configuration_manager.next_stage_names
            if next_stages is not None:
                _ = [maybe_mkdir_p(join(self.output_folder_base, 'predicted_next_stage', n)) for n in next_stages]
            results = []
            for i, k in enumerate(dataset_val.identifiers):
                proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                           allowed_num_queued=2)
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                               allowed_num_queued=2)
                self.print_to_log_file(f"predicting {k}")
                data, _, seg_prev, properties = dataset_val.load_case(k)
                data = data[:]
                if self.is_cascaded:
                    seg_prev = seg_prev[:]
                    data = np.vstack((data, convert_labelmap_to_one_hot(seg_prev, self.label_manager.foreground_labels,
                                                                        output_dtype=data.dtype)))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    data = torch.from_numpy(data)
                self.print_to_log_file(f'{k}, shape {data.shape}, rank {self.local_rank}')
                output_filename_truncated = join(validation_output_folder, k)
                prediction = predictor.predict_sliding_window_return_logits(data)
                prediction = prediction.cpu()
                results.append(
                    segmentation_export_pool.starmap_async(
                        export_prediction_from_logits, (
                            (prediction, properties, self.configuration_manager, self.plans_manager,
                             self.dataset_json, output_filename_truncated, save_probabilities),
                        )
                    )
                )
                if next_stages is not None:
                    for n in next_stages:
                        next_stage_config_manager = self.plans_manager.get_configuration(n)
                        expected_preprocessed_folder = join(nnUNet_preprocessed, self.plans_manager.dataset_name,
                                                            next_stage_config_manager.data_identifier)
                        dataset_class = infer_dataset_class(expected_preprocessed_folder)
                        try:
                            tmp = dataset_class(expected_preprocessed_folder, [k])
                            d, _, _, _ = tmp.load_case(k)
                        except FileNotFoundError:
                            self.print_to_log_file(
                                f"Predicting next stage {n} failed for case {k} because the preprocessed file is missing!")
                            continue
                        target_shape = d.shape[1:]
                        output_folder = join(self.output_folder_base, 'predicted_next_stage', n)
                        output_file_truncated = join(output_folder, k)
                        results.append(segmentation_export_pool.starmap_async(
                            resample_and_save, (
                                (prediction, target_shape, output_file_truncated, self.plans_manager,
                                 self.configuration_manager,
                                 properties,
                                 self.dataset_json,
                                 default_num_processes,
                                 dataset_class),
                            )
                        ))
                if self.is_ddp and i < last_barrier_at_idx and (i + 1) % 20 == 0:
                    dist.barrier()
            _ = [r.get() for r in results]
        if self.is_ddp:
            dist.barrier()
        if self.local_rank == 0:
            metrics = compute_metrics_on_folder(join(self.preprocessed_dataset_folder_base, 'gt_segmentations'),
                                                validation_output_folder,
                                                join(validation_output_folder, 'summary.json'),
                                                self.plans_manager.image_reader_writer_class(),
                                                self.dataset_json["file_ending"],
                                                self.label_manager.foreground_regions if self.label_manager.has_regions else
                                                self.label_manager.foreground_labels,
                                                self.label_manager.ignore_label, chill=True,
                                                num_processes=default_num_processes * dist.get_world_size() if
                                                self.is_ddp else default_num_processes)
            self.print_to_log_file("Validation complete", also_print_to_console=True)
            self.print_to_log_file("Mean Validation Dice: ", (metrics['foreground_mean']["Dice"]),
                                   also_print_to_console=True)
        self.set_deep_supervision_enabled(False)
        compute_gaussian.cache_clear()

    def run_training(self):
        self.on_train_start()
        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()
            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)
            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)
            self.on_epoch_end()
        self.on_train_end()