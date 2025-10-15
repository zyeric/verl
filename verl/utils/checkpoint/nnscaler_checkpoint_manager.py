# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import warnings
from typing import Optional, Union, Any, Dict
from pathlib import Path

import torch
import torch.distributed
from accelerate import init_empty_weights
from omegaconf import DictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardedOptimStateDictConfig, ShardedStateDictConfig, StateDictType
from transformers import GenerationConfig, PreTrainedTokenizer, ProcessorMixin

from verl.utils.device import is_cuda_available
from verl.utils.fs import copy_to_local, is_non_local
from verl.utils.fsdp_utils import fsdp_version, get_fsdp_full_state_dict, get_fsdp_state_ctx
from verl.utils.nnscaler_utils import get_nnscaler_full_state_dict
from verl.utils.logger import log_with_rank

import nnscaler
from nnscaler.runtime.module import ParallelModule
from nnscaler.runtime.device import DeviceGroup
from nnscaler.cli.trainer import Trainer

from .checkpoint_manager import BaseCheckpointManager

# Setup logging
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class NNScalerCheckpointManager(BaseCheckpointManager):
    """
    Manage nnScaler ParallelModule checkpointing in training.

    - Saves/loads per-rank sharded model & optimizer states
    - Persists full lr_scheduler and RNG state
    - Stores HF tokenizer/processor and model/config for unified restore

    Args:
        model (torch.nn.Module): Wrapped model instance.
        optimizer (Optimizer): Training optimizer.
        lr_scheduler (LRScheduler): Learning-rate scheduler.
        processing_class (PreTrainedTokenizer or ProcessorMixin, optional):
            Pre-/post-processing artifact handler.
        checkpoint_contents DictConfig: Configuration for checkpoint contents.
            - 'load': Components to load; must contain 'model'. Defaults to ['model', 'optimizer', 'extra'].
            - 'save': Components to save; must contain 'model'. Defaults to ['model', 'optimizer', 'extra'].
    """

    def __init__(
        self,
        config,
        model_config,
        hf_config,
        model: ParallelModule,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        processing_class: Union[PreTrainedTokenizer, ProcessorMixin] = None,
        checkpoint_contents: DictConfig = None,
        n_gpus_per_node: int = 1,
        with_merged: bool = False,
        load_type: str = "deduped",
        save_type: str = "deduped",
        can_generate: bool = False,
        **kwargs,
    ):
        if processing_class is None:
            assert "tokenizer" in kwargs, "tokenizer or processor must be provided"
            warnings.warn("`tokenizer` is deprecated. use `processing_class` instead.", DeprecationWarning, stacklevel=2)
            processing_class = kwargs.pop("tokenizer")

        super().__init__(
            model,
            optimizer,
            lr_scheduler=lr_scheduler,
            processing_class=processing_class,
            checkpoint_contents=checkpoint_contents,
        )

        assert self.should_load_model, "current implementation assumes model should be loaded"
        assert self.should_save_model, "current implementation assumes model should be saved"

        # These env variables are not correctly set in verl
        # self.local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE'))
        # self.local_rank = int(os.environ.get('LOCAL_RANK'))
        # self.node_rank = int(os.environ.get('GROUP_RANK'))
        self.local_world_size = n_gpus_per_node
        self.local_rank = self.rank % self.local_world_size
        self.node_rank = self.rank // self.local_world_size
        self.local_ranks = list(
            range(
                self.node_rank * self.local_world_size,
                (self.node_rank + 1) * self.local_world_size
            )
        )
        self.local_rank0 = self.local_ranks[0]
        # create local process groups
        for local_rank0 in range(0, self.world_size, self.local_world_size):
            DeviceGroup().get_group(list(range(local_rank0, local_rank0 + self.local_world_size)))

        self.config = config
        self.model_config = model_config
        self.hf_config = hf_config
        self.with_merged = with_merged
        self.load_type = load_type
        self.save_type = save_type
        self.can_generate = can_generate
        log_with_rank(f"load_type: {self.load_type}, save_type: {self.save_type}", rank=self.rank, logger=logger)

    def _broadcast_merged_state_dict(
        self,
        state_dict: Dict[str, Any],
        src_rank: int = 0,
        dst_ranks: Optional[list[int]] = None,
    ):
        """
        Broadcast the merged state dict to all ranks.
        We can't broadcast the whole state_dict at once, because it may be too large, and leads to OOM.
        Here we will break the model and optimizer state_dict into smaller pieces and broadcast them one by one.
        Please note we use `torch.distributed.broadcast_object_list` to broadcast the state_dict (including tensors inside).
        """
        dst_ranks = dst_ranks or list(range(torch.distributed.get_world_size()))
        if src_rank not in dst_ranks or self.rank not in dst_ranks:
            raise ValueError(f"src_rank and current rank must be in dst_ranks: {dst_ranks}")
        pg = DeviceGroup().get_group(dst_ranks)

        if self.rank == src_rank:
            if state_dict is None:
                raise ValueError("state_dict should not be None in rank 0 when broadcasting")
        else:
            if state_dict is not None:
                raise ValueError("state_dict should be None in other ranks when broadcasting")
            state_dict = {}

        def _broadcast_keys(sdict: Dict[str, Any], set_keys=True):
            if self.rank == src_rank:
                state_keys = list(sdict.keys())
            else:
                state_keys = None
            state_key_list = [state_keys]
            torch.distributed.broadcast_object_list(state_key_list, src=src_rank, group=pg)
            state_keys = state_key_list[0]
            if set_keys and self.rank != src_rank:
                for key in state_keys:
                    sdict[key] = {}  # assume the values are empty dicts
            return state_keys

        def _broadcast_value(sdict, key):
            if self.rank == src_rank:
                value_list = [sdict[key]]
            else:
                value_list = [None]
            torch.distributed.broadcast_object_list(value_list, src=src_rank, group=pg)
            if self.rank != src_rank:
                sdict[key] = value_list[0]

        def _broadcast_values(sdict, keys):
            for key in keys:
                _broadcast_value(sdict, key)

        state_keys = _broadcast_keys(state_dict)

        for skey in state_keys:
            logger.info(f"Broadcasting {skey}.")
            if skey == 'optimizer':
                opt_keys = _broadcast_keys(state_dict['optimizer'])
                opt_keys_without_state = [
                    k for k in opt_keys if k != 'state'
                ]
                _broadcast_values(state_dict['optimizer'], opt_keys_without_state)
                idxs = _broadcast_keys(state_dict['optimizer']['state'])
                for idx in idxs:
                    idx_keys = _broadcast_keys(state_dict['optimizer']['state'][idx])
                    _broadcast_values(state_dict['optimizer']['state'][idx], idx_keys)
            elif skey == 'model':
                model_keys = _broadcast_keys(state_dict['model'])
                _broadcast_values(state_dict['model'], model_keys)
            else:
                _broadcast_value(state_dict, skey)
        return state_dict

    def load_checkpoint(self, local_path: str, hdfs_path: str = None, del_local_after_load=False):
        """
        Load an FSDP checkpoint for this rank.

        Downloads and loads:
          - model and optimizer shards
          - extra state dict (scheduler + RNG)

        Args:
            local_path: Directory with per-rank checkpoint files.
            hdfs_path: Unused (for API compatibility).
            del_local_after_load: Remove local files after loading.
        """
        if local_path is None:
            return

        # check if the checkpoint_load_contents is valid
        if self.should_load_model:
            assert self.model is not None, "model must be provided when checkpoint_contents.load includes ['model']"
        if self.should_load_optimizer:
            assert self.optimizer is not None, "optimizer must be provided when checkpoint_contents.load includes ['optimizer']"

        # copy from `_load_checkpoint` in nnscaler/cli/trainer.py
        resume_from = Path(local_path)
        logger.info(f"Resuming from {resume_from}")
        load_from_merged = False

        if resume_from.is_file():
            # when we load from merged checkpoint
            load_from_merged = True
            state_dict = torch.load(resume_from, map_location='cpu', weights_only=False)
        else:
            ckpt_files = list(resume_from.glob('*.ckpt'))
            rank_ckpt_files = {int(f.stem): f for f in ckpt_files if f.stem.isdigit()}
            if set(rank_ckpt_files.keys()) != set(range(len(rank_ckpt_files))):
                raise ValueError(f"Checkpoint files in {resume_from} are not complete: {rank_ckpt_files.keys()}")
            if len(rank_ckpt_files) != self.world_size and self.with_merged is False:
                raise ValueError(f"World size is different with original one: {len(rank_ckpt_files)} != {self.world_size}")

            if len(rank_ckpt_files) != self.world_size or self.with_merged:
                # merge the checkpoint files from all ranks and broadcast to all ranks
                torch.distributed.barrier()
                if self.local_rank == 0:
                    logger.info(f"Merging checkpoint files from {resume_from}")
                    state_dicts = [torch.load(f, map_location='cpu', weights_only=False) for f in rank_ckpt_files.values()]
                    module_state_dict, opt_state_dict = nnscaler.merge_state_dicts(
                        [s['model'] for s in state_dicts],
                        [s['optimizer'] for s in state_dicts]
                    )
                    state_dict = {
                        'model': module_state_dict if self.should_load_model else None,
                        'optimizer': opt_state_dict if self.should_load_optimizer else None,
                    }
                else:
                    state_dict = None

                load_from_merged = True
                logger.info(f"Broadcasting merged checkpoint to all ranks.")
                state_dict = self._broadcast_merged_state_dict(
                    state_dict, src_rank=self.local_rank0, dst_ranks=self.local_ranks
                )
                logger.info(f"Broadcasted merged checkpoint to all ranks.")
            else:
                resume_from = resume_from / f'{self.rank}.ckpt'
                state_dict = torch.load(resume_from, map_location='cpu', weights_only=False)

        model = self.model if self.should_load_model else None
        optimizer = self.optimizer if self.should_load_optimizer else None

        if load_from_merged:
            nnscaler.load_merged_state_dict(
                model, state_dict['model'],
                optimizer, state_dict['optimizer'],
                )
        elif self.load_type == 'sharded':
            nnscaler.load_sharded_state_dict(
                model, state_dict['model'],
                optimizer, state_dict['optimizer'],
            )
        elif self.load_type == 'deduped':
            nnscaler.load_deduped_state_dict(
                model, state_dict['model'],
                optimizer, state_dict['optimizer'],
            )
        else:
            raise ValueError(f"Unknown checkpoint type: {self.load_type}")

        if self.should_load_extra:
            remote_extra_state_path = os.path.join(local_path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")
            local_extra_state_path = copy_to_local(remote_extra_state_path)
            extra_state_dict = torch.load(local_extra_state_path, weights_only=False)
            # recover random state
            if "rng" in extra_state_dict:
                # 'rng' may not exist for backward compatibility
                self.load_rng_state(extra_state_dict["rng"])
                log_with_rank(f"Loaded rng from {remote_extra_state_path}", rank=self.rank, logger=logger)

            lr_scheduler_state_dict = extra_state_dict["lr_scheduler"]
            if lr_scheduler_state_dict is not None and self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(lr_scheduler_state_dict)
                log_with_rank(f"Loaded lr_scheduler from {remote_extra_state_path}", rank=self.rank, logger=logger)

        # wait for everyone to load checkpoints
        torch.distributed.barrier()

    def save_checkpoint(self, local_path: str, hdfs_path: str = None, global_step: int = 0, max_ckpt_to_keep=None):
        """
        Save an ParallelModule checkpoint for this rank.

        Writes:
          - model & optimizer shard files
          - extra state dict (scheduler + RNG)
          - HF tokenizer/processor and model/config on rank 0
          - optional full HF model under 'huggingface/' if requested

        Rotates old checkpoints, keeping at most `max_ckpt_to_keep`.

        Args:
            local_path: Target directory for checkpoint files.
            hdfs_path: Unused (for API compatibility).
            global_step: Current training step (used for bookkeeping).
            max_ckpt_to_keep: Number of recent checkpoints to retain.
        """
        if local_path is None:
            return

        # record the previous global step
        self.previous_global_step = global_step

        # remove previous local_path, only rank 0 should do this
        if self.rank == 0 and max_ckpt_to_keep and isinstance(max_ckpt_to_keep, int) and max_ckpt_to_keep > 0 and len(self.previous_saved_paths) >= max_ckpt_to_keep:
            keep_start = len(self.previous_saved_paths) - max_ckpt_to_keep + 1
            self.remove_previous_save_local_path(self.previous_saved_paths[:keep_start])
            self.previous_saved_paths = self.previous_saved_paths[keep_start:]

        local_path = self.local_mkdir(local_path)
        torch.distributed.barrier()

        # check if the checkpoint_save_contents is valid
        if self.should_save_model:
            assert self.model is not None, "model must be provided when checkpoint_contents.save includes ['model']"
        if self.should_save_optimizer:
            assert self.optimizer is not None, "optimizer must be provided when checkpoint_contents.save includes ['optimizer']"

        # every rank will save its own model and optim shard
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            state_dict_path = os.path.join(local_path, f"{self.rank}.ckpt")
            extra_path = os.path.join(local_path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")

            if self.save_type == 'sharded':
                model_state_dict = self.model.state_dict()
                optimizer_state_dict = self.optimizer.state_dict()
            elif self.save_type == 'deduped':
                model_state_dict, optimizer_state_dict = nnscaler.deduped_state_dict(
                    self.model, self.optimizer
                )
            elif self.save_type == 'merged':
                raise ValueError("merged checkpoint is not supported for saving")
            else:
                raise ValueError(f"Unknown checkpoint type: {self.save_type}")

            if not self.should_save_model:
                model_state_dict = None
            if not self.should_save_optimizer:
                optimizer_state_dict = None

            state_dict = {
                "model": model_state_dict,
                "optimizer": optimizer_state_dict,
            }
            torch.save(state_dict, state_dict_path)
            log_with_rank(f"Saved state_dict to {os.path.abspath(state_dict_path)}", rank=self.rank, logger=logger)

            if self.should_save_extra:
                lr_scheduler_state_dict = self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None
                extra_state_dict = {
                    "lr_scheduler": lr_scheduler_state_dict,
                    "rng": self.get_rng_state(),
                }
                torch.save(extra_state_dict, extra_path)
                log_with_rank(f"Saved extra_state to {os.path.abspath(extra_path)}", rank=self.rank, logger=logger)

        if self.rank == 0:
            model_config = self.hf_config
            if self.can_generate and hasattr(model_config, "name_or_path") and model_config.name_or_path:
                # Some model's name_or_path is empty if not initialized from pretrained,
                # in this cases, we don't save generation config.
                generation_config = GenerationConfig.from_pretrained(model_config.name_or_path)
                generation_config.save_pretrained(local_path)
            else:
                generation_config = None

            model_config.save_pretrained(local_path)
            self.processing_class.save_pretrained(local_path)
            log_with_rank(f"Saved model config and tokenizer class to {os.path.abspath(local_path)}", rank=self.rank, logger=logger, log_only_rank_0=True)

        # wait for everyone to dump to local
        torch.distributed.barrier()

        if self.should_save_hf_model:
            if self.rank == 0:
                # Only rank 0 will save hf model and,
                # offload to cpu to save LLMs which may be too large to fit in one GPU
                state_dict = get_nnscaler_full_state_dict(self.model, offload_to_cpu=True)

                hf_local_path = os.path.join(local_path, "huggingface")
                os.makedirs(hf_local_path, exist_ok=True)

                if "ForTokenClassification" in model_config.architectures[0]:
                    from transformers import AutoModelForTokenClassification

                    auto_model_cls = AutoModelForTokenClassification
                elif "ForCausalLM" in model_config.architectures[0]:
                    from transformers import AutoModelForCausalLM

                    auto_model_cls = AutoModelForCausalLM
                elif "ForConditionalGeneration" in model_config.architectures[0]:
                    from transformers import AutoModelForVision2Seq

                    auto_model_cls = AutoModelForVision2Seq
                else:
                    raise NotImplementedError(f"Unknown architecture {model_config['architectures']}")

                with init_empty_weights():
                    save_model = auto_model_cls.from_config(model_config, torch_dtype=torch.bfloat16)
                save_model.to_empty(device="cpu")

                if save_model.can_generate():
                    if generation_config is not None:
                        save_model.generation_config = generation_config
                    else:
                        print(f"Warning: {self.__class__.__name__}.save_checkpoint: Generation config file not found in, using a generation config created from the model config when saving hf_model.")

                save_model.save_pretrained(hf_local_path, state_dict=state_dict)
                self.processing_class.save_pretrained(hf_local_path)
                log_with_rank(f"Saved hf_model to {os.path.abspath(hf_local_path)}", rank=self.rank, logger=logger, log_only_rank_0=True)
                del state_dict
                del save_model

            # wait for rank0 to dump hf_model to local
            torch.distributed.barrier()

        self.previous_saved_paths.append(local_path)
