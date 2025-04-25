"""
A specialized dataset class for block pushing demonstrations that attaches a FiLM conditioning signal
to each trajectory. The conditioning is determined by a dataset_kind string of the form "LABEL1_LABEL2"
(e.g., "b1t1_b1t2" or "b1t1_b2t1") and a split_index that divides the episodes. The first split_index
episodes are labeled with the embedding corresponding to LABEL1 and the remaining episodes with LABEL2.

The mapping from a label (of the form "bXtY") to a 4-dimensional embedding is as follows:
  - "b1t1" -> [1, 0, 1, 0]
  - "b1t2" -> [1, 0, 0, 1]
  - "b2t1" -> [0, 1, 1, 0]
  - "b2t2" -> [0, 1, 0, 1]
"""

from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset

class BlockPushLowdimFilmDataset(BaseLowdimDataset):
    def __init__(self, 
                 zarr_path, 
                 horizon=1,
                 pad_before=0,
                 pad_after=0,
                 obs_key='obs',
                 action_key='action',
                 obs_eef_target=True,
                 use_manual_normalizer=False,
                 seed=42,
                 val_ratio=0.0,
                 # New parameters for FiLM conditioning:
                 dataset_kind: str = None,   # e.g., "b1t1_b1t2", "b1t1_b2t1", or any "bXtY_bWtZ"
                 split_index: int = None     # e.g., 1000, the episode index that divides the two conditions
                 ):
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=[obs_key, action_key])
        
        # Save new conditioning parameters.
        self.dataset_kind = dataset_kind
        self.split_index = split_index
        
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.obs_key = obs_key
        self.action_key = action_key
        self.obs_eef_target = obs_eef_target
        self.use_manual_normalizer = use_manual_normalizer
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = self._sample_to_data(self.replay_buffer)
        normalizer = LinearNormalizer()
        if not self.use_manual_normalizer:
            normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        else:
            x = data['obs']
            stat = {
                'max': np.max(x, axis=0),
                'min': np.min(x, axis=0),
                'mean': np.mean(x, axis=0),
                'std': np.std(x, axis=0)
            }
            is_x = np.zeros(stat['max'].shape, dtype=bool)
            is_y = np.zeros_like(is_x)
            is_x[[0,3,6,8,10,13]] = True
            is_y[[1,4,7,9,11,14]] = True
            is_rot = ~(is_x|is_y)
            def normalizer_with_masks(stat, masks):
                global_scale = np.ones_like(stat['max'])
                global_offset = np.zeros_like(stat['max'])
                for mask in masks:
                    output_max = 1
                    output_min = -1
                    input_max = stat['max'][mask].max()
                    input_min = stat['min'][mask].min()
                    input_range = input_max - input_min
                    scale = (output_max - output_min) / input_range
                    offset = output_min - scale * input_min
                    global_scale[mask] = scale
                    global_offset[mask] = offset
                return SingleFieldLinearNormalizer.create_manual(
                    scale=global_scale,
                    offset=global_offset,
                    input_stats_dict=stat
                )
            normalizer['obs'] = normalizer_with_masks(stat, [is_x, is_y, is_rot])
            normalizer['action'] = SingleFieldLinearNormalizer.create_fit(
                data['action'], last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        obs = sample[self.obs_key]  # shape: (T, D_obs)
        if not self.obs_eef_target:
            obs[:,8:10] = 0
        data = {
            'obs': obs,
            'action': sample[self.action_key],  # shape: (T, D_a)
        }
        return data

    def _film_embedding_from_label(self, label: str) -> np.ndarray:
        """
        Given a label of the form "bXtY", returns a flattened 2x2 film embedding.
        For example:
          "b1t1" -> [1, 0, 1, 0]
          "b1t2" -> [1, 0, 0, 1]
          "b2t1" -> [0, 1, 1, 0]
          "b2t2" -> [0, 1, 0, 1]
        """
        if len(label) < 4 or label[0] != 'b' or label[2] != 't':
            raise ValueError(f"Label {label} does not follow expected format 'bXtY'")
        block_digit = int(label[1])
        target_digit = int(label[3])
        row1 = [1, 0] if block_digit == 1 else [0, 1]
        row2 = [1, 0] if target_digit == 1 else [0, 1]
        return np.array(row1 + row2, dtype=np.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        
        # Determine an episode index.
        try:
            episode_idx = sample.get('episode_index', idx % self.replay_buffer.n_episodes)
        except AttributeError:
            episode_idx = idx % self.replay_buffer.n_episodes
        
        # Attach FiLM conditioning if dataset_kind and split_index are provided.
        if self.dataset_kind is not None and self.split_index is not None:
            parts = self.dataset_kind.split('_')
            if len(parts) != 2:
                raise ValueError("dataset_kind must be of the form 'label1_label2'")
            label1, label2 = parts
            if episode_idx < self.split_index:
                film_cond = self._film_embedding_from_label(label1)
            else:
                film_cond = self._film_embedding_from_label(label2)
            torch_data['film_cond'] = torch.tensor(film_cond, dtype=torch.float32)
        return torch_data
