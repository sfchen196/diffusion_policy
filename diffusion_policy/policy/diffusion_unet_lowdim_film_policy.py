"""
File: diffusion_unet_lowdim_film_policy.py

A FiLM-enabled diffusion policy that supports global conditioning via an extra
'film_cond' signal. During training the dataset attaches a film_cond label to
each trajectory. During evaluation rollouts, if no film_cond is provided, the policy
randomly selects one of the two valid embeddings based on its dataset_kind.
During test inference you can force a specific embedding by setting the policy's
forced_embedding_option (or by passing an embedding_option parameter to predict_action).
When forced, the lookup is done using a fixed mapping independent of dataset_kind.

The fixed mapping is:
  "b1t1" -> [1, 0, 1, 0]
  "b1t2" -> [1, 0, 0, 1]
  "b2t1" -> [0, 1, 1, 0]
  "b2t2" -> [0, 1, 0, 1]
"""

from typing import Dict, Optional
import torch
import torch.nn.functional as F
from einops import reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator

def get_random_film_embedding(dataset_kind: str, batch_size: int, device: torch.device) -> torch.Tensor:
    """
    Returns a random FiLM conditioning tensor for the batch (shape: (batch_size,4)).
    For a dataset_kind string of the form "LABEL1_LABEL2", the function randomly picks one of:
      - Embedding for LABEL1 (for example, for "b1t1_b1t2": LABEL1 = "b1t1" -> [1,0,1,0])
      - Embedding for LABEL2 (for example, "b1t1_b1t2": LABEL2 = "b1t2" -> [1,0,0,1])
    """
    # Split the dataset_kind into two labels.
    try:
        label1, label2 = dataset_kind.split('_')
    except Exception as e:
        raise ValueError(f"dataset_kind must be of the form 'label1_label2', got {dataset_kind}") from e
    # Lookup embeddings using the fixed mapping.
    emb1 = get_forced_film_embedding(label1, device)
    emb2 = get_forced_film_embedding(label2, device)
    options = torch.stack([emb1, emb2], dim=0)  # shape: (2,4)
    rand_idx = torch.randint(0, 2, (batch_size,), device=device)
    film_cond = options[rand_idx]  # shape: (batch_size,4)
    return film_cond

def get_forced_film_embedding(option_name: str, device: torch.device) -> torch.Tensor:
    """
    Returns a forced FiLM embedding (4,) for the given option name.
    The mapping is:
      "b1t1" -> [1, 0, 1, 0]
      "b1t2" -> [1, 0, 0, 1]
      "b2t1" -> [0, 1, 1, 0]
      "b2t2" -> [0, 1, 0, 1]
    """
    mapping = {
        "b1t1": [1, 0, 1, 0],
        "b1t2": [1, 0, 0, 1],
        "b2t1": [0, 1, 1, 0],
        "b2t2": [0, 1, 0, 1],
    }
    if option_name not in mapping:
        raise ValueError(f"Unknown forced embedding option: {option_name}")
    return torch.tensor(mapping[option_name], dtype=torch.float32, device=device)

class DiffusionUnetLowdimFilmPolicy(BaseLowdimPolicy):
    def __init__(self, 
                 model: ConditionalUnet1D,
                 noise_scheduler: DDPMScheduler,
                 horizon: int, 
                 obs_dim: int, 
                 action_dim: int, 
                 n_action_steps: int, 
                 n_obs_steps: int,
                 num_inference_steps: int = None,
                 obs_as_local_cond: bool = False,
                 obs_as_global_cond: bool = False,
                 pred_action_steps_only: bool = False,
                 oa_step_convention: bool = False,
                 dataset_kind: Optional[str] = None,
                 **kwargs):
        super().__init__()
        # Do not allow both local and global conditioning.
        assert not (obs_as_local_cond and obs_as_global_cond)
        if pred_action_steps_only:
            assert obs_as_global_cond
        
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_local_cond or obs_as_global_cond) else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()  # Only 'obs' and 'action' are normalized.
        
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_local_cond = obs_as_local_cond
        self.obs_as_global_cond = obs_as_global_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.oa_step_convention = oa_step_convention
        self.dataset_kind = dataset_kind  # Used for random selection when not forced.
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        # The policy will use forced embedding if forced_embedding_option is set.
        self.forced_embedding_option = None
        # Fallback default.
        self.default_film_cond = torch.tensor([0,1,0,1], dtype=torch.float32)

    # ========== Inference-Related Methods ==========

    def conditional_sample(self, 
                           condition_data: torch.Tensor,
                           condition_mask: torch.Tensor,
                           local_cond: torch.Tensor = None,
                           global_cond: torch.Tensor = None,
                           generator=None,
                           **kwargs) -> torch.Tensor:
        """
        Reverse diffusion sampling with optional local/global conditioning.
        """
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator
        )
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            model_output = model(trajectory, t, local_cond=local_cond, global_cond=global_cond)
            trajectory = scheduler.step(model_output, t, trajectory, generator=generator, **kwargs).prev_sample

        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor],
                       embedding_option: Optional[str] = None
                       ) -> Dict[str, torch.Tensor]:
        """
        Expects:
          - obs_dict['obs']: shape (B, T, obs_dim)
        If a forced embedding_option is provided (or if self.forced_embedding_option is set),
        then that forced embedding is used for all samples.
        Otherwise, if no film_cond is provided in obs_dict, a random film_cond is selected based
        on dataset_kind (if set) or the default is used.
        Returns a dictionary with:
          - 'action': (B, n_action_steps, action_dim)
          - 'action_pred': (B, T, action_dim)
        """
        assert 'obs' in obs_dict
        assert 'past_action' not in obs_dict

        # Get film_cond from the obs_dict (if provided).
        film_cond = obs_dict.get('film_cond', None)
        norm_input = {'obs': obs_dict['obs']}
        normed = self.normalizer.normalize(norm_input)
        nobs = normed['obs']  # shape: (B, T, obs_dim)

        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim, f"Expected obs_dim {self.obs_dim}, got {Do}"
        T = self.horizon
        Da = self.action_dim

        device = self.device
        dtype = self.dtype
        nobs = nobs.to(device=device, dtype=dtype)

        local_cond = None
        global_cond = None

        if self.obs_as_global_cond:
            global_cond = nobs[:, :To].reshape(B, -1)
            # Forced option takes precedence.
            if embedding_option is None and self.forced_embedding_option is not None:
                embedding_option = self.forced_embedding_option
            if embedding_option is not None:
                forced_emb = get_forced_film_embedding(embedding_option, device)
                film_cond = forced_emb.unsqueeze(0).expand(B, -1)
            elif film_cond is None:
                if self.dataset_kind is not None:
                    film_cond = get_random_film_embedding(self.dataset_kind, B, device)
                else:
                    film_cond = self.default_film_cond.to(device=device, dtype=dtype).unsqueeze(0).expand(B, -1)
            else:
                if film_cond.ndim == 1:
                    film_cond = film_cond.unsqueeze(0).expand(B, -1)
                film_cond = film_cond.to(device=device, dtype=dtype)
            global_cond = torch.cat([global_cond, film_cond], dim=1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        elif self.obs_as_local_cond:
            local_cond = torch.zeros((B, T, Do), device=device, dtype=dtype)
            local_cond[:, :To] = nobs[:, :To]
            shape = (B, T, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            shape = (B, T, Da + Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs[:, :To]
            cond_mask[:, :To, Da:] = True

        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs
        )

        naction_pred = nsample[..., :Da]
        unnorm_input = {'action': naction_pred}
        unnormed = self.normalizer.unnormalize(unnorm_input)
        action_pred = unnormed['action'].to(device=device, dtype=dtype)

        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To
            if self.oa_step_convention:
                start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:, start:end]

        result = {
            'action': action,
            'action_pred': action_pred
        }
        if not (self.obs_as_local_cond or self.obs_as_global_cond):
            nobs_pred = nsample[..., Da:]
            unnorm_obs_input = {'obs': nobs_pred}
            unnorm_obs = self.normalizer.unnormalize(unnorm_obs_input)
            obs_pred = unnorm_obs['obs'].to(device=device, dtype=dtype)
            action_obs_pred = obs_pred[:, start:end]
            result['action_obs_pred'] = action_obs_pred
            result['obs_pred'] = obs_pred
        return result

    # ========== Training Methods ==========

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Expects:
          - 'obs': (B, T, obs_dim)
          - 'action': (B, T, action_dim)
          - optionally 'film_cond': (B,4)
        """
        film_cond = batch.get('film_cond', None)
        norm_input = {k: v for k, v in batch.items() if k in ['obs', 'action']}
        nbatch = self.normalizer.normalize(norm_input)
        if film_cond is not None:
            nbatch['film_cond'] = film_cond

        obs = nbatch['obs'].to(self.device, self.dtype)
        action = nbatch['action'].to(self.device, self.dtype)

        local_cond = None
        global_cond = None
        trajectory = action

        if self.obs_as_local_cond:
            local_cond = obs
            local_cond[:, self.n_obs_steps:, :] = 0
        elif self.obs_as_global_cond:
            global_cond = obs[:, :self.n_obs_steps, :].reshape(obs.shape[0], -1)
            if 'film_cond' in nbatch:
                fc = nbatch['film_cond'].to(self.device, self.dtype)
                if fc.ndim == 1:
                    fc = fc.unsqueeze(0).expand(obs.shape[0], -1)
                global_cond = torch.cat([global_cond, fc], dim=1)
        else:
            trajectory = torch.cat([action, obs], dim=-1)

        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        noise = torch.randn(trajectory.shape, device=self.device)
        bsz = trajectory.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (bsz,), device=self.device
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        loss_mask = ~condition_mask
        noisy_trajectory[condition_mask] = trajectory[condition_mask]

        pred = self.model(noisy_trajectory, timesteps,
                          local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
