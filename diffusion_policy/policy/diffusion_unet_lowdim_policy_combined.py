from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy


class CombinedDiffusionUnetLowdimPolicy(DiffusionUnetLowdimPolicy):
    def __init__(self, 
                 model_A: ConditionalUnet1D,
                 model_B: ConditionalUnet1D,
                 noise_scheduler,  # assuming both models share the same scheduler
                 horizon, 
                 obs_dim, 
                 action_dim, 
                 n_action_steps, 
                 n_obs_steps,
                 num_inference_steps=None,
                 obs_as_local_cond=False,
                 obs_as_global_cond=False,
                 pred_action_steps_only=False,
                 oa_step_convention=False,
                 **kwargs):
        # Initialize using model_A and the scheduler for compatibility.
        super().__init__(model=model_A,
                         noise_scheduler=noise_scheduler,
                         horizon=horizon, 
                         obs_dim=obs_dim, 
                         action_dim=action_dim, 
                         n_action_steps=n_action_steps, 
                         n_obs_steps=n_obs_steps,
                         num_inference_steps=num_inference_steps,
                         obs_as_local_cond=obs_as_local_cond,
                         obs_as_global_cond=obs_as_global_cond,
                         pred_action_steps_only=pred_action_steps_only,
                         oa_step_convention=oa_step_convention,
                         **kwargs)
        # Save both models.
        self.model_A = model_A
        self.model_B = model_B

    def conditional_sample(self, 
                           condition_data, condition_mask,
                           local_cond=None, global_cond=None,
                           generator=None,
                           **kwargs):
        scheduler = self.noise_scheduler

        # Initialize the trajectory as in the original code.
        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # Enforce conditioning:
            trajectory[condition_mask] = condition_data[condition_mask]

            # Get outputs from both models.
            model_output_A = self.model_A(trajectory, t, 
                                          local_cond=local_cond, global_cond=global_cond)
            model_output_B = self.model_B(trajectory, t, 
                                          local_cond=local_cond, global_cond=global_cond)
            # Combine the diffusion scores.
            combined_output = (model_output_A + model_output_B) / 2
            
            # Compute the previous sample using the combined diffusion score.
            trajectory = scheduler.step(
                combined_output, t, trajectory, 
                generator=generator,
                **kwargs
            ).prev_sample
        
        trajectory[condition_mask] = condition_data[condition_mask]        
        return trajectory

