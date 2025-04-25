"""
Usage:
python eval_combined.py --checkpoint_a path/to/checkpoint_a.ckpt \
                        --checkpoint_b path/to/checkpoint_b.ckpt \
                        -o path/to/eval_output \
                        --device cuda:0
"""

import sys
# Use line-buffering for stdout and stderr.
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_lowdim_policy_combined import CombinedDiffusionUnetLowdimPolicy

@click.command()
@click.option('--checkpoint_a', required=True, help="Path to checkpoint A")
@click.option('--checkpoint_b', required=True, help="Path to checkpoint B")
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
def main(checkpoint_a, checkpoint_b, output_dir, device):
    # Create output directory if it doesn't exist.
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint A.
    payload_a = torch.load(open(checkpoint_a, 'rb'), pickle_module=dill)
    cfg_a = payload_a['cfg']
    cls = hydra.utils.get_class(cfg_a._target_)
    workspace_a = cls(cfg_a, output_dir=output_dir)
    workspace_a.load_payload(payload_a, exclude_keys=None, include_keys=None)
    
    # Get model_A from workspace_a.
    model_A_full = workspace_a.model
    if cfg_a.training.use_ema:
        model_A_full = workspace_a.ema_model
    # Unwrap the underlying diffusion model.
    if hasattr(model_A_full, 'model'):
        model_A = model_A_full.model
    else:
        model_A = model_A_full

    # Load checkpoint B.
    payload_b = torch.load(open(checkpoint_b, 'rb'), pickle_module=dill)
    cfg_b = payload_b['cfg']
    # We assume the workspace class is the same for both checkpoints.
    workspace_b = cls(cfg_b, output_dir=output_dir)
    workspace_b.load_payload(payload_b, exclude_keys=None, include_keys=None)
    model_B_full = workspace_b.model
    if cfg_b.training.use_ema:
        model_B_full = workspace_b.ema_model
    # Unwrap the underlying diffusion model.
    if hasattr(model_B_full, 'model'):
        model_B = model_B_full.model
    else:
        model_B = model_B_full

    device = torch.device(device)
    model_A.to(device)
    model_B.to(device)
    model_A.eval()
    model_B.eval()
    
    # Retrieve necessary parameters from workspace_a.model.
    base_policy = workspace_a.model
    noise_scheduler = base_policy.noise_scheduler  # retrieved from the policy
    horizon = base_policy.horizon
    obs_dim = base_policy.obs_dim
    action_dim = base_policy.action_dim
    n_action_steps = base_policy.n_action_steps
    n_obs_steps = base_policy.n_obs_steps
    num_inference_steps = base_policy.num_inference_steps
    obs_as_local_cond = base_policy.obs_as_local_cond
    obs_as_global_cond = base_policy.obs_as_global_cond
    pred_action_steps_only = base_policy.pred_action_steps_only
    oa_step_convention = base_policy.oa_step_convention
    kwargs_policy = base_policy.kwargs

    # Create a combined policy that uses both underlying diffusion models.
    combined_policy = CombinedDiffusionUnetLowdimPolicy(
        model_A=model_A,
        model_B=model_B,
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
        **kwargs_policy
    )
    combined_policy.to(device)
    combined_policy.eval()
    
    # **New step**: Set and transfer the normalizer.
    combined_policy.set_normalizer(workspace_a.model.normalizer)
    # If the normalizer is an nn.Module, move its parameters to device.
    if hasattr(combined_policy.normalizer, "to"):
        combined_policy.normalizer.to(device)
    
    # Instantiate the environment runner using the configuration from cfg_a.
    env_runner = hydra.utils.instantiate(cfg_a.task.env_runner, output_dir=output_dir)
    runner_log = env_runner.run(combined_policy)
    
    # Dump the evaluation log to JSON.
    json_log = {}
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    out_path = os.path.join(output_dir, 'eval_log.json')
    with open(out_path, 'w') as f:
        json.dump(json_log, f, indent=2, sort_keys=True)

if __name__ == '__main__':
    main()
