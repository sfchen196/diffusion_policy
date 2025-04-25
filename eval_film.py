"""
Usage:
python eval_film.py --checkpoint path/to/checkpoint.ckpt \
                    --output_dir path/to/eval_output \
                    --device cuda:0 \
                    --dataset_kind b1t1_b2t1 \
                    --embedding_option b2t2
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

@click.command()
@click.option('--checkpoint', required=True, help="Path to checkpoint")
@click.option('--output_dir', required=True, help="Output directory for evaluation logs")
@click.option('--device', default='cuda:0', help="Device to use (e.g., cuda:0)")
@click.option('--dataset_kind', required=True, help="Dataset kind (e.g., b1t1_b2t1 or b1t1_b1t2)")
@click.option('--embedding_option', required=True, help="Forced embedding option (e.g., b2t2)")
def main(checkpoint, output_dir, device, dataset_kind, embedding_option):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint using dill.
    with open(checkpoint, 'rb') as f:
        payload = torch.load(f, pickle_module=dill)
    cfg = payload['cfg']
    
    # Instantiate workspace.
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # Get policy; if using EMA, use that.
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    device = torch.device(device)
    policy.to(device)
    policy.eval()
    
    # Inject dataset_kind and force the embedding option.
    policy.dataset_kind = dataset_kind
    policy.forced_embedding_option = embedding_option
    
    # Now simply run the environment runner. It will call predict_action internally,
    # and predict_action will use the forced embedding because forced_embedding_option is set.
    env_runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=output_dir)
    runner_log = env_runner.run(policy)
    
    # Log evaluation results.
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
