#!/usr/bin/env python
"""
Usage:
  python generate_combined_demos.py --output path/to/output.zarr \
         --n_episodes 1500 \
         --chunk_length -1 \
         --block1 <block_id_for_b1t1> --zone1 <target1_for_b1t1> \
         --block2 <block_id_for_b1t2> --zone2 <target2_for_b1t2> \
         --block3 <block_id_for_b2t1> --zone3 <target1_for_b2t1>

This script generates a combined dataset from three demo types:
  - Type 1: block1 to target1 (b1t1 demos)
  - Type 2: block1 to target2 (b1t2 demos)
  - Type 3: block2 to target1 (b2t1 demos)

Each type gets n_episodes/3 trajectories. The resulting dataset will be saved
to the specified output path.
"""
if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

import sys
import os
import pathlib
import click
import numpy as np
from tqdm import tqdm

from diffusion_policy.common.replay_buffer import ReplayBuffer
from tf_agents.environments.wrappers import TimeLimit
from tf_agents.environments.gym_wrapper import GymWrapper
from tf_agents.trajectories.time_step import StepType
from diffusion_policy.env.block_pushing.block_pushing_multimodal import BlockPushMultimodal
from diffusion_policy.env.block_pushing.oracles.multimodal_push_oracle_customized import MultimodalOrientedPushOracleCustomized


@click.command()
@click.option('-o', '--output', required=True, help='Output zarr dataset path')
@click.option('-n', '--n_episodes', default=1500, help='Total number of episodes (will be divided equally among three demo types)')
@click.option('-c', '--chunk_length', default=-1, help='Chunk length for saving the dataset')
@click.option('--block1', required=True, help='Block to push for demo type 1 (b1t1)')
@click.option('--zone1', required=True, help='Target zone for demo type 1 (b1t1)')
@click.option('--block2', required=True, help='Block to push for demo type 2 (b1t2)')
@click.option('--zone2', required=True, help='Target zone for demo type 2 (b1t2)')
@click.option('--block3', required=True, help='Block to push for demo type 3 (b2t1)')
@click.option('--zone3', required=True, help='Target zone for demo type 3 (b2t1)')
def main(output, n_episodes, chunk_length, block1, zone1, block2, zone2, block3, zone3):
    # Create an empty replay buffer.
    buffer = ReplayBuffer.create_empty_numpy()

    # Create the environment.
    env = TimeLimit(GymWrapper(BlockPushMultimodal()), duration=350)

    def generate_demos(block, zone, num_eps):
        for i in tqdm(range(num_eps), desc=f"Generating demos for block {block} to zone {zone}"):
            obs_history = []
            action_history = []

            env.seed(i)
            # Create an oracle policy with the desired block and target.
            policy = MultimodalOrientedPushOracleCustomized(env, block=block, target=zone)
            time_step = env.reset()
            policy_state = policy.get_initial_state(1)
            while True:
                action_step = policy.action(time_step, policy_state)
                # Concatenate observations (assumed to be a dict of arrays) along last axis.
                obs = np.concatenate(list(time_step.observation.values()), axis=-1)
                action = action_step.action
                obs_history.append(obs)
                action_history.append(action)

                if time_step.step_type == StepType.LAST:
                    break

                time_step = env.step(action)
                env.render()
            obs_history = np.array(obs_history)
            action_history = np.array(action_history)
            episode = {'obs': obs_history, 'action': action_history}
            buffer.add_episode(episode)

    # Divide total episodes equally among three demo types.
    num_each = n_episodes // 3

    # Generate demos for each type.
    generate_demos(block1, zone1, num_each)  # b1t1 demos
    generate_demos(block2, zone2, num_each)  # b1t2 demos
    generate_demos(block3, zone3, num_each)  # b2t1 demos

    buffer.save_to_path(output, chunk_length=chunk_length)


if __name__ == '__main__':
    main()
