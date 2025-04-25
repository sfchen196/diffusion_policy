if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

import os
import click
import pathlib
import numpy as np
from tqdm import tqdm
from diffusion_policy.common.replay_buffer import ReplayBuffer
from tf_agents.environments.wrappers import TimeLimit
from tf_agents.environments.gym_wrapper import GymWrapper
from tf_agents.trajectories.time_step import StepType
from diffusion_policy.env.block_pushing.block_pushing_multimodal_fixed import BlockPushMultimodalFixed
from diffusion_policy.env.block_pushing.oracles.multimodal_push_oracle_customized import MultimodalOrientedPushOracleCustomized

@click.command()
@click.option('-o', '--output', required=True)
@click.option('-n', '--n_episodes', default=1000)
@click.option('-c', '--chunk_length', default=-1)
@click.option('--block_a', required=True, help='Specify the block to push for demo type A.')
@click.option('--zone_a', required=True, help='Specify the target zone for demo type A.')
@click.option('--block_b', required=True, help='Specify the block to push for demo type B.')
@click.option('--zone_b', required=True, help='Specify the target zone for demo type B.')
def main(output, n_episodes, chunk_length, block_a, zone_a, block_b, zone_b):
    buffer = ReplayBuffer.create_empty_numpy()
    env = TimeLimit(GymWrapper(BlockPushMultimodalFixed()), duration=350)
    
    def generate_demos(block, zone, num_episodes):
        for i in tqdm(range(num_episodes), desc=f"Generating demos for block {block} to zone {zone}"):
            obs_history = list()
            action_history = list()

            env.seed(i)
            policy = MultimodalOrientedPushOracleCustomized(env, block=block, target=zone)
            time_step = env.reset()
            policy_state = policy.get_initial_state(1)
            while True:
                action_step = policy.action(time_step, policy_state)
                obs = np.concatenate(list(time_step.observation.values()), axis=-1)
                action = action_step.action
                obs_history.append(obs)
                action_history.append(action)

                if time_step.step_type == 2:
                    break

                time_step = env.step(action)
                env.render()
            obs_history = np.array(obs_history)
            action_history = np.array(action_history)

            episode = {
                'obs': obs_history,
                'action': action_history
            }
            buffer.add_episode(episode)
    
    # Generate demos for type A
    generate_demos(block_a, zone_a, n_episodes // 2)
    
    # Generate demos for type B
    generate_demos(block_b, zone_b, n_episodes // 2)
    
    buffer.save_to_path(output, chunk_length=chunk_length)
        
if __name__ == '__main__':
    main()
