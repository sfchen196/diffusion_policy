import os
import click
import numpy as np
from tqdm import tqdm
from metaworld import MT1
from metaworld.policies.sawyer_reach_v2_policy import SawyerReachV2Policy
from diffusion_policy.common.replay_buffer import ReplayBuffer

@click.command()
@click.option('-o', '--output', required=True)  # Output file for saving the generated demos
@click.option('-n', '--n_episodes', default=1000)  # Number of episodes to generate
@click.option('-c', '--chunk_length', default=-1)  # Chunk size for saving the buffer

def generate_expert_demos(output, n_episodes=1000, chunk_length=-1):
    # Initialize the MT1 environment for the 'reach-v2' task
    mt1 = MT1('reach-v2', seed=42)
    env = mt1.train_classes['reach-v2']()  # Create the training environment
    env.set_task(mt1.train_tasks[0])  # Set the task instance

    # Initialize the expert policy for 'reach-v2'
    policy = SawyerReachV2Policy()

    # Create an empty replay buffer to store demonstration data
    buffer = ReplayBuffer.create_empty_numpy()

    # Generate demonstrations for the specified number of episodes
    for episode in tqdm(range(n_episodes), desc="Generating Demos"):
        obs_history = []  # Store observations for the episode
        action_history = []  # Store actions for the episode

        # Reset the environment at the start of an episode
        obs, info = env.reset()
        done = False

        while not done:
            # Query the expert policy for an action based on the current observation
            action = policy.get_action(obs)

            # Step the environment with the selected action
            obs, _, _, _, info = env.step(action)

            # Record the observation and action
            obs_history.append(obs)
            action_history.append(action)

            # Check if the task is successfully completed
            done = int(info['success']) == 1

        # Convert observation and action histories to numpy arrays
        obs_history = np.array(obs_history)
        action_history = np.array(action_history)

        # Add the episode data to the replay buffer
        episode_data = {
            'obs': obs_history,
            'action': action_history
        }
        buffer.add_episode(episode_data)

    # Save the replay buffer to the specified output path
    buffer.save_to_path(output, chunk_length=chunk_length)

if __name__ == '__main__':
    generate_expert_demos()
