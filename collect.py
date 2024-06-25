import gymnasium as gym
import matplotlib.pyplot as plt
from gymnasium.wrappers import PixelObservationWrapper
from wrapper import ResizePixelObservation
import jax
from jax import numpy as jnp
import pickle

# Data collection parameters
num_sequences = 100
sequence_length = 15
image_size = 16

# Define environment
env = ResizePixelObservation(
    PixelObservationWrapper(
        gym.make("Pendulum-v1", g=9.81, render_mode="rgb_array")
    ),
    64,
)

# Collect data
observations = []
actions = []
for _ in range(num_sequences):
    observation, info = env.reset()
    batch_actions = []
    batch_observations = [observation["pixels"]]
    for _ in range(sequence_length - 1):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

        batch_observations.append(observation["pixels"])
        batch_actions.append(action)

    observations.append(jnp.stack(batch_observations))
    actions.append(jnp.array(batch_actions))
env.close()

# Data dimensions (num_sequences, sequence_length, image_size, image_size, 3)
observations = jnp.stack(observations)
actions.append(jnp.array(action))

# Save data
with open("data/pendulum_data.pkl", "wb") as f:
    pickle.dump((observations, actions), f)
