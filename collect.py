import gymnasium as gym
import matplotlib.pyplot as plt
from gymnasium.wrappers import PixelObservationWrapper
from gymnasium import spaces
from wrapper import ResizePixelObservation
from pendulum import PendulumEnv
import jax
from jax import numpy as jnp
import pickle
from view import view_sequence

# Data collection parameters
num_sequences = 1000
sequence_length = 50
image_size = 16  # 16x16 image

# Define environment
env = ResizePixelObservation(
    PixelObservationWrapper(
        PendulumEnv(g=9.81, render_mode="rgb_array"), pixels_only=False
    ),
    image_size,
)

# Collect data
states = []
actions = []
observations = []
for _ in range(num_sequences):
    observation, info = env.reset()
    batch_state = [observation["state"]]
    batch_actions = []
    batch_observations = [observation["pixels"]]
    for _ in range(sequence_length - 1):
        # Simple feedback control
        # State is [cos(theta), sin(theta), theta_dot]
        # theta = jnp.arctan2(observation["state"][1], observation["state"][0])
        # theta_dot = observation["state"][2]
        # position_gain = 100.0
        # velocity_gain = 1.0
        # action = -position_gain * theta - velocity_gain * theta_dot

        # sample action
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action[0])

        if terminated or truncated:
            observation, info = env.reset()

        batch_state.append(observation["state"])
        batch_actions.append(action)
        batch_observations.append(observation["pixels"])

    batch_actions.append(env.action_space.sample())  # Dummy action

    states.append(jnp.stack(batch_state))
    actions.append(jnp.array(batch_actions))
    observations.append(jnp.stack(batch_observations))
env.close()

# Data dimensions (num_sequences, sequence_length, image_size, image_size, 1)
states = jnp.stack(states)
actions = jnp.stack(actions)
observations = jnp.stack(observations)

# Reshape observations to (num_sequences, sequence_length, image_size**2)
observations_reshaped = observations.reshape(
    num_sequences, sequence_length, image_size**2
)

# Save data
with open("data/pendulum_data.pkl", "wb") as f:
    pickle.dump((states, actions, observations_reshaped), f)

# Visualize data
view_sequence(0, states, actions, observations_reshaped)
