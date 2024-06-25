import gymnasium as gym
import matplotlib.pyplot as plt
from gymnasium.wrappers import PixelObservationWrapper
from wrapper import ResizePixelObservation

# Data collection parameters
num_sequences = 100
sequence_length = 15
image_size = 16

# Define environment
env = ResizePixelObservation(
    PixelObservationWrapper(
        gym.make("Pendulum-v1", g=9.81, render_mode="rgb_array")
    ),
    16,
)
observation, info = env.reset()

# observations = [observation]
# for j in range(sequence_length - 1):
#     action = env.action_space.sample()
#     observation, reward, terminated, truncated, info = env.step(action)

#     if terminated or truncated:
#         observation, info = env.reset()


env.close()
