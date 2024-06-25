import gymnasium as gym
import matplotlib.pyplot as plt
from gymnasium.wrappers import PixelObservationWrapper
from gymnasium import spaces
from wrapper import ResizePixelObservation
from pendulum import PendulumEnv
import jax
from jax import numpy as jnp
import pickle

# Data collection parameters
num_sequences = 1
sequence_length = 50
image_size = 128

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
        action = env.action_space.sample()

        # Simple feedback control
        # State is [cos(theta), sin(theta), theta_dot]
        theta = jnp.arctan2(observation["state"][1], observation["state"][0])
        theta_dot = observation["state"][2]
        position_gain = 100.0
        velocity_gain = 10.0
        action = -position_gain * theta - velocity_gain * theta_dot

        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

        batch_state.append(observation["state"])
        batch_actions.append(action)
        batch_observations.append(observation["pixels"])

    states.append(jnp.stack(batch_state))
    actions.append(jnp.array(batch_actions))
    observations.append(jnp.stack(batch_observations))
env.close()

# Data dimensions (num_sequences, sequence_length, image_size, image_size, 3)
states = jnp.stack(states)
actions = jnp.stack(actions)
observations = jnp.stack(observations)

# Save data
with open("data/pendulum_data.pkl", "wb") as f:
    pickle.dump((states, actions, observations), f)


# Select a single sequence
from matplotlib.animation import FuncAnimation
import numpy as np

# Select a single sequence
sequence_index = 0
sequence = observations[sequence_index]
sequence_actions = actions[sequence_index]
sequence_states = states[sequence_index]

# Create the figure and subplots
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(10, 20))
fig.suptitle("Pendulum Animation with Control Values and State Components")

# Initialize the image plot
im = ax1.imshow(sequence[0], cmap="gray")
ax1.axis("off")

# Initialize the control plot
time = np.arange(len(sequence_actions) + 1)
(line_control,) = ax2.plot(time[:-1], sequence_actions, "b-")
ax2.set_xlim(0, len(sequence_actions))
ax2.set_ylim(
    min(sequence_actions).item() - 0.1, max(sequence_actions).item() + 0.1
)
ax2.set_xlabel("Time Step")
ax2.set_ylabel("Control Value")
ax2.set_title("Control Values Over Time")

# Initialize the state component plots
(line_x,) = ax3.plot(time, sequence_states[:, 0], "r-")
ax3.set_xlim(0, len(sequence_states) - 1)
ax3.set_ylim(
    min(sequence_states[:, 0]) - 0.1, max(sequence_states[:, 0]) + 0.1
)
ax3.set_ylabel("X Coordinate")
ax3.set_title("Pendulum Free End X Coordinate")

(line_y,) = ax4.plot(time, sequence_states[:, 1], "g-")
ax4.set_xlim(0, len(sequence_states) - 1)
ax4.set_ylim(
    min(sequence_states[:, 1]) - 0.1, max(sequence_states[:, 1]) + 0.1
)
ax4.set_ylabel("Y Coordinate")
ax4.set_title("Pendulum Free End Y Coordinate")

(line_angular_vel,) = ax5.plot(time, sequence_states[:, 2], "m-")
ax5.set_xlim(0, len(sequence_states) - 1)
ax5.set_ylim(
    min(sequence_states[:, 2]) - 0.1, max(sequence_states[:, 2]) + 0.1
)
ax5.set_xlabel("Time Step")
ax5.set_ylabel("Angular Velocity")
ax5.set_title("Pendulum Angular Velocity")


# Animation update function
def update(frame):
    im.set_array(sequence[frame])
    line_control.set_data(time[: frame + 1], sequence_actions[: frame + 1])
    line_x.set_data(time[: frame + 1], sequence_states[: frame + 1, 0])
    line_y.set_data(time[: frame + 1], sequence_states[: frame + 1, 1])
    line_angular_vel.set_data(
        time[: frame + 1], sequence_states[: frame + 1, 2]
    )
    return im, line_control, line_x, line_y, line_angular_vel


# Create the animation
anim = FuncAnimation(
    fig, update, frames=len(sequence), interval=100, blit=True
)

# Display the animation
plt.tight_layout()
plt.show()
