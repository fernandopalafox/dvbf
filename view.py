from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt
import pickle


def view_sequence(i, states, actions, reshaped_observations):

    num_sequences = states.shape[0]
    sequence_length = states.shape[1]
    image_size = int(np.sqrt(reshaped_observations.shape[-1]))

    # Revert reshape
    observations = reshaped_observations.reshape(
        num_sequences, sequence_length, image_size, image_size, 1
    )

    # Select a single sequence
    sequence_index = i
    sequence = observations[sequence_index]
    sequence_actions = actions[sequence_index]
    sequence_states = states[sequence_index]

    # Compute angle from x and y coordinates
    angles = np.arctan2(sequence_states[:, 1], sequence_states[:, 0])

    # Create the figure and subplots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 16))

    # Initialize the image plot
    im = ax1.imshow(sequence[0], cmap="gray")
    ax1.axis("off")

    # Initialize the control plot
    time = np.arange(len(sequence_actions))
    (line_control,) = ax2.plot(time, sequence_actions, "b-")
    ax2.set_xlim(0, len(sequence_actions))
    ax2.set_ylim(
        min(sequence_actions).item() - 0.1, max(sequence_actions).item() + 0.1
    )
    ax2.set_ylabel("Control Value")
    ax2.set_title("Control")

    # Initialize the angle plot
    (line_angle,) = ax3.plot(time, angles, "r-")
    ax3.set_xlim(0, len(angles) - 1)
    ax3.set_ylim(min(angles) - 0.1, max(angles) + 0.1)
    ax3.set_ylabel("Angle (radians)")
    ax3.set_title("Angle")

    # Initialize the angular velocity plot
    (line_angular_vel,) = ax4.plot(time, sequence_states[:, 2], "m-")
    ax4.set_xlim(0, len(sequence_states) - 1)
    ax4.set_ylim(
        min(sequence_states[:, 2]) - 0.1, max(sequence_states[:, 2]) + 0.1
    )
    ax4.set_xlabel("Time Step")
    ax4.set_ylabel("Angular Velocity")
    ax4.set_title("Angular Velocity")

    # Animation update function
    def update(frame):
        im.set_array(sequence[frame])
        line_control.set_data(time[: frame + 1], sequence_actions[: frame + 1])
        line_angle.set_data(time[: frame + 1], angles[: frame + 1])
        line_angular_vel.set_data(
            time[: frame + 1], sequence_states[: frame + 1, 2]
        )
        return im, line_control, line_angle, line_angular_vel

    # Create the animation
    anim = FuncAnimation(
        fig, update, frames=len(sequence), interval=100, blit=True
    )

    # Display the animation
    plt.tight_layout()
    plt.show()


def view_grid(xs, title):
    """Create a grid of images from reshaped observations."""
    num_sequences = xs.shape[0]
    sequence_length = xs.shape[1]
    image_size = int(np.sqrt(xs.shape[-1]))

    # Revert reshape
    observations = xs.reshape(
        num_sequences, sequence_length, image_size, image_size
    )

    # Create the figure
    fig, axs = plt.subplots(
        num_sequences,
        sequence_length,
        figsize=(sequence_length, num_sequences),
    )

    # Plot each image
    for i in range(num_sequences):
        for j in range(sequence_length):
            plt.subplot(
                num_sequences, sequence_length, i * sequence_length + j + 1
            )
            plt.imshow(observations[i, j], cmap="gray")
            plt.axis("off")

    # Display the figure
    plt.tight_layout()
    plt.show()

    # Save the figure
    fig.savefig("figures/" + title + ".png")


# # Load data
# with open("data/pendulum_data.pkl", "rb") as f:
#     states, actions, reshaped_observations = pickle.load(f)

# # View a single sequence
# view_sequence(1, states, actions, reshaped_observations)
