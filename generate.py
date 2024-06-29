import jax
import jax.numpy as jnp
from model import DVBF
import pickle
import matplotlib.pyplot as plt

# Simulation parameters
T = 10
key = jax.random.PRNGKey(0)


# Load model and weights
latent_dim = 3
obs_dim = 16**2
control_dim = 1
num_matrices = 16

model = DVBF(
    latent_dim=latent_dim,
    obs_dim=obs_dim,
    control_dim=control_dim,
    num_matrices=num_matrices,
)
with open("data/model_params_epoch_181.pkl", "rb") as f:
    params = pickle.load(f)

# Load data
with open("data/pendulum_data.pkl", "rb") as f:
    states, actions, observations = pickle.load(f)

xs = observations
xs = xs / 255.0  # Normalize to [0, 1]
us = actions

# Draw initial observation x_1
x_1 = xs[jnp.newaxis, jnp.newaxis, 0, 0]
u_1 = us[jnp.newaxis, jnp.newaxis, 0, 0]

# Forward pass to get x_2
key, subkey = jax.random.split(key)
w_1_mean, w_1_logvar, zs, x_2_reconstructed = model.apply(
    params, x_1, u_1, rngs={"rng_stream": subkey}
)

# Compare to ground truth x_{t+1}
x_2_reconstructed = x_2_reconstructed[0, 0]  # (1, 1, 256) -> (256,)
x_2 = xs[0, 1]
x_2_reshaped = x_2.reshape(16, 16)
x_2_reconstructed_reshaped = x_2_reconstructed.reshape(16, 16)

# Plot
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(x_2_reshaped, cmap="gray")
ax[0].set_title("Ground truth")
ax[0].axis("off")
ax[1].imshow(x_2_reconstructed_reshaped, cmap="gray")
ax[1].set_title("Reconstructed")
ax[1].axis("off")

plt.show()
