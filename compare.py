import jax
import jax.numpy as jnp
from model_single import DVBFSingle
import pickle
import matplotlib.pyplot as plt


# CHECK WHETHER LOSS IS CORRECT


def make_loss_fn(model):
    def elbo_loss(params, xs, us, rng_key, c=1.0):
        # Forward pass.
        w_means, w_logvars, zs, xs_reconstructed = model.apply(
            params, xs, us, rngs={"rng_stream": rng_key}
        )

        # Reconstruction loss.
        # Logprobs from isotropic Gaussian observation model.
        logprob_xs = jax.scipy.stats.multivariate_normal.logpdf(
            xs, xs_reconstructed[:, :-1], jnp.eye(obs_dim)
        )
        expected_logprob = jnp.sum(logprob_xs, axis=1)  # sum over time axis.
        reconstruction_loss = -expected_logprob  # flip sign
        annealed_reconstruction_loss = c * reconstruction_loss

        # Temporary: norm squared
        annealed_reconstruction_loss = jnp.linalg.norm(
            xs - xs_reconstructed[:, :-1], axis=-1
        )

        return annealed_reconstruction_loss, xs_reconstructed

    return elbo_loss


# Load model and weights
init_key = jax.random.PRNGKey(0)
latent_dim = 3
obs_dim = 16**2
control_dim = 1
num_matrices = 16
num_plotted_images = 1
data_split = 0.5

model = DVBFSingle(
    latent_dim=latent_dim,
    obs_dim=obs_dim,
    control_dim=control_dim,
    num_matrices=num_matrices,
)
with open("data/model_params_epoch_340.pkl", "rb") as f:
    params = pickle.load(f)

# Load data
with open("data/pendulum_data.pkl", "rb") as f:
    states, actions, observations = pickle.load(f)


xs = observations
xs = xs / 255.0  # Normalize to [0, 1]
us = actions

random_permutation = jax.random.permutation(init_key, xs.shape[0])
xs = xs[random_permutation]
us = us[random_permutation]
xs = xs[:2, :num_plotted_images]
us = us[:2, :num_plotted_images]

sequence_length = xs.shape[1]
train_size = int(data_split * xs.shape[0])

xs_train = xs[:train_size]
us_train = us[:train_size]
xs_val = xs[train_size:]
us_val = us[train_size:]

# Make loss function and forward pass
loss_fn = make_loss_fn(model)

# Compute loss and reconstruction
key, subkey = jax.random.split(init_key)
loss_train, xs_r_train = loss_fn(params, xs_train, us_train, subkey)
key, subkey = jax.random.split(init_key)
loss_val, xs_r_val = loss_fn(params, xs_val, us_val, subkey)

# In one row, plot train and val, in second row plot reconstructions
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0, 0].imshow(xs_train[0, 0].reshape(16, 16), cmap="gray")
axs[0, 0].set_title("Train")
axs[0, 0].axis("off")
axs[0, 1].imshow(xs_val[0, 0].reshape(16, 16), cmap="gray")
axs[0, 1].set_title("Val")
axs[0, 1].axis("off")
axs[1, 0].imshow(xs_r_train[0, 0].reshape(16, 16), cmap="gray")
axs[1, 0].set_title(f"Reconstruction Train\nLoss: {loss_train.mean()}")
axs[1, 0].axis("off")
axs[1, 1].imshow(xs_r_val[0, 0].reshape(16, 16), cmap="gray")
axs[1, 1].set_title(f"Reconstruction Val\nLoss: {loss_val.mean()}")
axs[1, 1].axis("off")

plt.show()
