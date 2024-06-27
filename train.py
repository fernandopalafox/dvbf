import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
import flax.linen as nn
from model import DVBF
import pickle

# Parameters
init_key = jax.random.PRNGKey(0)
learning_rate = 0.1
batch_size = 1
set_sizes = 500

latent_dim = 3
obs_dim = 16**2
control_dim = 1
num_matrices = 16

# Load data
with open("data/pendulum_data.pkl", "rb") as f:
    states, actions, observations = pickle.load(f)

xs = observations
xs = xs / 255.0 - 0.5  # Normalize to [-0.5, 0.5]
us = actions

sequence_length = xs.shape[1]

xs_train = xs[:set_sizes]
us_train = us[:set_sizes]

xs_val = xs[set_sizes:]
us_val = us[set_sizes:]

# Load model
model = DVBF(latent_dim, obs_dim, control_dim, num_matrices)
key, subkey = jax.random.split(init_key, 2)
params = model.init(
    {
        "params": subkey,
        "rng_stream": subkey,
    },
    jax.random.normal(subkey, (batch_size, sequence_length, obs_dim)),
    jax.random.normal(subkey, (batch_size, sequence_length, control_dim)),
)


def compute_kl_divergence(mean1, logvar1, mean2, logvar2):
    """KL divergence between two Gaussian distributions.
    Assumes diagonal covariance matrices for both distributions.
    """
    kl = 0.5 * (
        jnp.sum(logvar2 - logvar1, axis=-1)
        + jnp.sum(jnp.exp(logvar1 - logvar2), axis=-1)
        + jnp.sum((mean2 - mean1) ** 2 / jnp.exp(logvar2), axis=-1)
        - mean1.shape[-1]
    )
    return kl


# Compute loss
def loss(params, model, xs, us, rng_key):
    # Forward pass.
    w_means, w_logvars, zs, xs_reconstructed = model.apply(
        params, xs, us, rngs={"rng_stream": rng_key}
    )

    # Reconstruction loss.
    # Logprobs from isotropic Gaussian observation model.
    logprob_xs = jax.scipy.stats.multivariate_normal.logpdf(
        xs, xs_reconstructed, jnp.eye(obs_dim)
    )
    # Sum over time and batch dimensions.
    reconstruction_loss = -jnp.sum(logprob_xs, axis=1)  # check dim

    # KL divergence between approximate posterior and prior posterior
    # Assumes diagonal covariance matrices and zero mean prior
    posterior_kl = jnp.sum(
        compute_kl_divergence(
            w_means,
            w_logvars,
            jnp.zeros_like(w_means),
            jnp.zeros_like(w_logvars),
        )
    )

    return reconstruction_loss - posterior_kl


# TEMPORARY
key, subkey = jax.random.split(key, 2)
test_loss = loss(
    params, model, xs_train[jnp.newaxis, 0], us_train[jnp.newaxis, 0], subkey
)


def create_train_state(key, model, learning_rate, x_seq, u_seq):
    """Creates initial `TrainState`."""
    params = model.init(key, x_seq, u_seq)
    tx = optax.adadelta(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx
    )
