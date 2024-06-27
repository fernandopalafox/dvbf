import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
import flax.linen as nn
from model import DVBF
import pickle
import time


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
def compute_loss(params, apply_fn, xs, us, rng_key):
    # Forward pass.
    w_means, w_logvars, zs, xs_reconstructed = apply_fn(
        params, xs, us, rngs={"rng_stream": rng_key}
    )

    # Reconstruction loss.
    # Logprobs from isotropic Gaussian observation model.
    logprob_xs = jax.scipy.stats.multivariate_normal.logpdf(
        xs, xs_reconstructed, jnp.eye(obs_dim)
    )
    # Sum over time and batch dimensions.
    reconstruction_loss = jnp.sum(logprob_xs, axis=1)  # check dim

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

    return -(reconstruction_loss - posterior_kl)


def create_train_state(rng_key, model, learning_rate, xs, us):
    """Creates initial `TrainState`."""
    params = model.init(
        {
            "params": rng_key,
            "rng_stream": rng_key,
        },
        xs,
        us,
    )
    tx = optax.adadelta(learning_rate)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def train_step(state, batch, rng_key):
    xs, us = batch

    def loss_fn(params):
        return compute_loss(params, state.apply_fn, xs, us, rng_key)[0]

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


# Parameters
init_key = jax.random.PRNGKey(0)
learning_rate = 0.1
batch_size = 1  # One sequence at a time
data_split = 0.8
num_epochs = 100

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

# Temporary
xs = xs[:50]
us = us[:50]

sequence_length = xs.shape[1]
train_size = int(data_split * xs.shape[0])

xs_train = xs[:train_size]
us_train = us[:train_size]
xs_val = xs[train_size:]
us_val = us[train_size:]

# Load model
model = DVBF(latent_dim, obs_dim, control_dim, num_matrices)
key, subkey = jax.random.split(init_key, 2)
optimizer = optax.adadelta(learning_rate)


# Training loop
train_state = create_train_state(
    subkey, model, learning_rate, xs_train, us_train
)
for epoch in range(num_epochs):
    start_time = time.time()

    key, subkey = jax.random.split(key, 2)
    random_permutation = jax.random.permutation(subkey, train_size)
    xs_permuted = xs_train[random_permutation]
    us_permuted = us_train[random_permutation]

    # Training loss
    total_train_loss = 0
    for i in range(0, train_size, batch_size):
        key, subkey = jax.random.split(key, 2)
        batch = (
            xs_permuted[i : i + batch_size],
            us_permuted[i : i + batch_size],
        )
        train_state, loss = train_step(train_state, batch, subkey)
        total_train_loss += loss

    # Validation loss
    total_val_loss = 0
    for i in range(0, xs_val.shape[0], batch_size):
        key, subkey = jax.random.split(key, 2)
        batch = (xs_val[i : i + batch_size], us_val[i : i + batch_size])
        val_loss = compute_loss(
            train_state.params, train_state.apply_fn, *batch, subkey
        )[0]
        total_val_loss += val_loss

    total_train_loss /= train_size
    total_val_loss /= xs_val.shape[0]

    # Print epoch statistics
    epoch_time = time.time() - start_time
    eta = epoch_time * (num_epochs - epoch - 1)
    eta = time.strftime("%H:%M:%S", time.gmtime(eta))
    print(f"E {epoch + 1} | TL {total_train_loss:.2f} |", end="")
    print(f" VL {total_val_loss:.2f} | t {epoch_time:.2f}s", end="")
    print(f" | ETA: {eta}s")
