import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
import flax.linen as nn
from model import DVBF
import pickle
import time
import matplotlib.pyplot as plt
import signal


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


def compute_annealed_kl_divergence(mean1, logvar1, mean2, logvar2, c):
    kl = jnp.sum(
        c * 1 / 2 * jnp.log(2 * jnp.pi)
        + c * 1 / 2 * logvar2
        - 1 / 2 * jnp.log(2 * jnp.pi)
        - 1 / 2 * logvar1
        + c
        * (jnp.exp(logvar1) + (mean1 - mean2) ** 2)
        / (2 * jnp.exp(logvar2))
        - 1 / 2,
        axis=-1,
    )
    return kl


# Compute loss
def compute_loss(params, apply_fn, xs, us, rng_key, separate=False, c=1.0):
    # Forward pass.
    w_means, w_logvars, zs, xs_reconstructed = apply_fn(
        params, xs, us, rngs={"rng_stream": rng_key}
    )

    # Reconstruction loss.
    # Logprobs from isotropic Gaussian observation model.
    logprob_xs = jax.scipy.stats.multivariate_normal.logpdf(
        xs, xs_reconstructed, jnp.eye(obs_dim)
    )
    reconstruction_loss = jnp.sum(logprob_xs, axis=1)  # sum over time axis
    reconstruction_loss = -c * reconstruction_loss  # anneal and flip

    # KL divergence between approximate posterior and prior posterior
    # Assumes diagonal covariance matrices and zero mean prior
    posterior_kl = jnp.sum(
        compute_annealed_kl_divergence(
            w_means,
            w_logvars,
            jnp.zeros_like(w_means),
            jnp.zeros_like(w_logvars),
            c,
        ),
        axis=1,
    )  # sum over time axis

    if separate:
        return (
            reconstruction_loss + posterior_kl,
            reconstruction_loss,
            posterior_kl,
        )
    else:
        return reconstruction_loss + posterior_kl


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
def train_step(state, batch, rng_key, c=1.0):
    xs, us = batch

    def loss_fn(params):
        return compute_loss(params, state.apply_fn, xs, us, rng_key, c=c)[0]

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def annealing_scheduler(i, T_a):
    # return jnp.min(jnp.array([1.0, 0.01 + i / T_a]))
    return 1.0


# Signal handler
continue_training = True


def signal_handler(sig, frame):
    global continue_training
    print("\nInterrupt received. Stopping training and saving model...")
    continue_training = False


# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)


# Function to save the model
def save_model(train_state, epoch):
    with open(f"data/model_params_epoch_{epoch}.pkl", "wb") as f:
        pickle.dump(train_state.params, f)
    print(f"Model saved at epoch {epoch}")


# Function to save the plot
def save_plot(
    train_losses,
    train_recon_losses,
    train_kl_losses,
    val_losses,
    val_recon_losses,
    val_kl_losses,
):
    plt.figure(figsize=(12, 16))

    plt.subplot(2, 1, 1)
    plt.plot(train_losses, "b-", label="Train Loss")
    plt.plot(val_losses, "r-", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Total Loss")
    plt.title("DVBF Training Progress - Total Loss")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(train_recon_losses, "b-", label="Train Recon Loss")
    plt.plot(train_kl_losses, "b.-", label="Train KL Loss")
    plt.plot(val_recon_losses, "r-", label="Val Recon Loss")
    plt.plot(val_kl_losses, "r.-", label="Val KL Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Component Losses")
    plt.title("DVBF Training Progress - Component Losses")
    plt.tight_layout()
    plt.savefig(f"figures/training_progress_detailed.png")
    plt.close()
    print(f"Detailed plot saved as 'training_progress_detailed.png'")


# Parameters
init_key = jax.random.PRNGKey(0)
learning_rate = 0.1
batch_size = 1  # One sequence at a time
data_split = 0.9
num_epochs = 500
c_0 = 0.01
T_a = 10**5

latent_dim = 3
obs_dim = 16**2
control_dim = 1
num_matrices = 16

# Load data
with open("data/pendulum_data.pkl", "rb") as f:
    states, actions, observations = pickle.load(f)

xs = observations
xs = xs / 255.0  # Normalize to [0, 1]
us = actions

# TEMPORARY: try to overfit a small dataset
xs = xs[:2]
ys = us[:2]

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
# Set up the plot
plt.ion()  # Turn on interactive mode
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
(line1,) = ax1.plot([], [], "b-", label="Train Loss")
(line2,) = ax1.plot([], [], "r-", label="Validation Loss")
ax1.set_ylabel("Total Loss")
ax1.set_title("DVBF Training Progress")
ax1.legend()

(line3,) = ax2.plot([], [], "b-", label="Train Recon Loss")
(line4,) = ax2.plot([], [], "b.-", label="Train KL Loss")
(line5,) = ax2.plot([], [], "r-", label="Val Recon Loss")
(line6,) = ax2.plot([], [], "r.-", label="Val KL loss")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Component Losses")
ax2.legend()

train_losses = []
train_recon_losses = []
train_kl_losses = []
val_losses = []
val_recon_losses = []
val_kl_losses = []

# Training loop
train_state = create_train_state(
    subkey, model, learning_rate, xs_train, us_train
)


try:
    for epoch in range(num_epochs):
        if not continue_training:
            break

        start_time = time.time()

        key, subkey = jax.random.split(key, 2)
        random_permutation = jax.random.permutation(subkey, train_size)
        xs_permuted = xs_train[random_permutation]
        us_permuted = us_train[random_permutation]

        # Training loss
        total_train_loss = 0
        total_train_recon_loss = 0
        total_train_kl_loss = 0
        c_i = 0
        for i in range(0, train_size, batch_size):
            key, subkey = jax.random.split(key, 2)
            update_i = epoch * (train_size // batch_size) + i
            c_i = annealing_scheduler(update_i, T_a)
            batch = (
                xs_permuted[i : i + batch_size],
                us_permuted[i : i + batch_size],
            )
            old_train_state = train_state
            train_state, loss = train_step(train_state, batch, subkey, c=c_i)
            train_loss, train_recon, train_kl = compute_loss(
                old_train_state.params,
                old_train_state.apply_fn,
                *batch,
                subkey,
                separate=True,
                c=c_i,
            )  # use old state for loss computation? Or new one?
            total_train_loss += loss
            total_train_recon_loss += train_recon[0]
            total_train_kl_loss += train_kl[0]

        # Validation loss
        total_val_loss = 0
        total_val_recon_loss = 0
        total_val_kl_loss = 0
        for i in range(0, xs_val.shape[0], batch_size):
            key, subkey = jax.random.split(key, 2)
            batch = (xs_val[i : i + batch_size], us_val[i : i + batch_size])
            val_loss, val_recon, val_kl = compute_loss(
                train_state.params,
                train_state.apply_fn,
                *batch,
                subkey,
                separate=True,
                c=c_i,
            )

            total_val_loss += val_loss[0]
            total_val_recon_loss += val_recon[0]
            total_val_kl_loss += val_kl[0]

        total_train_loss /= train_size
        total_train_recon_loss /= train_size
        total_train_kl_loss /= train_size
        total_val_loss /= xs_val.shape[0]
        total_val_recon_loss /= xs_val.shape[0]
        total_val_kl_loss /= xs_val.shape[0]

        train_losses.append(total_train_loss)
        train_recon_losses.append(total_train_recon_loss)
        train_kl_losses.append(total_train_kl_loss)
        val_losses.append(total_val_loss)
        val_recon_losses.append(total_val_recon_loss)
        val_kl_losses.append(total_val_kl_loss)

        # Update the plot
        epochs = range(1, len(train_losses) + 1)
        line1.set_data(epochs, train_losses)
        line2.set_data(epochs, val_losses)
        line3.set_data(epochs, train_recon_losses)
        line4.set_data(epochs, train_kl_losses)
        line5.set_data(epochs, val_recon_losses)
        line6.set_data(epochs, val_kl_losses)
        for ax in (ax1, ax2):
            ax.relim()
            ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()

        # Print epoch statistics
        epoch_time = time.time() - start_time
        eta = epoch_time * (num_epochs - epoch - 1)
        eta = time.strftime("%H:%M:%S", time.gmtime(eta))
        print(f"E {epoch + 1} | TL {total_train_loss:.2f} |", end="")
        print(f" VL {total_val_loss:.2f} | t {epoch_time:.2f}s", end="")
        print(f" | ETA: {eta}s")

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Save the model and plot regardless of how the training ends
    save_model(train_state, epoch)
    save_plot(
        train_losses,
        train_recon_losses,
        train_kl_losses,
        val_losses,
        val_recon_losses,
        val_kl_losses,
    )


plt.ioff()
plt.show()

print("Training complete or interrupted. Model and plot saved.")
