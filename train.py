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
from matplotlib.gridspec import GridSpec
from model_single import DVBFSingle


def compute_annealed_kl_divergence(m_q, logvar_q, mean_p, logvar_p, c_i):
    kl = jnp.sum(
        0.5
        * (
            c_i * jnp.log(2 * jnp.pi)
            + c_i * logvar_p
            - jnp.log(2 * jnp.pi)
            - logvar_q
            + c_i
            * (jnp.exp(logvar_q) + (m_q - mean_p) ** 2)
            / jnp.exp(logvar_p)
            - 1
        ),
        axis=-1,
    )
    return -kl


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

        # KL divergence between approximate posterior and prior posterior
        # Assumes diagonal covariance matrices and zero mean prior
        annealed_posterior_kl = jnp.sum(
            compute_annealed_kl_divergence(
                w_means,
                w_logvars,
                jnp.zeros_like(w_means),
                jnp.zeros_like(w_logvars),
                c,
            ),
            axis=1,
        )  # sum over time axis

        return (
            annealed_reconstruction_loss + annealed_posterior_kl,
            annealed_reconstruction_loss,
            annealed_posterior_kl,
        )

    return jax.jit(elbo_loss)


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
        loss, _, _ = elbo_loss(params, xs, us, rng_key, c=c)
        return jnp.mean(loss)  # sgd

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def annealing_scheduler(i, T_a):
    return jnp.min(jnp.array([1.0, 0.01 + i / T_a]))


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
    plt.ylim(bottom=0)

    plt.subplot(2, 1, 2)
    plt.plot(train_recon_losses, "b-", label="Train Recon Loss")
    plt.plot(train_kl_losses, "b.-", label="Train KL Loss")
    plt.plot(val_recon_losses, "r-", label="Val Recon Loss")
    plt.plot(val_kl_losses, "r.-", label="Val KL Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Component Losses")
    plt.title("DVBF Training Progress - Component Losses")
    plt.ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(f"figures/training_progress_detailed.png")
    plt.close()
    print(f"Detailed plot saved as 'training_progress_detailed.png'")


def make_forward_pass_fn(model):
    def forward_pass(params, xs, us, key):
        w_means, w_logvars, zs, xs_reconstructed = model.apply(
            params, xs, us, rngs={"rng_stream": key}
        )
        return w_means, w_logvars, zs, xs_reconstructed

    return jax.jit(forward_pass)


# Parameters
init_key = jax.random.PRNGKey(0)
learning_rate = 0.1
optimizer = optax.adadelta(learning_rate)
batch_size = 500
data_split = 0.5
num_epochs = 5000
c_0 = 0.01
T_a = 10**5
update_interval = 250
max_visible_points = 30
reconstruction_interval = 1
num_plotted_images = 1

latent_dim = 3
obs_dim = 16**2
control_dim = 1
num_matrices = 16

# Load data
with open("data/pendulum_data.pkl", "rb") as f:
    states, actions, observations = pickle.load(f)

xs = observations
# xs = xs / 255.0  # Normalize to [0, 1]
us = actions

# TEMPORARY
random_permutation = jax.random.permutation(init_key, xs.shape[0])
xs = xs[random_permutation]
us = us[random_permutation]
xs = xs[:, :num_plotted_images]
us = us[:, :num_plotted_images]

sequence_length = xs.shape[1]
train_size = int(data_split * xs.shape[0])

xs_train = xs[:train_size]
us_train = us[:train_size]
xs_val = xs[train_size:]
us_val = us[train_size:]

# Load model
model = DVBF(latent_dim, obs_dim, control_dim, num_matrices)

# Training loop
# Set up the plot
plt.ion()  # Turn on interactive mode
fig = plt.figure(figsize=(15, 15))
gs = GridSpec(4, num_plotted_images, figure=fig)

ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, :])

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

image_axes_t = [fig.add_subplot(gs[2, i]) for i in range(num_plotted_images)]
image_axes_r = [fig.add_subplot(gs[3, i]) for i in range(num_plotted_images)]

plt.tight_layout()
for ax in image_axes_t + image_axes_r:
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0 - 0.05, pos.width, pos.height])

train_losses = []
train_recon_losses = []
train_kl_losses = []
val_losses = []
val_recon_losses = []
val_kl_losses = []

# Training loop
key, subkey = jax.random.split(init_key, 2)
train_state = create_train_state(
    subkey, model, learning_rate, xs_train, us_train
)
elbo_loss = make_loss_fn(model)
forward_pass = make_forward_pass_fn(model)
c_i = c_0
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
        epoch_train_loss = 0.0
        epoch_recon_losses = 0.0
        epoch_kl_losses = 0.0
        for i in range(0, train_size, batch_size):
            update_i = epoch * (train_size // batch_size) + i
            if jnp.mod(update_i, update_interval) == 0:
                c_i = annealing_scheduler(update_i, T_a)
            batch = (
                xs_permuted[i : i + batch_size],
                us_permuted[i : i + batch_size],
            )
            old_train_state = train_state
            key, subkey = jax.random.split(key, 2)
            train_state, loss = train_step(train_state, batch, subkey, c=c_i)
            train_loss, train_recon, train_kl = elbo_loss(
                old_train_state.params,
                *batch,
                subkey,
                c=c_i,
            )  # use old state for loss computation? Or new one?
            epoch_train_loss += jnp.mean(train_loss)
            epoch_recon_losses += jnp.mean(train_recon)
            epoch_kl_losses += jnp.mean(train_kl)

        # Validation loss
        epoch_val_losses = 0.0
        epoch_val_recon_losses = 0.0
        epoch_val_kl_losses = 0.0
        for i in range(0, xs_val.shape[0], batch_size):
            batch = (xs_val[i : i + batch_size], us_val[i : i + batch_size])
            key, subkey = jax.random.split(key, 2)
            val_loss, val_recon, val_kl = elbo_loss(
                train_state.params,
                *batch,
                subkey,
                c=c_i,
            )
            epoch_val_losses += jnp.mean(val_loss)
            epoch_val_recon_losses += jnp.mean(val_recon)
            epoch_val_kl_losses += jnp.mean(val_kl)

        num_batches_train = train_size // batch_size
        num_batches_val = xs_val.shape[0] // batch_size
        train_losses.append(epoch_train_loss / num_batches_train)
        train_recon_losses.append(epoch_recon_losses / num_batches_train)
        train_kl_losses.append(epoch_kl_losses / num_batches_train)
        val_losses.append(epoch_val_losses / num_batches_val)
        val_recon_losses.append(epoch_val_recon_losses / num_batches_val)
        val_kl_losses.append(epoch_val_kl_losses / num_batches_val)

        # Reconstruct images and plot
        if epoch % reconstruction_interval == 0:
            selected_batch = 0
            key, subkey = jax.random.split(key, 2)
            w_means, w_logvars, zs, xs_reconstructed = forward_pass(
                train_state.params,
                xs_val[jnp.newaxis, selected_batch],
                us_val[jnp.newaxis, selected_batch],
                subkey,
            )

            xs_truth_reshaped = xs_val[selected_batch].reshape(-1, 16, 16)
            xs_reconstructed_reshaped = xs_reconstructed[
                selected_batch
            ].reshape(-1, 16, 16)
            xs_reconstructed_reshaped = xs_reconstructed_reshaped[:-1]

            variance_norms = jnp.linalg.norm(
                jnp.exp(w_logvars[selected_batch]), axis=-1
            )

            for i, ax in enumerate(image_axes_t):
                if i < num_plotted_images:
                    ax.clear()
                    ax.imshow(
                        xs_truth_reshaped[i],
                        cmap="gray",
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if i == 0:
                        ax.set_ylabel("Ground truth")
            for i, ax in enumerate(image_axes_r):
                if i < num_plotted_images:
                    ax.clear()
                    ax.imshow(
                        xs_reconstructed_reshaped[i],
                        cmap="gray",
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title(
                        f"Var: {variance_norms[i]:.2f}",
                    )
                    if i == 0:
                        ax.set_ylabel("Reconstructed")

        # Update the plot
        epochs = range(1, len(train_losses) + 1)
        line1.set_data(epochs, train_losses)
        line2.set_data(epochs, val_losses)
        line3.set_data(epochs, train_recon_losses)
        line4.set_data(epochs, train_kl_losses)
        line5.set_data(epochs, val_recon_losses)
        line6.set_data(epochs, val_kl_losses)

        # Set the x-axis limits
        if epoch > max_visible_points:
            start_epoch = epoch - max_visible_points
            end_epoch = epoch + 1
        else:
            start_epoch = 0
            end_epoch = epoch + 1

        # Find max y value for visible points
        max_y_ax1 = max(
            max(train_losses[start_epoch:end_epoch]),
            max(val_losses[start_epoch:end_epoch]),
        )
        max_y_ax2 = max(
            max(train_recon_losses[start_epoch:end_epoch]),
            max(train_kl_losses[start_epoch:end_epoch]),
            max(val_recon_losses[start_epoch:end_epoch]),
            max(val_kl_losses[start_epoch:end_epoch]),
        )
        min_y_ax2 = min(
            min(train_recon_losses[start_epoch:end_epoch]),
            min(train_kl_losses[start_epoch:end_epoch]),
            min(val_recon_losses[start_epoch:end_epoch]),
            min(val_kl_losses[start_epoch:end_epoch]),
        )
        ylims = (0.0, max_y_ax1 * 1.10, min_y_ax2 * 1.10, max_y_ax2 * 1.10)

        for i, ax in enumerate((ax1, ax2)):
            ax.set_xlim(start_epoch, end_epoch)
            ax.set_ylim(ylims[i * 2], ylims[i * 2 + 1])
            ax.relim()
            ax.autoscale_view()

        fig.canvas.draw()
        fig.canvas.flush_events()

        # Print epoch statistics
        epoch_time = time.time() - start_time
        eta = epoch_time * (num_epochs - epoch - 1)
        eta = time.strftime("%H:%M:%S", time.gmtime(eta))
        print(f"E {epoch + 1} | TL {train_losses[-1]:.2f} |", end="")
        print(
            f" VL {val_losses[-1]:.2f} | t {epoch_time:.2f}s",
            end="",
        )
        print(f" | c {c_i:.2f}", end="")
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
