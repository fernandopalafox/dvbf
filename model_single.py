import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
import time


class InitialNetwork(nn.Module):
    latent_dim: int

    @nn.compact
    def __call__(self, xs):
        # Simple LSTM network
        lstm_output = nn.Bidirectional(
            nn.RNN(nn.OptimizedLSTMCell(128)),
            nn.RNN(nn.OptimizedLSTMCell(128)),
        )(xs)

        # Output mean and logvar for w_1_init
        out = nn.Dense(128)(lstm_output[:, -1])
        out = nn.relu(out)
        params = nn.Dense(2 * self.latent_dim)(out)
        mean, logvar = jnp.split(params, 2, axis=-1)
        return mean, logvar


class InitialTransition(nn.Module):
    latent_dim: int

    @nn.compact
    def __call__(self, w_1_init):
        # 128 ReLU + latent_dim output
        out = nn.Dense(128)(w_1_init)
        out = nn.relu(out)
        z_1 = nn.Dense(self.latent_dim)(out)
        return z_1


class Observation(nn.Module):
    obs_dim: int

    @nn.compact
    def __call__(self, z):
        # 128 ReLU + 16^2 identity output
        x = nn.Dense(128)(z)
        x = nn.relu(x)
        mean = nn.Dense(self.obs_dim)(x)
        return mean


class Recognition(nn.Module):
    latent_dim: int

    @nn.compact
    def __call__(self, z_t, x_t_plus_one, u_t):
        # 128 ReLU + 2 * latent_dim output
        out = jnp.concatenate([z_t, x_t_plus_one, u_t], axis=-1)
        out = nn.Dense(128)(out)
        out = nn.relu(out)
        params = nn.Dense(2 * self.latent_dim)(out)
        w_mean, w_logvar = jnp.split(params, 2, axis=-1)
        return w_mean, w_logvar


class DVBFSingle(nn.Module):
    latent_dim: int
    obs_dim: int
    control_dim: int
    num_matrices: int

    @nn.compact
    def __call__(self, xs, us):
        # Initialize model components
        initial_network = InitialNetwork(self.latent_dim)
        initial_transition = InitialTransition(self.latent_dim)
        observation = Observation(self.obs_dim)

        # Sample initial network
        w_mean_init, w_logvar_init = initial_network(xs)
        key = self.make_rng("rng_stream")
        w_1_init = w_mean_init + jnp.exp(
            0.5 * w_logvar_init
        ) * jax.random.normal(key, w_mean_init.shape)

        # Compute initial transition
        z_1 = initial_transition(w_1_init)
        x_1 = observation(z_1)

        # Compute sequence of states
        zs = [z_1]
        xs_reconstructed = [observation(z_1), observation(z_1)]
        w_means = [w_mean_init]
        w_logvars = [w_logvar_init]

        w_means = jnp.stack(w_means, axis=1)
        w_logvars = jnp.stack(w_logvars, axis=1)
        zs = jnp.stack(zs, axis=1)
        xs_reconstructed = jnp.stack(xs_reconstructed, axis=1)

        return w_means, w_logvars, zs, xs_reconstructed
