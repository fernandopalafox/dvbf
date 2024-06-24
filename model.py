import jax
import jax.numpy as jnp
import flax
from flax import linen as nn


class InitialNetwork(nn.Module):
    latent_dim: int

    @nn.compact
    def __call__(self, xs):
        # Simple LSTM network
        xs = jnp.transpose(xs, (1, 0, 2))  # (T, B, D)
        lstm = nn.OptimizedLSTMCell(128)
        _, lstm_state = nn.RNN(lstm)(xs)
        lstm_output = lstm_state.hidden

        # Output mean and logvar for w_1
        input = nn.Dense(128)(lstm_output)
        input = nn.relu(input)
        params = nn.Dense(2 * self.latent_dim)(input)
        mean, logvar = jnp.split(params, 2, axis=-1)
        return mean, logvar


class InitialTransition(nn.Module):
    latent_dim: int

    @nn.compact
    def __call__(self, w_1):
        # 128 ReLU + latent_dim output
        input = nn.Dense(128)(w_1)
        input = nn.relu(input)
        z_1 = nn.Dense(self.latent_dim)(input)
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
        input = jnp.concatenate([z_t, x_t_plus_one, u_t], axis=-1)
        input = nn.Dense(128)(input)
        input = nn.relu(input)
        params = nn.Dense(2 * self.latent_dim)(input)
        w_mean, w_logvar = jnp.split(params, 2, axis=-1)
        return w_mean, w_logvar


class Transition(nn.Module):
    latent_dim: int
    num_matrices: int

    @nn.compact
    def __call__(self, z_t, u_t):
        # 16 softmax output
        input = jnp.concatenate([z_t, u_t], axis=-1)
        alphas = nn.Dense(
            self.num_matrices,
            use_bias=False,
            kernel_init=nn.initializers.normal(0.01),
        )(input)
        alphas = nn.softmax(alphas)
        return alphas
