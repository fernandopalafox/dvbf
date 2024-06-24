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
        output = lstm_state.hidden

        # Output mean and logvar for w_1
        x = nn.Dense(128)(output)
        x = nn.relu(x)
        params = nn.Dense(2 * self.latent_dim)(x)
        mean, logvar = jnp.split(params, 2, axis=-1)
        return mean, jnp.exp(logvar)


class Observation(nn.Module):
    obs_dim: int

    @nn.compact
    def __call__(self, z):
        # 128 ReLU + 16^2 identity output
        x = nn.Dense(128)(z)
        x = nn.relu(x)
        mean = nn.Dense(self.obs_dim)(x)
        return mean
