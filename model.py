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


class TransitionWeights(nn.Module):
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


class DVBF(nn.Module):
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
        recognition = Recognition(self.latent_dim)
        transition_weights = TransitionWeights(
            self.latent_dim, self.control_dim
        )

        # Sample initial network
        w_1_mean, w_1_logvar = initial_network(xs)
        key = self.make_rng("rng_stream")
        w_1 = w_1_mean + jnp.exp(w_1_logvar / 2) * jax.random.normal(
            key, w_1_mean.shape
        )

        # Compute initial transition
        z_1 = initial_transition(w_1)

        # Transition matrices
        As = self.param(
            "A",
            nn.initializers.normal(0.01),
            (self.num_matrices, self.latent_dim, self.latent_dim),
        )
        Bs = self.param(
            "B",
            nn.initializers.normal(0.01),
            (self.num_matrices, self.latent_dim, self.control_dim),
        )
        Cs = self.param(
            "C",
            nn.initializers.normal(0.01),
            (self.num_matrices, self.obs_dim, self.latent_dim),
        )

        # Compute sequence of states
        zs = [z_1]
        xs_reconstructed = []
        for t in range(xs.shape[0] - 1):
            # Compute observation
            x_mean = observation(zs[-1])
            xs_reconstructed.append(x_mean)

            # Compute next latent state
            # Sample stochastic component
            w_mean, w_logvar = recognition(zs[-1], xs[t + 1], us[t])
            key = self.make_rng("rng_stream")
            w = w_mean + jnp.exp(w_logvar / 2) * jax.random.normal(
                key, w_mean.shape
            )
            # Compute deterministic component
            alphas = transition_weights(zs[-1], us[t])
            A_t = jnp.einsum("bi,ijk-> bjk", alphas, As)
            B_t = jnp.einsum("bi,ijk-> bjk", alphas, Bs)
            C_t = jnp.einsum("bi,ijk-> bjk", alphas, Cs)
            # Compute next latent state
            z_t_plus_one = (
                jnp.einsum("bjk, bk -> bj", A_t, zs[-1])
                + jnp.einsum("bjk, bk -> bj", B_t, us[t])
                + jnp.einsum("bjk, bk -> bj", C_t, w)
            )
            zs.append(z_t_plus_one)

            # Compute next observation
            x_mean = observation(z_t_plus_one)

        zs = jnp.vstack(zs)
        xs_reconstructed = jnp.vstack(xs_reconstructed)

        return zs, xs_reconstructed
