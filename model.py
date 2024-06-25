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
        lstm_output = nn.RNN(nn.OptimizedLSTMCell(128))(xs)

        # Output mean and logvar for w_1
        out = nn.Dense(128)(lstm_output[:, -1])
        out = nn.relu(out)
        params = nn.Dense(2 * self.latent_dim)(out)
        mean, logvar = jnp.split(params, 2, axis=-1)
        return mean, logvar


class InitialTransition(nn.Module):
    latent_dim: int

    @nn.compact
    def __call__(self, w_1):
        # 128 ReLU + latent_dim output
        out = nn.Dense(128)(w_1)
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


class TransitionWeights(nn.Module):
    latent_dim: int
    num_matrices: int

    @nn.compact
    def __call__(self, z_t, u_t):
        # 16 softmax output
        out = jnp.concatenate([z_t, u_t], axis=-1)
        alphas = nn.Dense(
            self.num_matrices,
            use_bias=False,
            kernel_init=nn.initializers.normal(0.01),
        )(out)
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
            (self.num_matrices, self.latent_dim, self.latent_dim),
        )

        # Compute sequence of states
        zs = [z_1]
        xs_reconstructed = []
        for t in range(xs.shape[1] - 1):
            zs_t = zs[-1]
            xs_t_plus_one = xs[:, t + 1]
            us_t = us[:, t]

            # Compute next latent state
            # Sample stochastic component
            w_mean, w_logvar = recognition(zs_t, xs_t_plus_one, us_t)
            key = self.make_rng("rng_stream")
            w_t = w_mean + jnp.exp(w_logvar / 2) * jax.random.normal(
                key, w_mean.shape
            )
            # Compute deterministic component
            alphas = transition_weights(zs_t, us_t)
            A_t = jnp.einsum("bi,ijk-> bjk", alphas, As)
            B_t = jnp.einsum("bi,ijk-> bjk", alphas, Bs)
            C_t = jnp.einsum("bi,ijk-> bjk", alphas, Cs)
            # Compute next latent state
            z_t_plus_one = (
                jnp.einsum("bjk, bk -> bj", A_t, zs_t)
                + jnp.einsum("bjk, bk -> bj", B_t, us_t)
                + jnp.einsum("bjk, bk -> bj", C_t, w_t)
            )
            zs.append(z_t_plus_one)

            # Compute next observation
            x_mean = observation(z_t_plus_one)
            xs_reconstructed.append(x_mean)

        zs = jnp.stack(zs)
        xs_reconstructed = jnp.stack(xs_reconstructed)

        return zs, xs_reconstructed


# Example usage
num_batches = 10
latent_dim = 3
obs_dim = 16**2
control_dim = 1
num_matrices = 4
sequence_length = 10
rng_key = jax.random.PRNGKey(0)


# Initialize model
start_time = time.time()
model = DVBF(latent_dim, obs_dim, control_dim, num_matrices)
key, subkey = jax.random.split(rng_key, 2)
xs = jax.random.normal(key, (num_batches, sequence_length, obs_dim))
us = jax.random.normal(key, (num_batches, sequence_length - 1, control_dim))
params = model.init(
    {
        "params": key,
        "rng_stream": key,
    },
    xs,
    us,
)
print("Initialization time:", time.time() - start_time)

# Forward pass
start_time = time.time()
key_forward, subkey = jax.random.split(subkey)
zs, xs_reconstructed = model.apply(
    params, xs, us, rngs={"rng_stream": key_forward}
)
print("Forward pass time:", time.time() - start_time)

print("zs.shape:", zs.shape)
print("xs_reconstructed.shape:", xs_reconstructed.shape)
