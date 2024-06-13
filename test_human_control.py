import jax
from jax import numpy as jnp
import matplotlib.pyplot as plt


sharpness_sigm = 2.0
sharpness_tanh = 20.0
min_distance = 1.0
control_scale = 1.0
x_H_t = 0.0

limits = 5.0


def sigmoid(x):
    return 1 / (1 + jnp.exp(-sharpness_sigm * x))


def f(x_R_t):
    diff = x_R_t - x_H_t
    return (
        sigmoid(diff + min_distance)
        * sigmoid(-diff + min_distance)
        * -jnp.tanh(sharpness_tanh * diff)
    )


x_R_t = jnp.linspace(-limits, limits, 100)

plt.plot(x_R_t, jax.vmap(f)(x_R_t))
plt.vlines(x_H_t, -1, 1, colors="r")
plt.vlines(x_H_t + min_distance, -1, 1, colors="r", linestyles="--")
plt.vlines(x_H_t - min_distance, -1, 1, colors="r", linestyles="--")
plt.xlabel("x_R_t")
plt.ylabel("f(x_R_t)")
plt.show()
