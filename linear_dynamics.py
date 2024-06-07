import jax
from jax import numpy as jnp

# jax.config.update("jax_disable_jit", True)

# Simulation parameters
N = 10
thetas = jnp.array([0.1, 0.2])
b_0 = jnp.array([0.01, 0.99])

# System dynamics: 1D double integrator.
# Two agents: robot and human.
x_0 = jnp.array([-1.0, 0.0, 1.0, 0.0])
dt = 0.1
A_single = jnp.array([[1.0, dt], [0.0, 1.0]])
B_single = jnp.array([[0.5 * dt**2], [dt]])
A = jnp.block([[A_single, jnp.zeros((2, 2))], [jnp.zeros((2, 2)), A_single]])
B = jnp.block([[B_single, jnp.zeros((2, 1))], [jnp.zeros((2, 1)), B_single]])
sigma = jnp.eye(4)


# Cost
def stage_cost(x_t, u_t):
    Q = jnp.eye(4)
    return jnp.dot(jnp.dot(x_t, Q), x_t)


def terminal_cost(x_T):
    return jnp.dot(x_T, x_T)


def step_dynamics(x_t: jnp.ndarray, u_t: jnp.ndarray) -> jnp.ndarray:
    return A @ x_t + B @ u_t


def eval_J_r(x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
    """Evaluate the robot cost given initial state and robot controls
    Inputs:
    - x: state trajectory (x_0, x_1, ..., x_{T})
    - u: control trajectory (u_0, u_1, ..., u_{T-1})

    Outputs:
    - J_r: robot cost
    """

    def stage_cost_scan(stage_cost_t, xu_t):
        return stage_cost_t + stage_cost(xu_t[0], xu_t[1]), None

    return (
        terminal_cost(x[-1])
        + jax.lax.scan(stage_cost_scan, 0.0, (x[:-1], u))[0]
    )


def eval_H(b: jnp.ndarray) -> jnp.ndarray:
    """Shannon entropy of a belief vector b"""
    return -jnp.sum(b * jnp.log(b))


def eval_J_i(x: jnp.ndarray, u: jnp.ndarray, b_0: jnp.ndarray) -> jnp.ndarray:
    """Evaluate the information cost given the state and control trajectories
    and initial belief.
    Inputs:
    - x: state trajectory (x_0, x_1, ..., x_{T})
    - u: control trajectory (u_0, u_1, ..., u_{T-1})
    - b_0: initial belief

    Outputs:
    - J_i: entropy of the belief at time T
    """
    return eval_H(eval_b_t(x, u, b_0))


def eval_E_J(
    x_0: jnp.ndarray, u_R: jnp.ndarray, b_0: jnp.ndarray, λ: float
) -> jnp.ndarray:
    """Evaluate the expected cost given initial state, initial belief,
    and robot controls
    Inputs:
    - x_0: initial state
    - u_R: robot controls (u_0, u_1, ..., u_{T-1})
    - b_0: initial belief
    - λ: curiosity parameter

    Outputs:
    - E_J: expected cost

    Note: This is the cost of the expected stat and control trajectories
    """
    E_x = eval_E_x(x_0, u_R, b_0)
    E_u_H = eval_E_u_H(E_x, thetas)
    E_u = jnp.concatenate([u_R, E_u_H[:-1]], axis=1)
    return eval_J_r(E_x, E_u) + λ * eval_J_i(E_x, E_u, b_0)


def eval_u_H(x: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    """Returns human controls given states and a vector of parameters
    theta

    Inputs:
    - x: state trajectory (x_0, x_1, ..., x_{t})
    - theta: human control parameters

    Outputs:
    - u_H: human controls (u_H_0, u_H_1, ..., u_H_{t})

    """
    if len(x.shape) == 1:
        return eval_u_H_t(x, theta)
    else:
        return jnp.array([eval_u_H_t(x_t, theta) for x_t in x])


def eval_u_H_t(x_t: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    """Returns a human at time t given a state and a vector of
    parameters theta
    """
    return jnp.array([theta])


def eval_E_u_H(x: jnp.ndarray, b_0: jnp.ndarray) -> jnp.ndarray:
    """Returns the expected human controls given states and a belief

    Inputs:
    - x: state trajectory (x_0, x_1, ..., x_{t})
    - b_0: initial belief

    Outputs:
    - E_u_H: expected human controls (E[u_H_0], E[u_H_1], ..., E[u_H_{t}])
    """

    def scan_expected_u_H(u_so_far, params_and_belief):
        return (
            u_so_far
            + eval_u_H(x, params_and_belief[0]) * params_and_belief[1],
            None,
        )

    return jax.lax.scan(
        scan_expected_u_H,
        eval_u_H(x, thetas[0]) * b_0[0],
        (thetas[1:], b_0[1:]),
    )[0]


def eval_x(
    x_0: jnp.ndarray, u_R: jnp.ndarray, theta: jnp.ndarray
) -> jnp.ndarray:
    """Compute the state trajectory given init. state, parameters, and robot
    controls

    Inputs:
    - x_0: initial state
    - theta: human control parameters
    - u_R: robot controls (u_0, u_1, ..., u_{t-1})

    Outputs:
    - x: state trajectory (x_0, x_1, ..., x_{t})
    """

    def step_dynamics_scan(x_t, u_t):
        x_t_plus_1 = step_dynamics(x_t, u_t)
        return x_t_plus_1, x_t_plus_1

    x_final, x = jax.lax.scan(step_dynamics_scan, x_0, u_R)
    return jnp.stack([x_0, *x])


def eval_E_x(
    x_0: jnp.ndarray, u_R: jnp.ndarray, b_0: jnp.ndarray
) -> jnp.ndarray:
    """Compute expected states given init. state, init. belief, and robot
    controls
    Inputs:
    - x_0: state at time 0
    - b_0: belief at time 0
    - u_R: set of robot controls (u_0, u_1, ..., u_{t-1})

    Outputs:
    - E_x: expected states (x_0, E[x_1], ..., E[x_{t}])
    """

    def step_dynamics_scan(x_t, u_R_t):
        E_u_H_t = eval_E_u_H(x_t, b_0)
        E_u_t = jnp.concatenate([u_R_t, E_u_H_t])
        x_t_plus_1 = step_dynamics(x_t, E_u_t)
        return x_t_plus_1, x_t_plus_1

    E_x_final, E_x = jax.lax.scan(step_dynamics_scan, x_0, u_R)
    return jnp.stack([x_0, *E_x])


def eval_b_t(x: jnp.ndarray, u: jnp.ndarray, b_0: jnp.ndarray) -> jnp.ndarray:
    """Returns the belief at time index t given the initial belief b_0.
    Inputs:
    - x: state trajectory (x_0, x_1, ..., x_{t})
    - u: control trajectory (u_0, u_1, ..., u_{t-1})
    - b_0: initial belief

    Outputs:
    - b_k: belief at time t
    """
    t = len(x) - 1

    # Base case
    if t == 1:
        return b_0

    # Recursive case
    else:
        b_t_minus_1 = eval_b_t(x[:-1], u[:-1], b_0)
        transition_model = make_transition_model(x, u)
        b_t = jnp.array(
            [
                b_t_minus_1[l] * transition_model(x[-1])
                for l, _ in enumerate(thetas)
            ]
        )
        return b_t


def make_transition_model(x: jnp.ndarray, u: jnp.ndarray):
    """Return the state transition model at time t.

    Assumes a linear system with additive Gaussian noise.

    Inputs:
    - x: state trajectory (x_0, x_1, ..., x_{t})
    - u: control trajectory (u_0, u_1, ..., u_{t-1})

    Outputs:
    - Gaussian pdf for x_t given x_{0:t-1}
    """
    # Expected mean and covariance at time t
    t = len(x) - 1
    mu_t = jnp.linalg.matrix_power(A, t) @ x[0] + sum(
        [jnp.linalg.matrix_power(A, t - 1 - j) @ B @ u[j] for j in range(t)]
    )
    sigma_t = sum(
        [
            jnp.linalg.matrix_power(A, j)
            @ sigma
            @ (jnp.linalg.matrix_power(A, j)).T
            for j in range(t)
        ]
    )
    k = len(mu_t)

    def gaussian_pdf(x):
        return (
            1
            / jnp.sqrt((2 * jnp.pi) ** k * jnp.linalg.det(sigma_t))
            * jnp.exp(
                -0.5
                * jnp.dot(
                    jnp.dot((x - mu_t), jnp.linalg.inv(sigma_t)), (x - mu_t)
                )
            )
        )

    return gaussian_pdf


# Test cost functions and gradients

# Sim parameters
u_R = jnp.array([[50.0], [10.0]])
λ = 0.5
num_points = 30
axis_limit = 50.0
learning_rate = 1.0
descent_steps = 100
plot_interval = descent_steps // 20
epsilon_cost = 0.0  # so that track is above surface

# 3D plot of the cost function
eval_E_J_jit = jax.jit(eval_E_J)

# Surface
u_axis = jnp.linspace(-axis_limit, axis_limit, num_points)
u1, u2 = jnp.meshgrid(u_axis, u_axis)
u_R_grid = jnp.stack([u1, u2], axis=-1)
u_R_grid = u_R_grid.reshape(-1, 2)
J_grid = jax.vmap(eval_E_J_jit, in_axes=(None, 0, None, None))(
    x_0, u_R_grid[:, :, jnp.newaxis], b_0, λ
)
J_grid = J_grid.reshape(num_points, num_points)


# Gradient descent
def grad_descent_scan(u_R, _):
    grad = jax.grad(eval_E_J_jit, argnums=1)(x_0, u_R, b_0, λ)
    u_R_new = u_R - learning_rate * grad
    return u_R_new, u_R_new


_, u_R_descent = jax.lax.scan(
    grad_descent_scan, u_R, jnp.arange(descent_steps)
)
u_R_descent = jnp.concatenate([u_R[jnp.newaxis, :, :], u_R_descent], axis=0)
J_descent = jax.vmap(eval_E_J_jit, in_axes=(None, 0, None, None))(
    x_0, u_R_descent, b_0, λ
)

import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.plot_surface(u1, u2, J_grid, cmap="viridis", linewidth=0.1, zorder=1)
ax.plot(
    u_R_descent[::plot_interval, 0].flatten(),
    u_R_descent[::plot_interval, 1].flatten(),
    J_descent[::plot_interval] + epsilon_cost,
    "-ro",
    markersize=5,
    zorder=4,
    label="Gradient descent",
)
ax.set_xlabel("u1")
ax.set_ylabel("u2")
ax.set_zlabel("J")
ax.title.set_text("Cost as a function of robot controls")
plt.show()
