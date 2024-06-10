import jax
from jax import numpy as jnp

# jax.config.update("jax_disable_jit", True)

# Scenario:
# 2 1-D agents. Human's control only cares about maintaining a minimum distance
# from the robot. Robot wants to minimize distance to origin, but would also
# like to minimize the uncertainty in the human's behavior. Human is to the
# left of the robot, and the origin is to the robots right. Therefore, the
# robot must balance between probing the human to the left and moving to the
# right.


# Simulation parameters
# TODO: Move these to a separate file
thetas = jnp.array([[-10.0], [0.0], [10.0], [20.0]])
b_0 = jnp.array([1 / len(thetas) for _ in thetas])

assert sum(b_0) == 1.0, "Initial belief must sum to 1.0"

# System dynamics: 1D double integrator.
# Two agents: robot and human.
x_0 = jnp.array([-2.0, 0.0, -4.0, 0.0])
dt = 0.1
w_scale = 0.5
A_single = jnp.array([[1.0, dt], [0.0, 1.0]])
B_single = jnp.array([[0.5 * dt**2], [dt]])
A = jnp.block([[A_single, jnp.zeros((2, 2))], [jnp.zeros((2, 2)), A_single]])
B = jnp.block([[B_single, jnp.zeros((2, 1))], [jnp.zeros((2, 1)), B_single]])
Sigma = w_scale * jnp.eye(4)


# Cost
def stage_cost(x_t, u_t):
    u_R = u_t[0]
    return 0.0
    # return jnp.dot(u_R, u_R)


def terminal_cost(x_T):
    x_R_T = x_T[0]
    return jnp.dot(x_R_T, x_R_T)


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
    return (1 - λ) * eval_J_r(E_x, E_u) + λ * eval_J_i(E_x, E_u, b_0)


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
    min_distance = theta[0]
    x_R_t = x_t[0]
    x_H_t = x_t[2]
    distance_squared = x_R_t - x_H_t
    k = 10.0
    # return jnp.array(
    #     [
    #         1
    #         / (1 + jnp.exp(k * (distance_squared - min_distance**2)))
    #         * jnp.tanh(x_H_t - x_R_t)
    #     ]
    # )

    return theta


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
    - u_R: robot controls (u_0, u_1, ..., u_{t-1})
    - theta: human control parameters

    Outputs:
    - x: state trajectory (x_0, x_1, ..., x_{t})
    """

    def step_dynamics_scan(x_t, u_R_t):
        u_H_t = eval_u_H_t(x_t, theta)
        u_t = jnp.concatenate([u_R_t, u_H_t])
        x_t_plus_1 = step_dynamics(x_t, u_t)
        return x_t_plus_1, x_t_plus_1

    x_final, x = jax.lax.scan(step_dynamics_scan, x_0, u_R)
    return jnp.stack([x_0, *x])


def eval_E_x(
    x_0: jnp.ndarray, u_R: jnp.ndarray, b_0: jnp.ndarray, theta=None
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
        if theta is None:
            E_u_H_t = eval_E_u_H(x_t, b_0)
        else:
            E_u_H_t = eval_u_H(x_t, theta)
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

    Note: Could be more efficient by saving the deterministic trajectories
    """
    t = len(x) - 1

    # Base case
    if t == 0:
        return b_0

    # Recursive case
    else:
        b_t_minus_1 = eval_b_t(x[:-1], u[:-1], b_0)
        u_R = u[:, 0]  # TODO make this more general for higher dim inputs
        if u_R.ndim == 1:
            u_R = u_R[:, jnp.newaxis]
        xs = jax.vmap(eval_x, in_axes=(None, None, 0))(x[0], u_R, thetas)
        u_Hs = jax.vmap(eval_u_H, in_axes=(0, 0))(
            xs, thetas
        )  # deterministic controls u_1, u_2, ..., u_{t}
        us = jax.vmap(lambda u_H: jnp.concatenate([u_R, u_H[:-1]], axis=1))(
            u_Hs
        )
        b_t = jax.vmap(
            lambda b_theta, x_theta, u_theta: b_theta
            * transition_model(x[-1], x_theta, u_theta)
        )(
            b_t_minus_1, xs, us
        )  # b_t = b_{t-1} * p(x_t | x_{0:t-1}, u_{0:t-1}, θ)
        b_t = b_t / jnp.sum(b_t)
        return b_t


def transition_model(x_query: jnp.ndarray, x: jnp.ndarray, u: jnp.ndarray):
    """Return the probability of transitioning to x_query given x and u

    Assumes a linear system with additive Gaussian noise.

    Inputs:
    - x_query: state to query the pdf at
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
    Sigma_t = sum(
        [
            jnp.linalg.matrix_power(A, j)
            @ Sigma
            @ (jnp.linalg.matrix_power(A, j)).T
            for j in range(t)
        ]
    )
    k = len(mu_t)

    def gaussian_pdf(x):
        return (
            1
            / jnp.sqrt((2 * jnp.pi) ** k * jnp.linalg.det(Sigma_t))
            * jnp.exp(
                -0.5
                * jnp.dot(
                    jnp.dot((x - mu_t), jnp.linalg.inv(Sigma_t)), (x - mu_t)
                )
            )
        )

    return gaussian_pdf(x_query)


# Sim parameters
horizon = 6  # time horizon. Control inputs are u_0, u_1, ..., u_{horizon-1}
u_R = jnp.array([[0.0] for _ in range(horizon)])  # init guess for u_R
λ = 1.0
learning_rate = 0.1
descent_steps = 200

num_points = 30
axis_limit = 50.0
plot_interval = descent_steps // 20
epsilon_cost = 0.0  # so that track is above surface
max_marker_size = 200
params_true = thetas[2]

# Gradient descent
eval_E_J_jit = jax.jit(eval_E_J)


def grad_descent_scan(u_R, _):
    grad = jax.grad(eval_E_J_jit, argnums=1)(x_0, u_R, b_0, λ)
    u_R_new = u_R - learning_rate * grad
    return u_R_new, (u_R_new, jnp.linalg.norm(grad))


_, descent_trajectory = jax.lax.scan(
    grad_descent_scan, u_R, jnp.arange(descent_steps)
)
u_R_descent = descent_trajectory[0]
u_R_descent = jnp.concatenate([u_R[jnp.newaxis, :, :], u_R_descent], axis=0)
J_descent = jax.vmap(eval_E_J_jit, in_axes=(None, 0, None, None))(
    x_0, u_R_descent, b_0, λ
)
x_descent = jax.jit(eval_x)(x_0, u_R_descent[-1], params_true)
u_H_descent = eval_u_H(x_descent[:-1], params_true)
u_descent = jnp.concatenate([u_R_descent[-1], u_H_descent], axis=1)
b_descent = [
    eval_b_t(x_descent[: t + 1], u_descent[:t], b_0)
    for t in range(horizon + 1)
]

# Visualization
import matplotlib.pyplot as plt


# Gradient descent norm
fig, ax = plt.subplots(1, 1)
ax.plot(descent_trajectory[1])
ax.set_xlabel("Descent step")
ax.set_ylabel("Gradient norm")
fig.savefig("figures/grad_norm.png")

# Robot and human position
# Subplot 1: Robot and human positions
# Subplot 2: Robot and human controls
# Subplot 3: Belief over time

# x-axis position, y-axis time
time_vec = jnp.arange(horizon + 1) * dt
fig, ax = plt.subplots(3, 1)
ax[0].plot(time_vec, x_descent[:, 0], label="Robot")
ax[0].plot(time_vec, x_descent[:, 2], label="Human")
ax[0].set_ylabel("Position [m]")
ax[0].legend()
ax[0].tick_params(
    axis="x",
    which="both",
    bottom=False,
    top=False,
    labelbottom=False,
)

ax[1].plot(time_vec[:-1], u_R_descent[-1], label="Robot")
ax[1].plot(time_vec[:-1], eval_u_H(x_descent[:-1], params_true), label="Human")
ax[1].set_ylabel("Control [m/s^2]")
ax[1].tick_params(
    axis="x",
    which="both",
    bottom=False,
    top=False,
    labelbottom=False,
)

for t, belief in zip(time_vec, b_descent):
    for i, belief_value in enumerate(belief):
        ax[2].scatter(
            t,
            i,
            s=40.0,
            alpha=belief_value.item(),
            color="r",
        )
ax[2].set_yticks(range(len(thetas)))
ax[2].set_ylabel("Parameter index")
ax[2].set_xlabel("Time [s]")

fig.savefig("figures/robot_human.png")

# Plot gradient descent on cost landscape (only works for time horizon of 1)
if len(u_R) > 1:
    raise ValueError("Plotting only works for time horizon of 1")

# Surface
u_axis = jnp.linspace(-axis_limit, axis_limit, num_points)
u1, u2 = jnp.meshgrid(u_axis, u_axis)
u_R_grid = jnp.stack([u1, u2], axis=-1)
u_R_grid = u_R_grid.reshape(-1, 2)
J_grid = jax.vmap(eval_E_J_jit, in_axes=(None, 0, None, None))(
    x_0, u_R_grid[:, :, jnp.newaxis], b_0, λ
)
J_grid = J_grid.reshape(num_points, num_points)

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
fig.savefig("figures/grad_descent.png")
