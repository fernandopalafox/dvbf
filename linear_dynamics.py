import jax
import optax

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
thetas = jnp.array([[0.1], [1.0], [10.0]])
b_init = jnp.array([1 / len(thetas) for _ in thetas])

assert sum(b_init) == 1.0, "Initial belief must sum to 1.0"

# System dynamics: 1D double integrator.
# Two agents: robot and human.
x_0 = jnp.array([-1.0, 0.0, -3.0, 0.0])
dt = 1.0
w_scale = 0.5
A_single = jnp.array([[1.0, dt], [0.0, 1.0]])
B_single = jnp.array([[0.5 * dt**2], [dt]])
A = jnp.block([[A_single, jnp.zeros((2, 2))], [jnp.zeros((2, 2)), A_single]])
B = jnp.block([[B_single, jnp.zeros((2, 1))], [jnp.zeros((2, 1)), B_single]])
Sigma = w_scale * jnp.eye(len(x_0))


# Cost
@jax.jit
def stage_cost(x_t, u_t):
    u_R_t = u_t[0]
    x_R_t = x_t[0]
    return jnp.dot(u_R_t, u_R_t)


@jax.jit
def terminal_cost(x_T):
    x_R_T = x_T[0]
    return jnp.dot(x_R_T, x_R_T)


@jax.jit
def step_dynamics(x_t: jnp.ndarray, u_t: jnp.ndarray, w_t=None) -> jnp.ndarray:
    if w_t is None:
        return A @ x_t + B @ u_t
    else:
        return A @ x_t + B @ u_t + w_t


@jax.jit
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


@jax.jit
def eval_H(b: jnp.ndarray) -> jnp.ndarray:
    """Shannon entropy of a belief vector b"""
    return -jnp.sum(b * jnp.log(b + 1e-8))  # avoid log(0) = -inf


@jax.jit
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


@jax.jit
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


@jax.jit
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
        return jax.vmap(eval_u_H_t, in_axes=(0, None))(x, theta)


@jax.jit
def eval_u_H_t(x_t: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    """Returns a human at time t given a state and a vector of
    parameters theta
    """
    min_distance = theta[0]
    x_R_t = x_t[0]
    x_H_t = x_t[2]
    sharpness_sigm = 2.0  # affects cutoff when r-h distance > min_distance
    sharpness_tanh = 20.0  # how quickly h-control goes between -1 and 1
    control_scale = 1.0

    def sigmoid(x):
        return 1 / (1 + jnp.exp(-sharpness_sigm * x))

    diff = x_R_t - x_H_t
    return jnp.array(
        [
            sigmoid(diff + min_distance)
            * sigmoid(-diff + min_distance)
            * -jnp.tanh(sharpness_tanh * diff)
            * control_scale
        ]
    )


@jax.jit
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


@jax.jit
def eval_x(
    x_0: jnp.ndarray, u_R: jnp.ndarray, theta: jnp.ndarray, w: jnp.ndarray
) -> jnp.ndarray:
    """Compute the state trajectory given init. state, robot controls, human
    control parameters, and noise.

    Inputs:
    - x_0: initial state
    - u_R: robot controls (u_0, u_1, ..., u_{t-1})
    - theta: human control parameters
    - w: noise trajectory (w_0, w_1, ..., w_{t-1})

    Outputs:
    - x: state trajectory (x_0, x_1, ..., x_{t})
    """

    def step_dynamics_scan(x_t, control_and_noise):
        u_R_t, w_t = control_and_noise
        u_H_t = eval_u_H_t(x_t, theta)
        u_t = jnp.concatenate([u_R_t, u_H_t])
        x_t_plus_1 = step_dynamics(x_t, u_t, w_t=w_t)
        return x_t_plus_1, x_t_plus_1

    x_trajectory, x = jax.lax.scan(step_dynamics_scan, x_0, (u_R, w))
    return jnp.stack([x_0, *x])


@jax.jit
def eval_E_x(
    x_0: jnp.ndarray, u_R: jnp.ndarray, b_0: jnp.ndarray, theta=None
) -> jnp.ndarray:
    """Compute expected states given init. state, init. belief, and robot
    controls
    Inputs:
    - x_0: state at time 0
    - b_0: belief at time 0
    - u_R: set of robot controls (u_0, u_1, ..., u_{t-1})
    - theta: human control parameters (optional)

    Outputs:
    - E_x: expected states (x_0, E[x_1], ..., E[x_{t}])

    Note: If theta is given, this is the expected control given the noise
    model. If theta is not given, this is the expected control given the noise
    model AND the belief over the human control parameters.
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


@jax.jit
def eval_b_t(
    x: jnp.ndarray, u: jnp.ndarray, b_0: jnp.ndarray, xs=None, us=None
) -> jnp.ndarray:
    """Returns the belief at time index t given the initial belief b_0.
    Inputs:
    - x: observed state trajectory (x_0, x_1, ..., x_{t})
    - u: observed control trajectory (u_0, u_1, ..., u_{t-1})
    - b_0: initial belief

    Optional:
    - xs: deterministic state trajectories for each theta
    - us: deterministic control trajectories for each theta

    Outputs:
    - b_k: belief at time t
    """
    t = len(x) - 1

    # Base case
    if t == 0:
        return b_0

    # Recursive case
    else:
        if xs is None and us is None:
            u_R = u[:, 0]  # TODO make this more general for higher dim inputs
            if u_R.ndim == 1:
                u_R = u_R[:, jnp.newaxis]
            xs = jax.vmap(eval_E_x, in_axes=(None, None, None, 0))(
                x[0], u_R, b_0, thetas
            )
            u_Hs = jax.vmap(eval_u_H, in_axes=(0, 0))(xs, thetas)
            us = jax.vmap(
                lambda u_H: jnp.concatenate([u_R, u_H[:-1]], axis=1)
            )(
                u_Hs
            )  # deterministic controls u_1, u_2, ..., u_{t-1} for each theta
        b_t_minus_1 = eval_b_t(x[:-1], u[:-1], b_0, xs[:, :-1], us[:, :-1])
        b_t = jax.vmap(
            lambda b_theta, x_theta, u_theta: b_theta
            * transition_model(x[-1], x_theta[:-1], u_theta)
        )(
            b_t_minus_1, xs, us
        )  # b_t = b_{t-1} * p(x_t | x_{0:t-1}, u_{0:t-1}, θ)
        b_t = b_t / jnp.sum(b_t)
        return b_t


@jax.jit
def transition_model(x_query: jnp.ndarray, x: jnp.ndarray, u: jnp.ndarray):
    """Return the probability of transitioning to x_query given x and u

    Assumes a linear system with additive Gaussian noise.

    Inputs:
    - x_query: state to query the pdf at
    - x: state trajectory (x_0, x_1, ..., x_{t-1})
    - u: control trajectory (u_0, u_1, ..., u_{t-1})

    Outputs:
    - Gaussian pdf for x_t given x_{0:t-1}
    """
    # Expected mean and covariance at time t
    t = len(x)
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
horizon = 7  # time horizon. Control inputs are u_0, u_1, ..., u_{horizon-1}
u_R_init = jnp.array([[0.0] for _ in range(horizon)])  # init guess for u_R
λ_init = 0.0  # curiosity parameter
λ_scale = 1.0  # curiosity parameter scaling factor
learning_rate = 0.001
optimizer = optax.adam(learning_rate)
descent_steps = 1000
mpc_steps = 10
rng_key = jax.random.PRNGKey(1)

num_points = 30
axis_limit = 50.0
plot_interval = descent_steps // 20
epsilon_cost = 0.0  # so that track is above surface
max_marker_size = 200
params_true = thetas[1]

# MPC loop
import time

eval_E_J_jit = jax.jit(eval_E_J)
sampled_w = jax.random.multivariate_normal(
    rng_key, jnp.zeros(len(x_0)), Sigma, (horizon,)
)


def solve_for_u_R(x_0, u_R_init, b, λ):
    # Note: Return last element of descent_trajectory instead of the whole for
    # efficiency gains
    def grad_descent_scan(carry, _):
        u_R, opt_state = carry
        grad = jax.grad(eval_E_J_jit, argnums=1)(x_0, u_R, b, λ)
        updates, opt_state = optimizer.update(grad, opt_state)
        u_R_new = optax.apply_updates(u_R, updates)
        return (u_R_new, opt_state), (u_R_new, jnp.linalg.norm(grad))

    opt_state = optimizer.init(u_R_init)
    _, descent_trajectory = jax.lax.scan(
        grad_descent_scan, (u_R_init, opt_state), jnp.arange(descent_steps)
    )
    return descent_trajectory


def update_λ(b_0, b_1, λ):
    ΔH = eval_H(b_1) - eval_H(b_0)
    return jax.lax.cond(
        ΔH > 0.0,
        lambda λ: λ + λ_scale * ΔH,
        lambda λ: jnp.clip(λ - λ_scale, 0.0, None),
        λ,
    )


def run_mpc(x_init, u_R_init, b_init, λ_init, w, mpc_steps):
    x = x_init[jnp.newaxis, :]
    u_R = u_R_init[0]
    b = b_init[jnp.newaxis, :]
    λ = jnp.array([λ_init])

    x_0 = x_init
    b_0 = b_init
    u_R_init = u_R_init
    λ_0 = λ_init
    solve_for_u_R_jit = jax.jit(solve_for_u_R)
    eval_x_jit = jax.jit(eval_x)
    eval_b_t_jit = jax.jit(eval_b_t)
    update_λ_jit = jax.jit(update_λ)
    for t in range(mpc_steps):
        time_start = time.time()
        optimization_results = solve_for_u_R_jit(x_0, u_R_init, b_0, λ_0)
        u_R_0 = optimization_results[0][-1]
        final_grad_norm = optimization_results[1][-1]
        u_R_0 = solve_for_u_R_jit(x_0, u_R_init, b_0, λ_0)[0][-1]
        x_1 = eval_x_jit(
            x_0, u_R_0[jnp.newaxis, 0], params_true, w[jnp.newaxis, t]
        )
        x = jnp.concatenate([x, x_1[jnp.newaxis, 1]], axis=0)
        if t == 0:
            u_R = u_R_0[jnp.newaxis, 0]
        else:
            u_R = jnp.concatenate([u_R, u_R_0[jnp.newaxis, 0]], axis=0)
        b_1 = eval_b_t_jit(
            x, u_R, b_0
        )  # All this would be scannable except for this line
        b = jnp.concatenate([b, b_1[jnp.newaxis, :]], axis=0)  # fix this
        λ_1 = update_λ_jit(b_0, b_1, λ_0)
        λ = jnp.append(λ, λ_1)

        x_0 = x_1[-1]
        b_0 = b_1
        u_R_init = u_R_0
        λ_0 = λ_1

        # print(
        #     f"{t}:{time.time() - time_start:.2f}s, "
        #     f"λ_1:{λ_1:.3f}, grad norm:{final_grad_norm:.3f}"
        # )

    return x, u_R, b, λ


# x, u_R, b, λ = run_mpc(x_0, u_R_init, b_init, λ_init, sampled_w, mpc_steps)

# # TEMPORARY: Profiling
with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    x, u_R, b, λ = run_mpc(x_0, u_R_init, b_init, λ_init, sampled_w, 1)

# Process the final trajectory
h = jax.vmap(eval_H)(jnp.array(b))

# Visualization
import matplotlib.pyplot as plt

# Robot and human position
# Subplot 1: Robot and human positions
# Subplot 2: Robot and human controls
# Subplot 3: Belief over time
# Subplot 4: Belief entropy
# Subplot 5: lambda over time

# x-axis position, y-axis time
fig, ax = plt.subplots(5, 1, figsize=(8, 8))
time_vec = jnp.arange(mpc_steps + 1) * dt
ax[0].plot(time_vec, x[:, 0], label="Robot")
ax[0].plot(time_vec, x[:, 2], label="Human")
ax[0].set_ylabel("Position [m]")
ax[0].legend()
ax[0].tick_params(
    axis="x",
    which="both",
    bottom=False,
    top=False,
    labelbottom=False,
)

ax[1].plot(time_vec[:-1], u_R, label="Robot")
ax[1].plot(time_vec[:-1], eval_u_H(x[:-1], params_true), label="Human")
ax[1].set_ylabel("Control [m/s^2]")
ax[1].tick_params(
    axis="x",
    which="both",
    bottom=False,
    top=False,
    labelbottom=False,
)

for t, belief in zip(time_vec, b):
    belief = belief / jnp.sum(belief)  # Shouldn't be necessary
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
ax[2].tick_params(
    axis="x",
    which="both",
    bottom=False,
    top=False,
    labelbottom=False,
)

ax[3].plot(time_vec, h)
ax[3].set_ylabel("Entropy")
ax[3].set_xlabel("Time [s]")
ax[3].set_ylim(bottom=0.0, top=max(h) + 0.1)

ax[4].plot(time_vec, λ)
ax[4].set_ylabel("Curiosity")
ax[4].set_xlabel("Time [s]")
ax[4].set_ylim(bottom=0.0, top=max(λ) + 0.1)

fig.tight_layout()

fig.savefig("figures/robot_human.png")
