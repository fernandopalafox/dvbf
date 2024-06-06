import jax
import numpy
from jax import numpy as jnp

# Simulation parameters
N = 10
thetas = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])
x_0 = jnp.array([0.0, 0.0])
A = jnp.array([[1.0, 1.0], [0.0, 1.0]])
B = jnp.array([[0.5], [1.0]])
sigma = jnp.array([[1.0, 0.0], [0.0, 1.0]])


# Cost
def stage_cost(x_t, u_t):
    Q = jnp.array([[1, 0], [0, 1]])
    return jnp.dot(jnp.dot(x_t, Q), x_t)


def terminal_cost(x_T):
    return jnp.dot(x_T, x_T)


def step_dynamics(x_t: jnp.ndarray, u_t: jnp.ndarray) -> jnp.ndarray:
    return A @ x_t + B @ u_t


def eval_E_J(b_0, u_r):
    """
    Returns the expected cost given the initial belief b_0 and robot controls u_r
    Inputs:
    - b_0: initial belief
    - u_r: robot controls

    Outputs:
    - E_J: expected cost
    """
    t = len(u_r) - 1
    b_t = eval_estimated_b_t(b_0, u_r)
    E_J = jnp.sum(
        jnp.stack(
            [
                jnp.dot(
                    b_t[l],
                    jnp.array(
                        [
                            stage_cost(eval_E_x_t(x_0, b_0, k), u_r[k])
                            for k in range(t)
                        ]
                        + [terminal_cost(eval_E_x_t(x_0, b_0, t))]
                    ),
                )
                for l in range(len(b_t))
            ]
        )
    )
    return E_J


def eval_u_h(x: jnp.ndarray, theta: jnp.ndarray):
    """Returns human controls given states and a vector of parameters theta"""
    return jnp.array([eval_u_h_t(x_t, theta) for x_t in x])


def eval_u_h_t(x_t: jnp.ndarray, theta: jnp.ndarray):
    """Returns a single human controls given a state and a vector of parameters theta"""
    print("warning: u_h is not implemented")
    return jnp.array([1.0, 1.0, 1.0])


def eval_E_x(x_0: jnp.ndarray, b_0: jnp.ndarray, u_r: jnp.ndarray):
    """
    Expected states given init. state, init. belief, and robot controls
    Inputs:
    - x_0: state at time 0
    - b_0: belief at time 0
    - u_r: set of robot controls (u_0, u_1, ..., u_{t-1})
    """

    def step_dynamics_scan(x_t, u_r_t):
        E_u_h_t = jnp.sum(
            jnp.stack(
                [
                    eval_u_h(x_t, theta) * b_0[l]
                    for l, theta in enumerate(thetas)
                ]
            ),
            axis=0,
        )
        E_u_t = jnp.concatenate([u_r_t, E_u_h_t], axis=1)
        x_t_plus_1 = step_dynamics(x_t, E_u_t)
        return x_t_plus_1, x_t_plus_1

    E_x_final, E_x = jax.lax.scan(step_dynamics_scan, x_0, u_r)
    return E_x


def eval_E_x_t(x_0: jnp.ndarray, b_0: jnp.ndarray, u_r: jnp.ndarray):
    """
    Expected state given init. state, init. belief, and robot controls
    Inputs:
    - x_0: state at time 0
    - b_0: belief at time 0
    - u_r: set of robot controls

    Outputs:
    - E_x_t: expected state after applying robot controls
    """
    t = len(u_r)
    if t == 1:
        return x_0
    else:
        E_x_t_minus_1 = eval_E_x_t(x_0, b_0, u_r[:-1])
        E_u_h_t_minus_1 = jnp.sum(
            jnp.stack(
                [
                    eval_u_h(E_x_t_minus_1, theta) * b_0[l]
                    for l, theta in enumerate(thetas)
                ]
            ),
            axis=0,
        )
        E_u_t_minus_1 = jnp.concatenate([u_r[-1], E_u_h_t_minus_1], axis=1)
        return step_dynamics(E_x_t_minus_1, E_u_t_minus_1)


def eval_E_u_h_t(x_0: jnp.ndarray, b_0: jnp.ndarray, t: int):
    """
    Returns the expected human control at time index t given state and belief at time 0
    Inputs:
    - x_0: initial state
    - b_0: initial belief
    - t: timestep index. Must be >= 0

    Outputs:
    - E_u_h_t: expected control at time index t
    """

    if t == 0:
        E_u_0 = jnp.sum(
            jnp.stack(
                [
                    eval_u_h_t(x_0, theta) * b_0[l]
                    for l, theta in enumerate(thetas)
                ]
            ),
            axis=0,
        )  # use x_0
        return E_u_0
    else:
        E_x_t = eval_E_x_t(x_0, b_0, t)
        E_u_t = jnp.sum(
            jnp.stack(
                [
                    eval_u_h_t(E_x_t, theta) * b_0[l]
                    for l, theta in enumerate(thetas)
                ]
            ),
            axis=0,
        )  # Use E_x_t
        return E_u_t


def eval_H(b: jnp.ndarray):
    """Shannon entropy of a belief vector b"""
    return -jnp.sum(b * jnp.log(b))


def eval_estimated_b_t(b_0: jnp.ndarray, u_r: jnp.ndarray):
    """
    Returns the estimated belief at time index t given the initial belief b_0.
    Inputs:
    - b_0: initial belief
    - u_r: robot controls
    - t: timestep index. Must be >= 0

    Outputs:
    - b_k: belief at time t
    """
    t = len(u_r) - 1

    # Base case
    if t == 0:
        return b_0

    # Recursive case
    else:
        b_t_minus_1 = eval_estimated_b_t(b_0, u_r[:-1])
        E_x_t = eval_E_x_t(x_0, b_0, t)
        transition_models = [
            make_transition_model(b_0, u_r, t, theta) for theta in thetas
        ]
        b_t = jnp.array(
            b_t_minus_1(theta) * transition_models[l](E_x_t)
            for l, theta in enumerate(thetas)
        )
        return b_t


def make_transition_model(
    b_0: jnp.ndarray, u_r: jnp.ndarray, t: int, theta: jnp.ndarray
):
    """Transition model for probability of x_{t} given x_{0}, b_{0}."""

    E_x = eval_E_x_t(x_0, b_0, u_r)  # expected x_{0:t-1}
    u_h = eval_u_h(E_x, theta)
    assert len(u_h) == len(u_r)
    u = jnp.concatenate([u_r, u_h], axis=1)  # u_{0:t-1}

    # Expected mean and covariance at time t
    mu_t = A**t @ x_0 + sum([A ** (t - 1 - j) @ B @ u[j] for j in range(t)])
    sigma_t = sum([A**j @ sigma @ (A**j).T for j in range(t)])

    k = len(mu_t)

    # Define a Gaussian PDF using expected mean and cov at time t
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
