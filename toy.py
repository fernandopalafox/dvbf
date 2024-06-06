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

print("UPDATE NVIDIA DRIVER!!!!")


# Cost
def stage_cost(x_t, u_t):
    Q = jnp.array([[1, 0], [0, 1]])
    return jnp.dot(jnp.dot(x_t, Q), x_t)


def terminal_cost(x_T):
    return jnp.dot(x_T, x_T)


def step_dynamics(x_t: jnp.ndarray, u_t: jnp.ndarray) -> jnp.ndarray:
    return A @ x_t + B @ u_t


def eval_J_r(x_0: jnp.ndarray, u_R: jnp.ndarray) -> jnp.ndarray:
    """Evaluate the robot cost given initial state and robot controls"""
    return 0.0


def eval_H(b: jnp.ndarray) -> jnp.ndarray:
    """Shannon entropy of a belief vector b"""
    return -jnp.sum(b * jnp.log(b))


def eval_J_i(
    x: jnp.ndarray, u_R: jnp.ndarray, b_0: jnp.ndarray
) -> jnp.ndarray:
    """Evaluate the information cost given the state trajectory, robot controls
    and initial belief.
    Inputs:
    - x: state trajectory (x_0, x_1, ..., x_{T})
    - u_R: robot controls (u_0, u_1, ..., u_{T-1})
    - b_0: initial belief
    """
    return eval_H(eval_b_t(x, u_R, b_0))


def eval_E_J(
    x_0: jnp.ndarray, u_R: jnp.ndarray, b_0: jnp.ndarray, λ: float
) -> jnp.ndarray:
    """Evaluate the expected cost given initial state, initial belief,
    and robot controls
    Note: This is the cost of the expected trajectory
    """
    E_x = eval_E_x(x_0, b_0, u_R)
    return eval_J_r(E_x, u_R) + λ * eval_J_i(E_x, u_R, b_0)


def eval_u_H(x: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    """Returns human controls given states and a vector of parameters
    theta
    """

    return jnp.array([eval_u_H_t(x_t, theta) for x_t in x])


def eval_u_H_t(x_t: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    """Returns a single human controls given a state and a vector of
    parameters theta
    """

    print("warning: u_H is not implemented")
    return jnp.array([1.0, 1.0, 1.0])


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
    return x


def eval_E_x(
    x_0: jnp.ndarray, u_R: jnp.ndarray, b_0: jnp.ndarray
) -> jnp.ndarray:
    """Compute expected states given init. state, init. belief, and robot
    controls
    Inputs:
    - x_0: state at time 0
    - b_0: belief at time 0
    - u_R: set of robot controls (u_0, u_1, ..., u_{t-1})
    """

    def step_dynamics_scan(x_t, u_R_t):
        E_u_H_t = jnp.sum(
            jnp.stack(
                [
                    eval_u_H(x_t, theta) * b_0[l]
                    for l, theta in enumerate(thetas)
                ]
            ),
            axis=0,
        )
        E_u_t = jnp.concatenate([u_R_t, E_u_H_t], axis=1)
        x_t_plus_1 = step_dynamics(x_t, E_u_t)
        return x_t_plus_1, x_t_plus_1

    E_x_final, E_x = jax.lax.scan(step_dynamics_scan, x_0, u_R)
    return E_x


def eval_b_t(
    x: jnp.ndarray, u_R: jnp.ndarray, b_0: jnp.ndarray
) -> jnp.ndarray:
    """Returns the belief at time index t given the initial belief b_0.
    Inputs:
    - b_0: initial belief
    - x: state trajectory (x_0, x_1, ..., x_{t})
    - u_R: robot controls (u_0, u_1, ..., u_{t-1})

    Outputs:
    - b_k: belief at time t
    """
    t = len(u_R)

    # Base case
    if t == 1:
        return b_0

    # Recursive case
    else:
        b_t_minus_1 = eval_b_t(x[:-1], u_R[:-1], b_0)
        transition_models = [
            make_transition_model(x, u_R, theta) for theta in thetas
        ]
        b_t = jnp.array(
            b_t_minus_1[l] * transition_models[l](x[-1])
            for l, _ in enumerate(thetas)
        )
        return b_t


def make_transition_model(
    x: jnp.ndarray, u_R: jnp.ndarray, theta: jnp.ndarray
):
    """Return the state transition model at time t.

    Inputs:
    - x: state trajectory (x_0, x_1, ..., x_{t})
    - u_R: robot controls (u_0, u_1, ..., u_{t-1})

    Outputs:
    - gaussian_pdf: a Gaussian pdf for the probability of x_t given x_{0:t-1}

    """

    # Controls up to and including time t-1
    u_H = eval_u_H(x[:-1], theta)
    assert len(u_H) == len(u_R)
    u = jnp.concatenate([u_R, u_H], axis=1)  # u_{0:t-1}

    # Expected mean and covariance at time t
    t = len(u_R)
    mu_t = A**t @ x_0 + sum([A ** (t - 1 - j) @ B @ u[j] for j in range(t)])
    sigma_t = sum([A**j @ sigma @ (A**j).T for j in range(t)])
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
