"""Model Predictive Path Integral

This implements a generic MPPI optimizer with optional coloured-noise perturbations.
"""

import functools
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp

from colorednoise import colored_noise

__all__ = ["mppi_step"]


@functools.partial(
    jax.jit,
    static_argnames=(
        "dynamics_fn",
        "cost_fn",
        "noise_freq_exponent",
        "integrate_noise",
        "top_k",
    ),
)
def mppi_step(
    key: jax.Array,
    current_state: Any,
    nominal_trajectory: jax.Array,
    goal_context: Any,
    dynamics_fn: Callable[[Any, jax.Array], Any],
    cost_fn: Callable,
    noise_std: jax.Array = jnp.array([0.5]),
    noise_freq_exponent: float = 0.0,
    integrate_noise: bool = False,
    lambda_: float = 1.0,
    num_samples: int = 10000,
    top_k: int = 0,
) -> Tuple[jax.Array, Tuple]:
    """Single MPPI update step.

    Draws ``num_samples`` Gaussian perturbations around ``nominal_trajectory``, rolls each
    sequence forward with ``dynamics_fn``, and sums per-step costs from ``cost_fn``.

    The cost function can be stateful with initial cost state provided by ``cost_fn.initial_cost_state``.
    ``cost_fn(cost_state, dyn_state, action, goal_context, i, horizon) -> (next_cost_state, step_cost)``.

    Args:
        key: JAX PRNG key for sampling noise.
        current_state: Dynamic state at the start of the horizon.
        nominal_trajectory: ``(horizon, action_dim)`` nominal control sequence.
        goal_context: User-defined context (e.g. goal) passed through to ``cost_fn``.
        dynamics_fn: ``(state, action) -> next_state``.
        cost_fn: Stateful per-step cost; see above.
        noise_std: Per-action noise scale, broadcast over ``action_dim``.
        noise_freq_exponent: Power-law exponent β for coloured noise (0 = white).
            β=1 pink, β=2 brown/red, higher β = smoother perturbations.
        integrate_noise: Integrate noise along the horizon (random-walk perturbations)
        lambda_: Temperature in the exponential weights (larger => softer selection).
        num_samples: Number of parallel sampled trajectories.
        top_k: If ``> 0``, also return the top-``k`` rollouts for debugging / visualization.

    Returns:
        ``(updated_trajectory, (top_k_state_trajectories, top_k_action_trajectories, top_k_costs))``.
    """
    horizon = nominal_trajectory.shape[0]
    action_dim = nominal_trajectory.shape[1]

    if noise_freq_exponent == 0.0:
        noise = (
            jax.random.normal(key, shape=(num_samples, horizon, action_dim)) * noise_std
        )
    else:
        raw = colored_noise(  # (num_samples, action_dim, horizon)
            key,
            exponent=noise_freq_exponent,
            size=(num_samples, action_dim, horizon),
        )
        noise = (
            jnp.transpose(raw, (0, 2, 1)) * noise_std
        )  # (num_samples, horizon, action_dim)

    if integrate_noise:
        noise = jnp.cumsum(noise, axis=1)

    perturbed_trajectories = nominal_trajectory + noise
    initial_cost_state = getattr(cost_fn, "initial_cost_state", None)

    def rollout_single_trajectory(control_sequence):
        def step(carry, idx_action):
            dyn_state, cost_state = carry
            i, current_action = idx_action
            next_dyn_state = dynamics_fn(dyn_state, current_action)
            next_cost_state, step_cost = cost_fn(
                cost_state, next_dyn_state, current_action, goal_context, i, horizon
            )
            return (next_dyn_state, next_cost_state), step_cost

        time_indices = jnp.arange(horizon)
        inputs = (time_indices, control_sequence)
        init_carry = (current_state, initial_cost_state)
        _, step_costs = jax.lax.scan(step, init_carry, inputs)
        return jnp.sum(step_costs)

    rollout_all = jax.vmap(rollout_single_trajectory)
    costs = rollout_all(perturbed_trajectories)

    costs_normalized = costs - jnp.min(costs)  # for numerical stability

    weights = jnp.exp(-(1.0 / lambda_) * costs_normalized)
    weights = weights / jnp.sum(weights)

    weighted_noise = jnp.sum(weights[:, None, None] * noise, axis=0)
    updated_trajectory = nominal_trajectory + weighted_noise

    top_state_trajectories, top_action_trajectories, top_costs = get_top_k_trajectories(
        current_state,
        perturbed_trajectories,
        costs,
        dynamics_fn,
        top_k,
    )
    return updated_trajectory, (
        top_state_trajectories,
        top_action_trajectories,
        top_costs,
    )


def get_top_k_trajectories(
    current_state: Any,
    perturbed_trajectories: jax.Array,
    costs: jax.Array,
    dynamics_fn: Callable[[Any, jax.Array], Any],
    top_k: int,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    _, top_indices = jax.lax.top_k(-costs, top_k)
    top_action_trajectories = perturbed_trajectories[top_indices]

    def rollout_states(control_sequence):
        def step(carry_state, action):
            next_state = dynamics_fn(carry_state, action)
            return next_state, next_state  # return state as the output

        _, state_history = jax.lax.scan(step, current_state, control_sequence)
        return state_history

    top_state_trajectories = jax.vmap(rollout_states)(top_action_trajectories)
    return top_state_trajectories, top_action_trajectories, costs[top_indices]


if __name__ == "__main__":
    """
    Simple example: MPPI on a unicycle driving to a goal.
    """
    import math

    import matplotlib.pyplot as plt
    import numpy as np

    def unicycle_dynamics(state, action, dt=0.1):
        """Unicycle model. State: [x, y, theta]. Action: [v, omega]"""
        x, y, theta = state
        v, omega = action
        v = jnp.clip(v, -1.0, 1.0)
        omega = jnp.clip(omega, -1.0, 1.0)

        next_x = x + v * jnp.cos(theta) * dt
        next_y = y + v * jnp.sin(theta) * dt
        next_theta = theta + omega * dt

        return jnp.array([next_x, next_y, next_theta])

    class unicycle_cost_fn:
        def __init__(self):
            self.initial_cost_state = None

        def __call__(self, cost_state, state, action, goal_context, i, N):
            """
            Stateful per-step cost: (cost_state, step_cost).

            cost_state is carried across the horizon; this default ignores it.
            i is the step index in [0, N). N is the horizon length.
            """
            pos_error = jnp.sum((state[:2] - goal_context[:2]) ** 2)
            theta_error = (state[2] - goal_context[2]) ** 2
            control_cost = 0.01 * jnp.sum(action**2)
            step_cost = pos_error + 0.1 * theta_error + control_cost
            return cost_state, step_cost

    rng = jax.random.PRNGKey(42)
    state = jnp.array([0.0, 0.0, 0.0])  # Start at origin, facing right
    target = jnp.array([5.0, 5.0, jnp.pi / 2])  # Target: x=5, y=5, facing up

    horizon = 30
    action_dim = 2
    nominal_traj = jnp.zeros((horizon, action_dim))  # Initial guess

    print(f"Initial State: {state}")
    print(f"Target State:  {target}\n")

    cost_fn = unicycle_cost_fn()

    state_history = [np.array(state)]
    control_history = []
    # Snapshots: (step, top_state_trajs, opt_path, cur_state)
    snapshots = []

    def rollout_states_np(init, controls):
        s = init
        path = []
        for u in controls:
            s = unicycle_dynamics(s, u)
            path.append(np.array(s))
        return np.array(path)

    for step in range(100):
        rng, step_key = jax.random.split(rng)

        nominal_traj, (top_states, _, top_costs) = mppi_step(
            step_key,
            state,
            nominal_traj,
            target,
            unicycle_dynamics,
            cost_fn,
            top_k=30,
        )
        u = nominal_traj[0]

        # Optimal rollout from current state under updated nominal
        opt_path = rollout_states_np(state, nominal_traj)

        if step % 5 == 0:
            snapshots.append(
                (step, np.array(top_states), np.array(opt_path), np.array(state))
            )

        state = unicycle_dynamics(state, u)
        state_history.append(np.array(state))
        control_history.append(np.array(u))

        # Receding horizon update
        nominal_traj = jnp.roll(nominal_traj, shift=-1, axis=0)
        nominal_traj = nominal_traj.at[-1].set(jnp.zeros(action_dim))

        print(
            f"Step {step:02d} | Control: [{u[0]:.2f}, {u[1]:.2f}] | State: [{state[0]:.2f}, {state[1]:.2f}, {state[2]:.2f}]"
        )

    # ── Plot ───────────────────────────────────────────────────────────────
    ncols = math.ceil(math.sqrt(len(snapshots)))
    nrows = math.ceil(len(snapshots) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 4 * nrows), sharex=True, sharey=True
    )
    axes_flat = axes.flat

    state_traj = np.array(state_history)

    for ax, (step, top_states, opt_path, cur_state) in zip(axes_flat, snapshots):
        # Top-30 sample trajectories
        for traj in top_states:
            ax.plot(
                traj[:, 0], traj[:, 1], color="steelblue", alpha=0.15, linewidth=0.8
            )

        # Optimal (nominal) trajectory
        ax.plot(
            opt_path[:, 0], opt_path[:, 1], color="crimson", linewidth=1.8, zorder=3
        )

        # Executed path so far
        ax.plot(
            state_traj[: step + 1, 0],
            state_traj[: step + 1, 1],
            color="k",
            linewidth=1.2,
            linestyle="--",
            zorder=4,
        )

        # Current position
        ax.scatter(*cur_state[:2], color="k", s=30, zorder=5)

        # Goal
        ax.scatter(
            *target[:2],
            marker="*",
            s=200,
            color="gold",
            edgecolors="k",
            linewidths=0.5,
            zorder=5,
        )

        ax.set_title(f"Step {step}")
        ax.set_aspect("equal")

    # Hide any leftover empty axes
    for ax in list(axes_flat)[len(snapshots) :]:
        ax.set_visible(False)

    # x-labels on bottom row, y-labels on left column
    for ax in axes[-1, :ncols]:
        ax.set_xlabel("x")
    for ax in axes[:, 0]:
        ax.set_ylabel("y")

    fig.suptitle("MPPI — unicycle (top 30 samples every 5 steps)", fontsize=11)
    plt.tight_layout()

    # ── Control history ─────────────────────────────────────────────────────
    controls = np.array(control_history)
    steps = np.arange(len(controls))

    fig2, (ax_v, ax_w) = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
    ax_v.plot(steps, controls[:, 0], color="steelblue")
    ax_v.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax_v.set_ylabel("v (m/s)")
    ax_v.set_ylim(-1.1, 1.1)

    ax_w.plot(steps, controls[:, 1], color="crimson")
    ax_w.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax_w.set_ylabel("ω (rad/s)")
    ax_w.set_ylim(-1.1, 1.1)
    ax_w.set_xlabel("Step")

    fig2.suptitle("MPPI — control history", fontsize=11)
    plt.tight_layout()
    plt.show()
