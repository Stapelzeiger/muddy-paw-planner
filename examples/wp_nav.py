"""Waypoint navigation for a ball using MPPI with MJX ground-truth dynamics."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
import mujoco.viewer
import numpy as np

from mppi import mppi_step
from sim import Sim, ball_model


def key_stream(key):
    while True:
        key, subkey = jax.random.split(key)
        yield subkey


def cost_fn(_cost_state, dyn_state, action, waypoint):
    print("tracing")
    ball_xy = dyn_state.qpos[:2]
    dist_sq = jnp.sum((ball_xy - waypoint) ** 2)
    control_cost = 0.01 * jnp.sum(action**2)
    return None, dist_sq + control_cost


def terminal_cost_fn(_cost_state, dyn_state, waypoint):
    ball_xy = dyn_state.qpos[:2]
    return 10.0 * jnp.sum((ball_xy - waypoint) ** 2)


def main():
    sim = Sim(ball_model, robot_body_name="ball")
    mjx_model = sim.get_mjx_model()
    dynamics_fn = sim.get_mjx_dynamics_fn()

    waypoints = [
        np.array([5.0, 0.0], dtype=np.float32),
        np.array([5.0, 5.0], dtype=np.float32),
        np.array([0.0, 5.0], dtype=np.float32),
        np.array([0.0, 0.0], dtype=np.float32),
    ]

    horizon = 20
    action_dim = sim.model.nu

    rng = key_stream(jax.random.PRNGKey(0))
    nominal_traj = jnp.zeros((horizon, action_dim))

    wp_idx = 0

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        for step in range(200):
            if not viewer.is_running():
                break

            step_start = time.time()
            mjx_state = sim.get_mjx_state()
            waypoint = jnp.asarray(waypoints[wp_idx])

            nominal_traj, _ = mppi_step(
                next(rng),
                mjx_state,
                nominal_traj,
                dynamics_fn,
                mjx_model,
                cost_fn,
                terminal_cost_fn,
                waypoint,
                num_samples=100,
                noise_std=jnp.array([0.5, 0.5]),
            )

            action = np.array(nominal_traj[0])
            sim.step(action)

            ball_pos = sim.data.xpos[sim.robot_body_id][:2]
            dist = float(np.linalg.norm(ball_pos - waypoints[wp_idx]))
            if dist < 0.5:
                wp_idx = min(wp_idx + 1, len(waypoints) - 1)

            print(
                f"Step {step:03d} | pos=[{ball_pos[0]:6.2f}, {ball_pos[1]:6.2f}] "
                f"| wp={wp_idx} | dist={dist:.2f}"
            )

            viewer.sync()
            viewer.update_hfield(sim.hfield_id)

            nominal_traj = jnp.roll(nominal_traj, shift=-1, axis=0)
            nominal_traj = nominal_traj.at[-1].set(jnp.zeros(action_dim))

            time_until_next_step = sim.dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
