import time

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np
from mujoco import mjx

xml_string = """
<mujoco>
    <asset>
        <hfield name="terrain" nrow="100" ncol="100" size="5 5 1.0 0.1"/>
    </asset>

    <worldbody>
        <light pos="0 0 10" dir="0 0 -1" directional="true"/>
        <geom name="floor" type="hfield" hfield="terrain" pos="0 0 0" rgba="0.3 0.5 0.3 1"/>

        <body name="ball" pos="0 0 3">
            <joint name="ball_free" type="free"/>
            <geom type="sphere" size="0.2" rgba="1 0 0 1" density="20"/>
        </body>
    </worldbody>

    <actuator>
        <motor name="push_x" joint="ball_free" gear="1 0 0 0 0 0"/>
        <motor name="push_y" joint="ball_free" gear="0 1 0 0 0 0"/>
    </actuator>
</mujoco>
"""


class Sim:
    def __init__(self, xml_string, dt=0.05):
        self.model = mujoco.MjModel.from_xml_string(xml_string)
        self.data = mujoco.MjData(self.model)
        self.substeps = max(1, round(dt / self.model.opt.timestep))

        hfield_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_HFIELD, "terrain")
        self.hfield_id = hfield_id
        self.hfield_adr = self.model.hfield_adr[hfield_id]
        self.hfield_nrow = self.model.hfield_nrow[hfield_id]
        self.hfield_ncol = self.model.hfield_ncol[hfield_id]

        x = np.linspace(-5, 5, self.hfield_nrow)
        y = np.linspace(-5, 5, self.hfield_ncol)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(X) * np.cos(Y)  # Bumpy surface matrix

        Z_normalized = (Z - Z.min()) / (Z.max() - Z.min())

        self.mjx_model = mjx.put_model(self.model)
        self.set_terrain(Z_normalized)
        self._mjx_state = mjx.make_data(self.mjx_model)

    @property
    def dt(self):
        return self.substeps * self.model.opt.timestep

    def set_terrain(self, heights):
        """Update the height field in place across host model and mjx model."""
        n = self.hfield_nrow * self.hfield_ncol
        self.model.hfield_data[self.hfield_adr : self.hfield_adr + n] = np.asarray(
            heights
        ).flatten()
        self.mjx_model = self.mjx_model.replace(
            hfield_data=jnp.asarray(self.model.hfield_data)
        )

    def step(self, action):
        self.data.ctrl[:] = action
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)

    def get_mjx_model(self):
        return self.mjx_model

    def get_mjx_dynamics_fn(self):
        def dynamics_fn(state, action, mjmodel):
            state = state.replace(ctrl=action)
            return jax.lax.fori_loop(
                0,
                self.substeps,
                lambda _, s: mjx.step(mjmodel, s),
                state,
            )

        return dynamics_fn

    def get_mjx_state(self):
        return self._mjx_state.replace(
            qpos=jnp.asarray(self.data.qpos),
            qvel=jnp.asarray(self.data.qvel),
            act=jnp.asarray(self.data.act),
            time=self.data.time,
        )


if __name__ == "__main__":
    sim = Sim(xml_string)

    horizon = int(0.5 / sim.dt)
    body_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "ball")

    @jax.jit
    def rollout(model, state, actions):
        step = sim.get_mjx_dynamics_fn()

        def scan_fn(state, action):
            next_state = step(state, action, model)
            return next_state, next_state

        _, trajectory = jax.lax.scan(scan_fn, state, actions)
        return trajectory

    def draw_trajectory(viewer, positions):
        with viewer.lock():
            viewer.user_scn.ngeom = 0
            for pos in positions:
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[viewer.user_scn.ngeom],
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([0.05, 0.0, 0.0]),
                    np.asarray(pos),
                    np.eye(3).flatten(),
                    np.array([1.0, 0.5, 0.0, 0.7], dtype=np.float32),
                )
                viewer.user_scn.ngeom += 1

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        actions = jnp.zeros((horizon, sim.model.nu))
        flattened = False
        start_time = time.time()
        while viewer.is_running() and (time.time() - start_time < 20.0):
            step_start = time.time()

            if not flattened and sim.data.time >= 5.0:
                sim.set_terrain(np.zeros(sim.hfield_nrow * sim.hfield_ncol))
                viewer.update_hfield(sim.hfield_id)
                flattened = True

            trajectory = rollout(sim.get_mjx_model(), sim.get_mjx_state(), actions)
            positions = np.asarray(trajectory.xpos)[:, body_id]
            draw_trajectory(viewer, positions)

            sim.step(np.zeros(2))
            viewer.sync()

            # Advance the sim in realtime at the control rate
            time_until_next_step = sim.dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
