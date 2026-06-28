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

        <texture name="checkerboard" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 .15 .2"
                 width="512" height="512" mark="none"/>
        <material name="background_mat" texture="checkerboard" texrepeat="10 10"/>
    </asset>

    <worldbody>
        <light pos="0 0 10" dir="0 0 -1" directional="true"/>

        <geom name="background_floor" type="plane" size="50 50 .1" pos="0 0 -0.05"
              material="background_mat" contype="0" conaffinity="0"/>

        <body name="floor_body" mocap="true" pos="0 0 0">
            <geom name="floor" type="hfield" hfield="terrain" rgba="0.3 0.5 0.3 1"/>
        </body>

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

        self.hfield_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_HFIELD, "terrain"
        )
        self.ball_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "ball"
        )
        self.floor_mocap_id = self.model.body_mocapid[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "floor_body")
        ]

        self.hfield_adr = self.model.hfield_adr[self.hfield_id]
        self.hfield_nrow = self.model.hfield_nrow[self.hfield_id]
        self.hfield_ncol = self.model.hfield_ncol[self.hfield_id]

        self.global_extent = 50.0
        self.global_res = 0.1
        x_global = np.arange(-self.global_extent, self.global_extent, self.global_res)
        y_global = np.arange(-self.global_extent, self.global_extent, self.global_res)
        X, Y = np.meshgrid(x_global, y_global)
        # Z = np.sin(X) * np.cos(Y) + 0.5 * np.sin(0.5 * X)
        Z = X + np.sin(Y)
        self.global_hmap = (Z - Z.min()) / (Z.max() - Z.min())
        self.model.hfield_size[self.hfield_id] = [10.0, 10.0, Z.min(), Z.max()]

        self.mjx_model = mjx.put_model(self.model)
        self.update_local_terrain()
        self._mjx_state = mjx.make_data(self.mjx_model)

    @property
    def dt(self):
        return self.substeps * self.model.opt.timestep

    def update_local_terrain(self):
        """Slice the global heightmap around the robot and sync it into data and mjx."""
        robot_pos = self.data.xpos[self.ball_body_id][:2]

        center_idx_x = int(round((robot_pos[0] + self.global_extent) / self.global_res))
        center_idx_y = int(round((robot_pos[1] + self.global_extent) / self.global_res))

        half_row = self.hfield_nrow // 2
        half_col = self.hfield_ncol // 2
        start_x = max(
            0,
            min(center_idx_x - half_row, self.global_hmap.shape[0] - self.hfield_nrow),
        )
        start_y = max(
            0,
            min(center_idx_y - half_col, self.global_hmap.shape[1] - self.hfield_ncol),
        )

        sub_hmap = self.global_hmap[
            start_x : start_x + self.hfield_nrow, start_y : start_y + self.hfield_ncol
        ]

        snapped_x = (start_x + half_row) * self.global_res - self.global_extent
        snapped_y = (start_y + half_col) * self.global_res - self.global_extent

        # 1. Update the heightmap matrix data array in the model
        n = self.hfield_nrow * self.hfield_ncol
        self.model.hfield_data[self.hfield_adr : self.hfield_adr + n] = (
            sub_hmap.flatten()
        )

        # 2. Update the dynamic position of the terrain using mocap_pos
        self.data.mocap_pos[self.floor_mocap_id, :2] = [snapped_x, snapped_y]

        # 3. Synchronize both down to MJX structures
        self.mjx_model = self.mjx_model.replace(
            hfield_data=jnp.asarray(self.model.hfield_data),
        )

    def step(self, action):
        self.data.ctrl[:] = action
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
        self.update_local_terrain()

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
            mocap_pos=jnp.asarray(self.data.mocap_pos),
            mocap_quat=jnp.asarray(self.data.mocap_quat),
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
        start_time = time.time()
        while viewer.is_running() and (time.time() - start_time < 30.0):
            step_start = time.time()

            trajectory = rollout(
                sim.get_mjx_model(),
                sim.get_mjx_state(),
                jnp.zeros((horizon, sim.model.nu)),
            )
            positions = np.asarray(trajectory.xpos)[:, body_id]
            draw_trajectory(viewer, positions)

            sim.step(np.zeros(sim.model.nu))
            viewer.update_hfield(sim.hfield_id)
            viewer.sync()

            time_until_next_step = sim.dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
