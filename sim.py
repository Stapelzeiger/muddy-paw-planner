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
        <texture name="checkerboard" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 .15 .2"
                 width="512" height="512" mark="none"/>
        <material name="background_mat" texture="checkerboard" texrepeat="10 10"/>
    </asset>

    <worldbody>
        <light pos="0 0 10" dir="0 0 -1" directional="true"/>

        <geom name="background_floor" type="plane" size="50 50 .1" pos="0 0 -10"
              material="background_mat" contype="0" conaffinity="0"/>

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


def elevation(x, y):
    return np.cos(x) + np.sin(2 * y) + 0.1 * x - 100


def add_hfield(spec, name, nrow, ncol, size, initial_data):
    """Attach a height-field asset plus a mocap floor body/geom to the spec.

    The floor body is mocap so the local terrain window can be repositioned and
    re-sliced around the robot at runtime without going through the physics.
    """
    hfield = spec.add_hfield(name=name, nrow=nrow, ncol=ncol, size=size)
    hfield.userdata = initial_data.flatten().astype(np.float32)

    body = spec.worldbody.add_body(name="floor_body", mocap=True)
    body.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_HFIELD,
        hfieldname=name,
        rgba=[0.3, 0.5, 0.3, 1],
    )
    return body


class Sim:
    def __init__(
        self,
        xml_string,
        dt=0.05,
        global_extent=50.0,
        global_res=0.1,
        hfield_nrow=101,
        hfield_ncol=101,
    ):
        self.global_extent = global_extent
        self.global_res = global_res
        self.hfield_nrow = hfield_nrow
        self.hfield_ncol = hfield_ncol

        global_x = np.arange(-global_extent, global_extent, global_res)
        global_y = np.arange(-global_extent, global_extent, global_res)
        X, Y = np.meshgrid(global_x, global_y)
        Z = elevation(X, Y)

        # Metric heights shifted so the origin sits at z=0.
        origin_idx = int(round(global_extent / global_res))
        Z = Z - Z[origin_idx, origin_idx]
        self.global_min = float(Z.min())
        self.global_range = float(Z.max() - Z.min())
        # Stored normalized to [0, 1] over the global range so a slice can be
        # written straight into the hfield; max_height/geom-z recover metric z.
        self.global_hmap = (Z - self.global_min) / self.global_range

        # Hfield half-extents aligned to the global grid spacing.
        hx = (hfield_ncol - 1) * global_res / 2
        hy = (hfield_nrow - 1) * global_res / 2
        base = 0.1
        size = [hx, hy, self.global_range, base]

        spec = mujoco.MjSpec.from_string(xml_string)
        add_hfield(
            spec,
            name="terrain",
            nrow=hfield_nrow,
            ncol=hfield_ncol,
            size=size,
            initial_data=np.zeros(hfield_nrow * hfield_ncol),
        )
        self.model = spec.compile()
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

        self.mjx_model = mjx.put_model(self.model)
        self.update_local_terrain()
        self._mjx_state = mjx.make_data(self.mjx_model)

    @property
    def dt(self):
        return self.substeps * self.model.opt.timestep

    def update_local_terrain(self):
        """Slice the global heightmap around the robot and sync it into data and mjx."""
        robot_x, robot_y = self.data.xpos[self.ball_body_id][:2]
        extent = self.global_extent
        res = self.global_res

        ix_center = int(round((robot_x + extent) / res))
        iy_center = int(round((robot_y + extent) / res))
        half_nrow = self.hfield_nrow // 2
        half_ncol = self.hfield_ncol // 2
        start_iy = max(
            0, min(iy_center - half_nrow, self.global_hmap.shape[0] - self.hfield_nrow)
        )
        start_ix = max(
            0, min(ix_center - half_ncol, self.global_hmap.shape[1] - self.hfield_ncol)
        )

        sub_hmap = self.global_hmap[
            start_iy : start_iy + self.hfield_nrow,
            start_ix : start_ix + self.hfield_ncol,
        ]

        x_lo = -extent + start_ix * res
        x_hi = -extent + (start_ix + self.hfield_ncol - 1) * res
        y_lo = -extent + start_iy * res
        y_hi = -extent + (start_iy + self.hfield_nrow - 1) * res
        center_x = (x_lo + x_hi) / 2
        center_y = (y_lo + y_hi) / 2

        n = self.hfield_nrow * self.hfield_ncol
        self.model.hfield_data[self.hfield_adr : self.hfield_adr + n] = (
            sub_hmap.flatten()
        )

        self.data.mocap_pos[self.floor_mocap_id] = [
            center_x,
            center_y,
            self.global_min,
        ]
        # Refresh derived body poses so the rendered hfield position matches the
        # updated data this frame (otherwise xpos lags mocap_pos by one step).
        mujoco.mj_kinematics(self.model, self.data)

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
    sim = Sim(xml_string, global_res=0.5)
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
            viewer.sync()
            viewer.update_hfield(sim.hfield_id)

            time_until_next_step = sim.dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
