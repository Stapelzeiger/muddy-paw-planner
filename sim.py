import functools
import time

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np
from mujoco import mjx

import gridmap

ball_model = """
<mujoco>
    <worldbody>
        <light pos="0 0 10" dir="0 0 -1" directional="true"/>

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
    return (np.cos(x) + np.sin(2 * y) + 0.1 * x - 100) * 0.05


def add_checkerboard(spec, extent, z):
    """Add a visualization-only checkerboard plane spanning the full map, below the terrain."""
    tex = spec.add_texture(name="checkerboard")
    tex.type = mujoco.mjtTexture.mjTEXTURE_2D
    tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
    tex.rgb1 = [0.2, 0.3, 0.4]
    tex.rgb2 = [0.1, 0.15, 0.2]
    tex.width = 512
    tex.height = 512
    mat = spec.add_material(name="checkerboard_mat", texrepeat=[10, 10])
    mat.textures[1] = "checkerboard"

    spec.worldbody.add_geom(
        name="checkerboard",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[extent, extent, 0.1],
        pos=[0, 0, z],
        material="checkerboard_mat",
        contype=0,
        conaffinity=0,
    )


class Sim:
    def __init__(
        self,
        xml_string,
        robot_body_name,
        dt=0.05,
        global_extent=50.0,
        global_res=0.1,
        hfield_nrow=101,
        hfield_ncol=101,
    ):
        mj_spec = mujoco.MjSpec.from_string(xml_string)

        mj_spec = self._build_hfields(
            mj_spec, global_extent, global_res, hfield_nrow, hfield_ncol
        )
        self.model = mj_spec.compile()
        self.data = mujoco.MjData(self.model)
        self.substeps = max(1, round(dt / self.model.opt.timestep))

        self.robot_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, robot_body_name
        )
        self.hfield_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_HFIELD, "terrain"
        )
        self.floor_mocap_id = self.model.body_mocapid[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "floor_body")
        ]
        self.hfield_adr = self.model.hfield_adr[self.hfield_id]

        self.mjx_model = mjx.put_model(self.model)
        self._shfit_local_collision_terrain()
        self._mjx_state = mjx.make_data(self.mjx_model)

    @property
    def dt(self):
        return self.substeps * self.model.opt.timestep

    def _build_hfields(self, mj_spec, extent, res, hfield_nrow, hfield_ncol):
        self.global_extent = extent
        self.global_res = res

        global_x = np.arange(-extent, extent, res)
        global_y = np.arange(-extent, extent, res)
        X, Y = np.meshgrid(global_x, global_y)
        Z = elevation(X, Y)

        origin_idx = int(round(extent / res))
        Z = Z - Z[origin_idx, origin_idx]
        self.global_min = float(Z.min())
        self.global_range = float(Z.max() - Z.min())
        self.global_hmap = (Z - self.global_min) / self.global_range

        self.hfield_nrow = min(hfield_nrow, self.global_hmap.shape[0])
        self.hfield_ncol = min(hfield_ncol, self.global_hmap.shape[1])
        base = 0.1
        hx = (self.hfield_ncol - 1) * res / 2
        hy = (self.hfield_nrow - 1) * res / 2

        global_hx = (global_x[-1] - global_x[0]) / 2
        global_hy = (global_y[-1] - global_y[0]) / 2
        global_cx = (global_x[0] + global_x[-1]) / 2
        global_cy = (global_y[0] + global_y[-1]) / 2

        add_checkerboard(mj_spec, extent=extent, z=self.global_min)

        nrow, ncol = self.global_hmap.shape
        hfield = mj_spec.add_hfield(
            name="terrain_global",
            nrow=nrow,
            ncol=ncol,
            size=[global_hx, global_hy, self.global_range, base],
        )
        hfield.userdata = self.global_hmap.flatten().astype(np.float32)
        mj_spec.worldbody.add_geom(
            name="terrain_global",
            type=mujoco.mjtGeom.mjGEOM_HFIELD,
            hfieldname="terrain_global",
            rgba=[0.5, 0.42, 0.32, 1],
            pos=[global_cx, global_cy, self.global_min - 0.05],
            contype=0,
            conaffinity=0,
        )

        hfield = mj_spec.add_hfield(
            name="terrain",
            nrow=self.hfield_nrow,
            ncol=self.hfield_ncol,
            size=[hx, hy, self.global_range, base],
        )
        hfield.userdata = np.zeros(
            self.hfield_nrow * self.hfield_ncol, dtype=np.float32
        )
        body = mj_spec.worldbody.add_body(name="floor_body", mocap=True)
        body.add_geom(
            name="floor",
            type=mujoco.mjtGeom.mjGEOM_HFIELD,
            hfieldname="terrain",
            rgba=[0.3, 0.5, 0.3, 1],
        )

        return mj_spec

    def _shfit_local_collision_terrain(self):
        """Slice the global heightmap around the robot and sync it into data and mjx."""
        robot_x, robot_y = self.data.xpos[self.robot_body_id][:2]
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

    def render_local_map(self, eye_height=0.5):
        """Raytrace a local elevation map around the robot using DDA.

        Returns a GridMap with the same extent as the local hfield.  Rays
        start at the robot position (+ eye_height in z) and travel to each
        local map cell.  A cell is marked occluded if terrain blocks the ray
        before it reaches the cell.

        Layers:
            elevation  – terrain height at visible cells, ``inf`` elsewhere
            occlusion  – 0.0 = visible, 1.0 = occluded
        """
        robot_xy, robot_z = (
            self.data.xpos[self.robot_body_id][:2],
            self.data.xpos[self.robot_body_id][2],
        )
        robot_pos = jnp.append(
            jnp.asarray(robot_xy, dtype=jnp.float32),
            jnp.float32(robot_z + eye_height),
        )
        global_elev = jnp.asarray(
            self.global_hmap * self.global_range + self.global_min
        )
        return _dda_raytrace(
            robot_pos,
            self.hfield_ncol,
            self.global_res,
            self.global_extent,  # half-width of the global heightfield
            global_elev,
        )

    def step(self, action):
        self.data.ctrl[:] = action
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
        self._shfit_local_collision_terrain()

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


@functools.partial(
    jax.jit, static_argnames=("map_size", "global_res", "global_half_width")
)
def _dda_raytrace(robot_pos, map_size, global_res, global_half_width, global_elev):
    """JIT-compiled DDA raytrace over a heightfield.

    Args:
        robot_pos:        (3,) array [x, y, z] of the ray origin in world frame.
        map_size:         number of cells per side of the square local output grid.
        global_res:       cell size (m) of the global heightfield.
        global_half_width: half-width (m) of the global heightfield.
        global_elev:      (H, W) array of terrain elevations in world frame.
    """
    robot_x, robot_y, eye_z = robot_pos[0], robot_pos[1], robot_pos[2]
    gw, gh = global_elev.shape

    half = (map_size - 1) * global_res / 2
    origin_x = robot_x - half
    origin_y = robot_y - half

    xs = origin_x + (jnp.arange(map_size) + 0.5) * global_res
    ys = origin_y + (jnp.arange(map_size) + 0.5) * global_res
    tx = jnp.broadcast_to(xs.reshape(1, map_size), (map_size, map_size))
    ty = jnp.broadcast_to(ys.reshape(map_size, 1), (map_size, map_size))

    gcol = jnp.round((tx + global_half_width) / global_res).astype(jnp.int32)
    grow = jnp.round((ty + global_half_width) / global_res).astype(jnp.int32)
    valid = (gcol >= 0) & (gcol < gw) & (grow >= 0) & (grow < gh)
    gcol_clip = jnp.clip(gcol, 0, gw - 1)
    grow_clip = jnp.clip(grow, 0, gh - 1)
    tz = jnp.where(valid, global_elev[grow_clip, gcol_clip], jnp.nan)

    dx = tx - robot_x
    dy = ty - robot_y
    dz = tz - eye_z

    start_col = jnp.floor((robot_x + global_half_width) / global_res).astype(jnp.int32)
    start_row = jnp.floor((robot_y + global_half_width) / global_res).astype(jnp.int32)

    eps = 1e-10
    step_x = jnp.where(dx > eps, 1, jnp.where(dx < -eps, -1, 0)).astype(jnp.int32)
    step_y = jnp.where(dy > eps, 1, jnp.where(dy < -eps, -1, 0)).astype(jnp.int32)

    t_delta_x = jnp.where(jnp.abs(dx) > eps, jnp.abs(global_res / dx), jnp.inf)
    t_delta_y = jnp.where(jnp.abs(dy) > eps, jnp.abs(global_res / dy), jnp.inf)

    next_x_pos = -global_half_width + (start_col + 1) * global_res
    next_x_neg = -global_half_width + start_col * global_res
    next_y_pos = -global_half_width + (start_row + 1) * global_res
    next_y_neg = -global_half_width + start_row * global_res

    t_max_x = jnp.where(
        dx > eps,
        (next_x_pos - robot_x) / dx,
        jnp.where(dx < -eps, (next_x_neg - robot_x) / dx, jnp.inf),
    )
    t_max_y = jnp.where(
        dy > eps,
        (next_y_pos - robot_y) / dy,
        jnp.where(dy < -eps, (next_y_neg - robot_y) / dy, jnp.inf),
    )

    col = jnp.full((map_size, map_size), start_col, dtype=jnp.int32)
    row = jnp.full((map_size, map_size), start_row, dtype=jnp.int32)

    occlusion = jnp.ones((map_size, map_size), dtype=jnp.float32)
    active = valid

    same_cell = active & (col == gcol) & (row == grow)
    occlusion = jnp.where(same_cell, 0.0, occlusion)
    active = active & ~same_cell

    def body_fn(_, carry):
        col, row, t_max_x, t_max_y, occlusion, active = carry

        step_x_mask = active & (t_max_x < t_max_y)
        t = jnp.where(step_x_mask, t_max_x, t_max_y)

        arrived = active & (t >= 1.0)
        occlusion = jnp.where(arrived, 0.0, occlusion)

        cc = jnp.clip(col, 0, gw - 1)
        cr = jnp.clip(row, 0, gh - 1)
        ray_z = eye_z + t * dz
        hit = active & (ray_z < global_elev[cr, cc])

        oob = active & ((col < 0) | (col >= gw) | (row < 0) | (row >= gh))

        terminated = arrived | hit | oob
        active = active & ~terminated

        col = jnp.where(active & step_x_mask, col + step_x, col)
        row = jnp.where(active & ~step_x_mask, row + step_y, row)
        t_max_x = jnp.where(active & step_x_mask, t_max_x + t_delta_x, t_max_x)
        t_max_y = jnp.where(active & ~step_x_mask, t_max_y + t_delta_y, t_max_y)

        return col, row, t_max_x, t_max_y, occlusion, active

    init = (col, row, t_max_x, t_max_y, occlusion, active)
    col, row, t_max_x, t_max_y, occlusion, active = jax.lax.fori_loop(
        0, map_size * 2, body_fn, init
    )

    elevation = jnp.where(occlusion < 0.5, tz, jnp.nan)

    return gridmap.GridMap(
        origin=jnp.array([origin_x, origin_y]),
        resolution=float(global_res),
        layers={"elevation": elevation, "occlusion": occlusion},
    )


if __name__ == "__main__":
    sim = Sim(ball_model, robot_body_name="ball")

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

            horizon = int(0.5 / sim.dt)
            trajectory = rollout(
                sim.get_mjx_model(),
                sim.get_mjx_state(),
                jnp.zeros((horizon, sim.model.nu)),
            )
            positions = np.asarray(trajectory.xpos)[:, sim.robot_body_id]
            draw_trajectory(viewer, positions)

            sim.step(np.zeros(sim.model.nu))
            viewer.sync()
            viewer.update_hfield(sim.hfield_id)

            time_until_next_step = sim.dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
