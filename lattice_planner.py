import ctypes
import subprocess
from dataclasses import dataclass
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import numpy as np

import gridmap


@jax.tree_util.register_static
@dataclass(frozen=True)
class LatticeDefinition:
    state_dim_per_pos: Tuple[int, ...]
    # cost_fn(pos, non_pos_state, gridmap) -> per_neighbor_costs (shape: num_neighbors)
    cost_fn: Callable[[jax.Array, jax.Array, gridmap.GridMap], jax.Array]
    # shape: (num_states_per_pos, num_neighbors), flat index into state_dim_per_pos
    neighbor_non_pos_state_idx: jax.Array

    # shape: (num_states_per_pos, num_neighbors, 2) [rel x, rel y]
    neighbor_rel_pos: jax.Array

    # TODO might delete this
    def non_pos_state_to_index(self, non_pos_state: jax.Array) -> jax.Array:
        non_pos_tuple = tuple(
            non_pos_state[..., i] for i in range(len(self.state_dim_per_pos))
        )
        return jnp.ravel_multi_index(non_pos_tuple, self.state_dim_per_pos)


@jax.tree_util.register_static
@dataclass(frozen=True)
class Lattice:
    # 2d grid indexing follows gridmap convention [y, x]
    # lattice state is (posy, posx, non_pos_state) index into map of shape (*map_dim, num_states_per_pos)
    lattice_definition: LatticeDefinition
    map_dim: Tuple[int, int]  # [dim y, dim x]
    next_state_indices: jax.Array  # (num_states, num_neighbors)

    @staticmethod
    def compute_next_states_table(
        lattice_def: LatticeDefinition, map_dim: Tuple[int, int]
    ) -> jax.Array:
        num_states = jnp.prod(jnp.array(map_dim + lattice_def.state_dim_per_pos))
        state_indices = jnp.arange(num_states)

        def get_next_states(state_idx):
            dims = map_dim + (
                np.prod(np.array(lattice_def.state_dim_per_pos)).astype(jnp.int32),
            )
            posy, posx, non_pos_idx = jnp.unravel_index(state_idx, dims)
            rel_pos = lattice_def.neighbor_rel_pos[non_pos_idx]
            new_non_pos_idx = lattice_def.neighbor_non_pos_state_idx[non_pos_idx]
            new_pos = jnp.array([posx, posy]) + rel_pos

            def pos_non_pos_idx_to_index(pos, non_pos_idx):
                return jnp.ravel_multi_index(
                    (pos[1], pos[0], non_pos_idx), dims, mode="clip"
                )

            next_indices = jax.vmap(pos_non_pos_idx_to_index)(new_pos, new_non_pos_idx)
            max_idx = jnp.array([map_dim[1], map_dim[0]])
            out_of_bounds = jnp.any(new_pos < 0) | jnp.any(new_pos >= max_idx)
            next_indices = jnp.where(
                out_of_bounds, jnp.full_like(next_indices, state_idx), next_indices
            )
            return next_indices

        return jax.vmap(get_next_states)(state_indices)

    @classmethod
    def create(cls, lattice: LatticeDefinition, map_dim: Tuple[int, int]) -> "Lattice":
        return cls(
            lattice_definition=lattice,
            map_dim=map_dim,
            next_state_indices=cls.compute_next_states_table(lattice, map_dim),
        )

    @jax.jit
    def compute_costs(self, gridmap: gridmap.GridMap) -> jax.Array:
        """Compute the costs for each state in the lattice.
        Returns:
            costs: jax.Array of shape (num_states, num_neighbors)
        """
        num_states = np.prod(
            np.array(self.map_dim + self.lattice_definition.state_dim_per_pos)
        )
        state_indices = jnp.arange(num_states)

        def compute_costs_idx(idx: jax.Array) -> jax.Array:
            """idx shape (), returns cost shape (num_neighbors,)"""
            pos, non_pos_state = self.index_to_state(idx)
            return self.lattice_definition.cost_fn(pos, non_pos_state, gridmap)

        costs = jax.vmap(compute_costs_idx)(state_indices)
        return costs

    @jax.jit
    def state_to_index(self, pos: jax.Array, non_pos_state: jax.Array) -> jax.Array:
        posx = pos[..., 0]
        posy = pos[..., 1]
        dims = self.map_dim + self.lattice_definition.state_dim_per_pos
        non_pos_tuple = tuple(
            non_pos_state[..., i]
            for i in range(len(self.lattice_definition.state_dim_per_pos))
        )
        return jnp.ravel_multi_index((posy, posx) + non_pos_tuple, dims, mode="clip")

    @jax.jit
    def index_to_state(self, index: jax.Array) -> Tuple[jax.Array, jax.Array]:
        dims = self.map_dim + self.lattice_definition.state_dim_per_pos
        posy, posx, *non_pos_state = jnp.unravel_index(index, dims)
        if non_pos_state:
            non_pos_stacked = jnp.stack(non_pos_state, axis=-1)
        else:
            batch_shape = index.shape[:-1] if index.ndim > 0 else ()
            non_pos_stacked = jnp.empty(batch_shape + (0,), dtype=index.dtype)
        return jnp.stack([posx, posy], axis=-1), non_pos_stacked


def build_extension() -> None:
    result = subprocess.run(
        [
            "g++",
            "-shared",
            "-o",
            "lattice_dijkstra.so",
            "-fPIC",
            "lattice_dijkstra.cpp",
        ],
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        print(result.stdout)
    if result.stderr.strip():
        print(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"gcc compilation failed with exit code {result.returncode}")


build_extension()
lattice_dijkstra = ctypes.CDLL("lattice_dijkstra.so")
lattice_dijkstra.solve.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # float *edge_costs
    ctypes.POINTER(ctypes.c_int),  # int *next_states
    ctypes.POINTER(ctypes.c_float),  # float *dist_out
    ctypes.c_int,  # int num_states
    ctypes.c_int,  # int num_edges
    ctypes.POINTER(ctypes.c_int),  # int *start_states
    ctypes.POINTER(ctypes.c_float),  # float *start_values
    ctypes.c_int,  # int num_start_states
    ctypes.c_int,  # int terminal_state
]
lattice_dijkstra.solve.restype = ctypes.c_bool


def dijkstra(
    initial_states: np.ndarray,
    initial_costs: np.ndarray,
    terminal_state: int,
    costs: np.ndarray,
    next_states: np.ndarray,
) -> Tuple[np.ndarray, bool]:
    dist_out = np.empty(costs.shape[0], dtype=np.float32)
    assert costs.shape[0] == dist_out.shape[0]
    assert costs.shape[0] == next_states.shape[0]
    assert costs.shape[1] == next_states.shape[1]
    assert costs.dtype == np.float32
    assert next_states.dtype == np.int32
    assert dist_out.dtype == np.float32
    assert initial_states.dtype == np.int32
    success = lattice_dijkstra.solve(
        np.ascontiguousarray(costs, dtype=np.float32).ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        ),
        np.ascontiguousarray(next_states, dtype=np.int32).ctypes.data_as(
            ctypes.POINTER(ctypes.c_int)
        ),
        np.ascontiguousarray(dist_out, dtype=np.float32).ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        ),
        costs.shape[0],
        costs.shape[1],
        np.ascontiguousarray(initial_states, dtype=np.int32).ctypes.data_as(
            ctypes.POINTER(ctypes.c_int)
        ),
        np.ascontiguousarray(initial_costs, dtype=np.float32).ctypes.data_as(
            ctypes.POINTER(ctypes.c_float)
        ),
        initial_states.shape[0],
        terminal_state,
    )
    return dist_out, success


def plan(
    lattice: Lattice,
    map: gridmap.GridMap,
    initial_states: np.ndarray,
    initial_costs: np.ndarray,
    terminal_state: np.ndarray,
) -> gridmap.GridMap:
    costs = lattice.compute_costs(map)
    initial_state_indices = lattice.state_to_index(
        initial_states[..., :2], initial_states[..., 2:]
    )
    terminal_state_index = lattice.state_to_index(
        terminal_state[..., :2], terminal_state[..., 2:]
    )
    cost_to_go, success = dijkstra(
        initial_state_indices,
        initial_costs,
        terminal_state_index,
        np.array(costs),
        np.array(lattice.next_state_indices),
    )
    ctg_shape = lattice.map_dim + lattice.lattice_definition.state_dim_per_pos
    cost_to_go_map = gridmap.GridMap(
        origin=map.origin,
        resolution=map.resolution,
        layers={"cost_to_go": jnp.asarray(cost_to_go).reshape(ctg_shape)},
    )
    return cost_to_go_map


def make_grid_2d_8conn_lattice(cost_layer="traversability"):
    neighbor_pos = jnp.array(
        [
            [1, 0],
            [1, 1],
            [0, 1],
            [-1, 1],
            [-1, 0],
            [-1, -1],
            [0, -1],
            [1, -1],
        ],
        dtype=jnp.int32,
    ).reshape((1, 8, 2))
    neighbor_dist = jnp.linalg.norm(neighbor_pos[0].astype(jnp.float32), axis=1)

    def cost_fn(pos, non_pos, gm):
        query_pos = pos[None, :] + neighbor_pos[0]  # query is shape(8, 2)
        cost = gm.layers[cost_layer][query_pos[:, 1], query_pos[:, 0]]
        return cost * neighbor_dist * gm.resolution

    return LatticeDefinition(
        state_dim_per_pos=(),  # no non-positional state
        cost_fn=cost_fn,
        neighbor_non_pos_state_idx=jnp.zeros((1, 8), dtype=jnp.int32),
        neighbor_rel_pos=neighbor_pos,
    )


# --------------------------------------------------
def dummy_cost_fn(pos, non_pos, gm):
    return jnp.zeros(2)


lat_def = LatticeDefinition(
    state_dim_per_pos=(3, 4),  # 2D non-positional state
    cost_fn=dummy_cost_fn,
    neighbor_non_pos_state_idx=jnp.zeros((12, 2), dtype=jnp.int32),
    neighbor_rel_pos=jnp.zeros((12, 2, 2), dtype=jnp.int32),
)

lattice = Lattice.create(lat_def, (10, 12))

pos = jnp.array([3, 7])
non_pos = jnp.array([1, 2])

idx = lattice.state_to_index(pos, non_pos)
print("state_to_index:", idx)

r_pos, r_nps = lattice.index_to_state(idx)
print("index_to_state  pos:", r_pos)
print("index_to_state  non_pos:", r_nps)

# Verify round-trip
assert jnp.array_equal(r_pos, pos), f"pos mismatch: {r_pos} != {pos}"
assert jnp.array_equal(r_nps, non_pos), f"non_pos mismatch: {r_nps} != {non_pos}"
print("Round-trip OK!")

# ----------------- Additional Tests for Batching and Vmap -----------------
print("\n--- Running Batched & Vmap Tests ---")

# Test 1: Directly passing a batch of positions and non-positions
pos_batch = jnp.array([[3, 7], [1, 2], [5, 9]])
non_pos_batch = jnp.array([[1, 2], [0, 3], [2, 1]])

idx_batch = lattice.state_to_index(pos_batch, non_pos_batch)
print("Batched state_to_index:", idx_batch)
assert idx_batch.shape == (3,)

r_pos_batch, r_nps_batch = lattice.index_to_state(idx_batch)
print("Batched index_to_state pos:\n", r_pos_batch)
print("Batched index_to_state non_pos:\n", r_nps_batch)
assert r_pos_batch.shape == (3, 2)
assert r_nps_batch.shape == (3, 2)

assert jnp.array_equal(r_pos_batch, pos_batch), "Batched pos mismatch"
assert jnp.array_equal(r_nps_batch, non_pos_batch), "Batched non_pos mismatch"
print("Direct batch round-trip OK!")

# Test 2: Using vmap with state_to_index and index_to_state
vmap_state_to_index = jax.vmap(lattice.state_to_index)
vmap_index_to_state = jax.vmap(lattice.index_to_state)

v_idx = vmap_state_to_index(pos_batch, non_pos_batch)
assert jnp.array_equal(v_idx, idx_batch)

v_pos, v_nps = vmap_index_to_state(v_idx)
assert jnp.array_equal(v_pos, pos_batch)
assert jnp.array_equal(v_nps, non_pos_batch)
print("Vmapped round-trip OK!")

# Test 3: Multi-dimensional batch (e.g., shape (2, 3))
pos_2d_batch = jnp.array([[[3, 7], [1, 2], [5, 9]], [[0, 0], [4, 4], [11, 9]]])
non_pos_2d_batch = jnp.array([[[1, 2], [0, 3], [2, 1]], [[0, 0], [2, 2], [1, 3]]])

idx_2d_batch = lattice.state_to_index(pos_2d_batch, non_pos_2d_batch)
assert idx_2d_batch.shape == (2, 3)

r_pos_2d, r_nps_2d = lattice.index_to_state(idx_2d_batch)
assert r_pos_2d.shape == (2, 3, 2)
assert r_nps_2d.shape == (2, 3, 2)

assert jnp.array_equal(r_pos_2d, pos_2d_batch)
assert jnp.array_equal(r_nps_2d, non_pos_2d_batch)
print("2D-batched round-trip OK!")

# Test 4: non_pos_state_to_index batched
non_pos_idx_single = lat_def.non_pos_state_to_index(non_pos)
print("Single non_pos_state_to_index:", non_pos_idx_single)
assert non_pos_idx_single == 6

non_pos_idx_batch = lat_def.non_pos_state_to_index(non_pos_batch)
print("Batched non_pos_state_to_index:", non_pos_idx_batch)
assert jnp.array_equal(non_pos_idx_batch, jnp.array([6, 3, 9]))
print("non_pos_state_to_index batching OK!")

# Test next_state_indices table shape
print("\n--- Checking next_state_indices table ---")
print("next_state_indices shape:", lattice.next_state_indices.shape)
expected_shape = (10 * 12 * 12, 2)
assert lattice.next_state_indices.shape == expected_shape, (
    f"shape mismatch: {lattice.next_state_indices.shape}"
)

# Test compute_costs
print("\n--- Checking compute_costs ---")
costs = lattice.compute_costs(None)
print("compute_costs shape:", costs.shape)
assert costs.shape == (10 * 12 * 12, 2), f"costs shape mismatch: {costs.shape}"
print("compute_costs OK!")

print("All tests passed!")

# ----------------- Test plan -----------------
print("\n--- Testing plan ---")

gm = gridmap.GridMap(
    origin=jnp.array([0.0, 0.0]),
    resolution=1.0,
    layers={"data": jnp.zeros((10, 12))},
)

# initial_states: [x, y, non_pos_0, non_pos_1]
initial_states = np.array([[2, 2, 0, 0]], dtype=np.int32)
initial_costs = np.array([0.0], dtype=np.float32)
terminal_state = np.array([7, 7, 2, 3], dtype=np.int32)

ctg_map = plan(lattice, gm, initial_states, initial_costs, terminal_state)
print("cost_to_go shape:", ctg_map.layers["cost_to_go"].shape)
assert ctg_map.layers["cost_to_go"].shape == (10, 12, 3, 4), (
    f"unexpected shape: {ctg_map.layers['cost_to_go'].shape}"
)
assert ctg_map.origin is not None
assert ctg_map.resolution == 1.0
print("plan() OK!")

print("All tests passed!")

# ----------------- Test 8-connected lattice -----------------
print("\n--- Testing 8-connected lattice ---")

lat_8conn = make_grid_2d_8conn_lattice(cost_layer="traversability")
lattice_8conn = Lattice.create(lat_8conn, (10, 12))

gm_8conn = gridmap.GridMap(
    origin=jnp.array([0.0, 0.0]),
    resolution=1.0,
    layers={"traversability": jnp.ones((10, 12), dtype=jnp.float32)},
)

initial_states_8conn = np.array([[2, 2]], dtype=np.int32)
initial_costs_8conn = np.array([0.0], dtype=np.float32)
terminal_state_8conn = np.array([7, 7], dtype=np.int32)

ctg_map_8conn = plan(
    lattice_8conn,
    gm_8conn,
    initial_states_8conn,
    initial_costs_8conn,
    terminal_state_8conn,
)
# ----------------- Bug trap test with visualization -----------------
print("\n--- Testing bug trap escape ---")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ASCII map: #=obstacle, ' '=free, s=start, g=goal
bugtrap_ascii = r"""
###########
#         #
#  #      #
#  #   #  #
#  # s #  #
#  #####  #
#         #
#    g    #
###########
"""
rows = bugtrap_ascii.strip().split("\n")
H, W = len(rows), len(rows[0])
print(f"Bug trap map: {H}x{W}")

trav = jnp.ones((H, W), dtype=jnp.float32)
start_pos = None  # (row, col) of 's'
goal_pos = None  # (row, col) of 'g'
for r in range(H):
    for c in range(W):
        ch = rows[r][c]
        if ch == "#":
            trav = trav.at[r, c].set(jnp.inf)
        elif ch == "s":
            start_pos = (r, c)
        elif ch == "g":
            goal_pos = (r, c)

print(
    f"Start (s) at row={start_pos[0]}, col={start_pos[1]}  -> [x={start_pos[1]}, y={start_pos[0]}]"
)
print(
    f"Goal  (g) at row={goal_pos[0]}, col={goal_pos[1]}  -> [x={goal_pos[1]}, y={goal_pos[0]}]"
)

# pos is [x, y] = [col, row]
# Agent starts at 'g', needs to reach 's'
init_st = np.array([[goal_pos[1], goal_pos[0]]], dtype=np.int32)
term_st = np.array([start_pos[1], start_pos[0]], dtype=np.int32)
init_cost = np.array([0.0], dtype=np.float32)

gm_trap = gridmap.GridMap(
    origin=jnp.array([0.0, 0.0]),
    resolution=1.0,
    layers={"traversability": trav},
)

# Build lattice for this map size
lattice_trap = Lattice.create(lat_8conn, (H, W))

ctg = plan(lattice_trap, gm_trap, init_st, init_cost, term_st)
ctg_layer = np.array(ctg.layers["cost_to_go"])
print(f"Bug trap cost_to_go shape: {ctg_layer.shape}")
print(f"Cost at start (s): {ctg_layer[start_pos[0], start_pos[1]]}")
print(f"Cost at goal (g): {ctg_layer[goal_pos[0], goal_pos[1]]}")

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
obs = np.isinf(trav).astype(float)
# Mask obstacles with NaN so the colormap doesn't show them
masked_ctg = np.where(obs > 0, np.nan, ctg_layer)
# Clamp color range to [0, max] so the gradient is vivid
max_cost = float(np.nanmax(masked_ctg))
# Use black axis background; NaN cells show through as black (obstacles)
ax.set_facecolor("black")
im = ax.imshow(masked_ctg, cmap="viridis", origin="upper", vmin=0, vmax=max_cost)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Cost-to-go")
# Robot is at 's' (start), wants to reach 'g' (goal)
# We plan in reverse from goal -> start to build cost-to-go
sx, sy = start_pos[1], start_pos[0]  # robot start position
gx, gy = goal_pos[1], goal_pos[0]  # robot goal position
ax.plot(
    sx,
    sy,
    "go",
    markersize=14,
    label=f"Start (s) cost={ctg_layer[start_pos[0], start_pos[1]]:.0f}",
    zorder=5,
)
ax.plot(
    gx,
    gy,
    "r*",
    markersize=16,
    label=f"Goal (g) cost={ctg_layer[goal_pos[0], goal_pos[1]]:.0f}",
    zorder=5,
)
ax.set_title(
    "Bug Trap: Cost-to-Go\n(robot starts at s, plans to g, cost computed in reverse)"
)
ax.legend()
ax.set_xlabel("x")
ax.set_ylabel("y")
fig.savefig("bugtrap_ctg.png", dpi=150, bbox_inches="tight")
print("Saved bugtrap_ctg.png")
plt.close(fig)

print("Bug trap test done!")
