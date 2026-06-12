import ctypes
import subprocess
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import numpy as np

import gridmap


@partial(
    jax.tree_util.register_dataclass,
    meta_fields=["state_dim_per_pos", "cost_fn"],
    data_fields=["neighbor_non_pos_state_idx", "neighbor_rel_pos"],
)
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


@partial(
    jax.tree_util.register_dataclass,
    meta_fields=["map_dim"],
    data_fields=["next_state_indices", "lattice_definition"],
)
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


def load_lattice_dijkstra() -> ctypes.CDLL:
    current_dir = Path(__file__).parent.resolve()
    cpp_source = current_dir / "lattice_dijkstra.cpp"
    binary_target = current_dir / "lattice_dijkstra.so"

    if not cpp_source.exists():
        raise FileNotFoundError(f"Missing source file: {cpp_source}")

    if (
        not binary_target.exists()
        or cpp_source.stat().st_mtime > binary_target.stat().st_mtime
    ):
        print(f"Compiling extension: {binary_target.name}...")
        result = subprocess.run(
            [
                "g++",
                "-O3",
                "-shared",
                "-fPIC",
                "-o",
                str(binary_target),
                str(cpp_source),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"C++ Build failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )

    return ctypes.CDLL(str(binary_target))


lattice_dijkstra = load_lattice_dijkstra()
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


if __name__ == "__main__":
    import unittest

    def dummy_cost_fn(pos, non_pos, gm):
        return jnp.zeros(2)

    class TestStateRoundTrip(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            cls.lat_def = LatticeDefinition(
                state_dim_per_pos=(3, 4),
                cost_fn=dummy_cost_fn,
                neighbor_non_pos_state_idx=jnp.zeros((12, 2), dtype=jnp.int32),
                neighbor_rel_pos=jnp.zeros((12, 2, 2), dtype=jnp.int32),
            )
            cls.lattice = Lattice.create(cls.lat_def, (10, 12))

        def test_single(self):
            pos = jnp.array([3, 7])
            non_pos = jnp.array([1, 2])
            idx = self.lattice.state_to_index(pos, non_pos)
            r_pos, r_nps = self.lattice.index_to_state(idx)
            self.assertTrue(jnp.array_equal(r_pos, pos))
            self.assertTrue(jnp.array_equal(r_nps, non_pos))

        def test_batched(self):
            pos = jnp.array([[3, 7], [1, 2], [5, 9]])
            non_pos = jnp.array([[1, 2], [0, 3], [2, 1]])
            idx = self.lattice.state_to_index(pos, non_pos)
            self.assertEqual(idx.shape, (3,))
            r_pos, r_nps = self.lattice.index_to_state(idx)
            self.assertTrue(jnp.array_equal(r_pos, pos))
            self.assertTrue(jnp.array_equal(r_nps, non_pos))

        def test_2d_batched(self):
            pos = jnp.array([[[3, 7], [1, 2], [5, 9]], [[0, 0], [4, 4], [11, 9]]])
            non_pos = jnp.array([[[1, 2], [0, 3], [2, 1]], [[0, 0], [2, 2], [1, 3]]])
            idx = self.lattice.state_to_index(pos, non_pos)
            self.assertEqual(idx.shape, (2, 3))
            r_pos, r_nps = self.lattice.index_to_state(idx)
            self.assertTrue(jnp.array_equal(r_pos, pos))
            self.assertTrue(jnp.array_equal(r_nps, non_pos))

    class TestNonPosStateToIndex(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            cls.lat_def = LatticeDefinition(
                state_dim_per_pos=(3, 4),
                cost_fn=dummy_cost_fn,
                neighbor_non_pos_state_idx=jnp.zeros((12, 2), dtype=jnp.int32),
                neighbor_rel_pos=jnp.zeros((12, 2, 2), dtype=jnp.int32),
            )

        def test_single(self):
            self.assertEqual(self.lat_def.non_pos_state_to_index(jnp.array([1, 2])), 6)

        def test_batched(self):
            result = self.lat_def.non_pos_state_to_index(
                jnp.array([[1, 2], [0, 3], [2, 1]])
            )
            self.assertTrue(jnp.array_equal(result, jnp.array([6, 3, 9])))

    class TestLatticeShape(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            cls.lat_def = LatticeDefinition(
                state_dim_per_pos=(3, 4),
                cost_fn=dummy_cost_fn,
                neighbor_non_pos_state_idx=jnp.zeros((12, 2), dtype=jnp.int32),
                neighbor_rel_pos=jnp.zeros((12, 2, 2), dtype=jnp.int32),
            )
            cls.lattice = Lattice.create(cls.lat_def, (10, 12))

        def test_next_state_indices_shape(self):
            self.assertEqual(self.lattice.next_state_indices.shape, (10 * 12 * 12, 2))

        def test_compute_costs_shape(self):
            gm = gridmap.GridMap(
                origin=jnp.array([0.0, 0.0]),
                resolution=1.0,
                layers={"data": jnp.zeros((10, 12))},
            )
            costs = self.lattice.compute_costs(gm)
            self.assertEqual(costs.shape, (10 * 12 * 12, 2))

    class TestPlan(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            cls.lat_def = LatticeDefinition(
                state_dim_per_pos=(3, 4),
                cost_fn=dummy_cost_fn,
                neighbor_non_pos_state_idx=jnp.zeros((12, 2), dtype=jnp.int32),
                neighbor_rel_pos=jnp.zeros((12, 2, 2), dtype=jnp.int32),
            )
            cls.lattice = Lattice.create(cls.lat_def, (10, 12))

        def test_plan_output(self):
            gm = gridmap.GridMap(
                origin=jnp.array([0.0, 0.0]),
                resolution=1.0,
                layers={"data": jnp.zeros((10, 12))},
            )
            initial_states = np.array([[2, 2, 0, 0]], dtype=np.int32)
            initial_costs = np.array([0.0], dtype=np.float32)
            terminal_state = np.array([7, 7, 2, 3], dtype=np.int32)
            ctg_map = plan(
                self.lattice, gm, initial_states, initial_costs, terminal_state
            )
            self.assertEqual(ctg_map.layers["cost_to_go"].shape, (10, 12, 3, 4))
            self.assertEqual(ctg_map.resolution, 1.0)

    class TestPlan8Connected(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            cls.lat_8conn = make_grid_2d_8conn_lattice(cost_layer="traversability")
            cls.lattice_8conn = Lattice.create(cls.lat_8conn, (10, 12))

        def test_plan(self):
            gm = gridmap.GridMap(
                origin=jnp.array([0.0, 0.0]),
                resolution=1.0,
                layers={"traversability": jnp.ones((10, 12), dtype=jnp.float32)},
            )
            initial_states = np.array([[2, 2]], dtype=np.int32)
            initial_costs = np.array([0.0], dtype=np.float32)
            terminal_state = np.array([7, 7], dtype=np.int32)
            ctg_map = plan(
                self.lattice_8conn,
                gm,
                initial_states,
                initial_costs,
                terminal_state,
            )
            self.assertEqual(ctg_map.layers["cost_to_go"].shape, (10, 12))

        def test_bug_trap(self):
            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
            except ImportError:
                self.skipTest("matplotlib not installed")

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

            trav = jnp.ones((H, W), dtype=jnp.float32)
            start_pos = goal_pos = None
            for r in range(H):
                for c in range(W):
                    ch = rows[r][c]
                    if ch == "#":
                        trav = trav.at[r, c].set(jnp.inf)
                    elif ch == "s":
                        start_pos = (r, c)
                    elif ch == "g":
                        goal_pos = (r, c)
            assert start_pos is not None and goal_pos is not None

            init_st = np.array([[goal_pos[1], goal_pos[0]]], dtype=np.int32)
            term_st = np.array([start_pos[1], start_pos[0]], dtype=np.int32)
            init_cost = np.array([0.0], dtype=np.float32)

            gm_trap = gridmap.GridMap(
                origin=jnp.array([0.0, 0.0]),
                resolution=1.0,
                layers={"traversability": trav},
            )
            lattice_trap = Lattice.create(self.lat_8conn, (H, W))
            ctg = plan(lattice_trap, gm_trap, init_st, init_cost, term_st)

            ctg_layer = np.array(ctg.layers["cost_to_go"])
            self.assertFalse(
                np.isinf(ctg_layer[start_pos[0], start_pos[1]]),
                "start should be reachable",
            )

            fig, ax = plt.subplots(figsize=(6, 5))
            obs = np.isinf(trav).astype(float)
            masked_ctg = np.where(obs > 0, np.nan, ctg_layer)
            max_cost = float(np.nanmax(masked_ctg))
            ax.imshow(masked_ctg, cmap="viridis", origin="upper", vmin=0, vmax=max_cost)
            ax.plot(start_pos[1], start_pos[0], "go", markersize=12, label="start")
            ax.plot(goal_pos[1], goal_pos[0], "r*", markersize=14, label="goal")
            ax.legend()
            fig.savefig("bugtrap_ctg.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    unittest.main()
