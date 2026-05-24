from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp


@jax.tree_util.register_dataclass
@dataclass
class GridMap:
    """A grid map with multiple data layers.

    Indexing convention:
        - row (axis 0) maps to world +y from origin[1].
        - col (axis 1) maps to world +x from origin[0].
        - origin is at the bottom-left corner of cell (0, 0).
        - Cell center for integer indices (row, col) is:
            x = origin[0] + (col + 0.5) * resolution
            y = origin[1] + (row + 0.5) * resolution
    """
    origin: jnp.ndarray  # (2,) - [x, y] coordinates of the corner of cell (0, 0)
    resolution: float  # meters per cell
    layers: dict[str, jnp.ndarray]  # layer name to arrays with shape (H, W, ...)


@partial(jax.jit, static_argnames=["data_layer"])
def query_nearest(
    gridmap: GridMap, data_layer: str, query_points: jnp.ndarray
) -> jnp.ndarray:
    """Query nearest neighbor in a data layer.

    Args:
        gridmap: The grid map information.
        data_layer: The name of the data layer to query.
        query_points: The query points as a JAX array of shape (N, 2), where each
            row is [x, y] in world coordinates.

    Returns:
        The nearest neighbor values as a JAX array of shape (N,).
    """
    layer = gridmap.layers[data_layer]

    # Convert world coordinates to grid indices
    grid_pos = (query_points - gridmap.origin) / gridmap.resolution
    grid_idx = jnp.floor(grid_pos).astype(jnp.int32)

    # Clip to valid index range
    # grid_idx = [x, y] = [col, row], so bounds are [W-1, H-1]
    max_idx = jnp.array([layer.shape[1] - 1, layer.shape[0] - 1])
    grid_idx = jnp.clip(grid_idx, 0, max_idx)

    # Index into layer: grid_idx[:, 0] is x (column), grid_idx[:, 1] is y (row)
    return layer[grid_idx[:, 1], grid_idx[:, 0], ...]


@partial(jax.jit, static_argnames=["data_layer"])
def query_bilinear(
    gridmap: GridMap, data_layer: str, query_points: jnp.ndarray
) -> jnp.ndarray:
    """Query bilinear interpolation in a data layer.

    Args:
        gridmap: The grid map information.
        data_layer: The name of the data layer to query.
        query_points: The query points as a JAX array of shape (N, 2), where each
            row is [x, y] in world coordinates.

    Returns:
        The interpolated values as a JAX array of shape (N,).
    """
    layer = gridmap.layers[data_layer]

    # Convert world coordinates to grid coordinates (float)
    grid_pos = (query_points - gridmap.origin) / gridmap.resolution

    # Get integer indices for the 4 surrounding pixels
    lower_grid_idx = jnp.floor(grid_pos - 0.5).astype(jnp.int32)
    upper_grid_idx = lower_grid_idx + 1

    # Compute fractional part for interpolation (before clipping)
    alpha = grid_pos - (lower_grid_idx.astype(jnp.float32) + 0.5)
    alpha = jnp.clip(alpha, 0.0, 1.0)

    # Clip to valid range
    # grid_idx = [x, y] = [col, row], so bounds are [W-1, H-1]
    max_idx = jnp.array([layer.shape[1] - 1, layer.shape[0] - 1])
    lower_grid_idx = jnp.clip(lower_grid_idx, 0, max_idx)
    upper_grid_idx = jnp.clip(upper_grid_idx, 0, max_idx)

    # Sample the 4 neighbors
    i00 = layer[lower_grid_idx[:, 1], lower_grid_idx[:, 0], ...]
    i10 = layer[lower_grid_idx[:, 1], upper_grid_idx[:, 0], ...]
    i01 = layer[upper_grid_idx[:, 1], lower_grid_idx[:, 0], ...]
    i11 = layer[upper_grid_idx[:, 1], upper_grid_idx[:, 0], ...]

    # Bilinear interpolation
    num_cell_dims = layer.ndim - 2
    weight_shape = (-1,) + (1,) * num_cell_dims # for broadcasting with layer
    w0 = ((1 - alpha[:, 0]) * (1 - alpha[:, 1])).reshape(weight_shape)
    w1 = (alpha[:, 0] * (1 - alpha[:, 1])).reshape(weight_shape)
    w2 = ((1 - alpha[:, 0]) * alpha[:, 1]).reshape(weight_shape)
    w3 = (alpha[:, 0] * alpha[:, 1]).reshape(weight_shape)

    return w0 * i00 + w1 * i10 + w2 * i01 + w3 * i11


if __name__ == "__main__":
    import unittest

    class TestQueryBilinear(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            cls.layer = jnp.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
            cls.gridmap = GridMap(
                origin=jnp.array([3.0, 5.0]),
                resolution=0.1,
                layers={"data": cls.layer},
            )

        def _query(self, x, y):
            return float(query_bilinear(self.gridmap, "data", jnp.array([[x, y]]))[0])

        def test_cell_centers(self):
            self.assertAlmostEqual(self._query(3.05, 5.05), 0.0, places=5)
            self.assertAlmostEqual(self._query(3.15, 5.05), 1.0, places=5)
            self.assertAlmostEqual(self._query(3.05, 5.15), 2.0, places=5)
            self.assertAlmostEqual(self._query(3.15, 5.15), 3.0, places=5)

        def test_out_of_bounds_clipped(self):
            self.assertAlmostEqual(self._query(0.0, 0.0), 0.0, places=5)
            self.assertAlmostEqual(self._query(100.0, 0.0), 1.0, places=5)
            self.assertAlmostEqual(self._query(0.0, 100.0), 4.0, places=5)
            self.assertAlmostEqual(self._query(100.0, 100.0), 5.0, places=5)

        def test_midpoints(self):
            self.assertAlmostEqual(self._query(3.1, 5.05), 0.5, places=5)
            self.assertAlmostEqual(self._query(3.05, 5.1), 1.0, places=5)
            self.assertAlmostEqual(self._query(3.1, 5.1), 1.5, places=5)

    class TestQueryNearest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            cls.layer = jnp.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
            cls.gridmap = GridMap(
                origin=jnp.array([3.0, 5.0]),
                resolution=0.1,
                layers={"data": cls.layer},
            )

        def _query(self, x, y):
            return float(query_nearest(self.gridmap, "data", jnp.array([[x, y]]))[0])

        def test_containing_cell(self):
            self.assertAlmostEqual(self._query(3.05, 5.05), 0.0, places=5)
            self.assertAlmostEqual(self._query(3.15, 5.05), 1.0, places=5)
            self.assertAlmostEqual(self._query(3.05, 5.15), 2.0, places=5)
            self.assertAlmostEqual(self._query(3.15, 5.15), 3.0, places=5)

        def test_boundary(self):
            self.assertAlmostEqual(self._query(3.0, 5.0), 0.0, places=5)
            self.assertAlmostEqual(self._query(3.09, 5.09), 0.0, places=5)
            self.assertAlmostEqual(self._query(3.11, 5.09), 1.0, places=5)

        def test_out_of_bounds_clipped(self):
            self.assertAlmostEqual(self._query(0.0, 0.0), 0.0, places=5)
            self.assertAlmostEqual(self._query(100.0, 100.0), 5.0, places=5)

    unittest.main()
