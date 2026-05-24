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


@partial(jax.jit, static_argnames=["data_layer"])
def query_bilinear_heading(
    gridmap: GridMap, data_layer: str, query_points: jnp.ndarray
) -> jnp.ndarray:
    """Query with bilinear spatial + linear heading interpolation.

    The layer is expected to have shape (H, W, num_headings, ...).

    Args:
        gridmap: The grid map information.
        data_layer: The name of the data layer to query.
        query_points: Shape (N, 3), where each row is [x, y, heading] with
            heading in [0, 2*pi) radians. Heading 0 maps to index 0.

    Returns:
        Interpolated values as a JAX array of shape (N, ...).
    """
    layer = gridmap.layers[data_layer]
    num_headings = layer.shape[2]

    # Convert world coordinates to grid indices
    grid_pos = (query_points[:, :2] - gridmap.origin) / gridmap.resolution
    lower_grid_idx = jnp.floor(grid_pos - 0.5).astype(jnp.int32)
    upper_grid_idx = lower_grid_idx + 1

    # Compute fractional part for spatial interpolation (before clipping)
    alpha = grid_pos - (lower_grid_idx.astype(jnp.float32) + 0.5)
    alpha = jnp.clip(alpha, 0.0, 1.0)

    # Clip spatial indices
    max_idx = jnp.array([layer.shape[1] - 1, layer.shape[0] - 1])
    lower_grid_idx = jnp.clip(lower_grid_idx, 0, max_idx)
    upper_grid_idx = jnp.clip(upper_grid_idx, 0, max_idx)

    # Heading interpolation (circular)
    heading = query_points[:, 2]
    heading_idx = heading / (2.0 * jnp.pi) * num_headings
    lower_heading_idx = jnp.floor(heading_idx).astype(jnp.int32) % num_headings
    upper_heading_idx = (lower_heading_idx + 1) % num_headings
    heading_alpha = (heading_idx - jnp.floor(heading_idx)).astype(jnp.float32)

    # Sample at lower heading for all 4 spatial corners
    i00_lo = layer[lower_grid_idx[:, 1], lower_grid_idx[:, 0], lower_heading_idx, ...]
    i10_lo = layer[lower_grid_idx[:, 1], upper_grid_idx[:, 0], lower_heading_idx, ...]
    i01_lo = layer[upper_grid_idx[:, 1], lower_grid_idx[:, 0], lower_heading_idx, ...]
    i11_lo = layer[upper_grid_idx[:, 1], upper_grid_idx[:, 0], lower_heading_idx, ...]

    # Sample at upper heading for all 4 spatial corners
    i00_hi = layer[lower_grid_idx[:, 1], lower_grid_idx[:, 0], upper_heading_idx, ...]
    i10_hi = layer[lower_grid_idx[:, 1], upper_grid_idx[:, 0], upper_heading_idx, ...]
    i01_hi = layer[upper_grid_idx[:, 1], lower_grid_idx[:, 0], upper_heading_idx, ...]
    i11_hi = layer[upper_grid_idx[:, 1], upper_grid_idx[:, 0], upper_heading_idx, ...]

    # Spatial bilinear weights
    num_extra_dims = layer.ndim - 3
    weight_shape = (-1,) + (1,) * num_extra_dims
    w0 = ((1 - alpha[:, 0]) * (1 - alpha[:, 1])).reshape(weight_shape)
    w1 = (alpha[:, 0] * (1 - alpha[:, 1])).reshape(weight_shape)
    w2 = ((1 - alpha[:, 0]) * alpha[:, 1]).reshape(weight_shape)
    w3 = (alpha[:, 0] * alpha[:, 1]).reshape(weight_shape)

    # Bilinear interpolate at each heading level
    spatial_lo = w0 * i00_lo + w1 * i10_lo + w2 * i01_lo + w3 * i11_lo
    spatial_hi = w0 * i00_hi + w1 * i10_hi + w2 * i01_hi + w3 * i11_hi

    # Interpolate between heading levels
    heading_weight_shape = (-1,) + (1,) * num_extra_dims
    h_alpha = heading_alpha.reshape(heading_weight_shape)
    return (1 - h_alpha) * spatial_lo + h_alpha * spatial_hi


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

    class TestQueryBilinearHeading(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            # Layer shape (3, 2, 8): H=3, W=2, 8 headings (0..7)
            # Each heading bin is pi/4 wide; heading 0 maps to index 0.
            cls.layer = jnp.arange(48).reshape(3, 2, 8).astype(jnp.float32)
            cls.gridmap = GridMap(
                origin=jnp.array([0.0, 0.0]),
                resolution=1.0,
                layers={"data": cls.layer},
            )

        def _query(self, x, y, heading):
            return float(
                query_bilinear_heading(
                    self.gridmap, "data", jnp.array([[x, y, heading]])
                )[0]
            )

        def test_exact_cell_and_heading(self):
            # Cell (row=0, col=0), heading index 0 -> value 0.0
            self.assertAlmostEqual(self._query(0.5, 0.5, 0.0), 0.0, places=5)
            # heading index 1 (pi/4) -> value 1.0
            self.assertAlmostEqual(self._query(0.5, 0.5, jnp.pi / 4), 1.0, places=5)
            # heading index 4 (pi) -> value 4.0
            self.assertAlmostEqual(self._query(0.5, 0.5, jnp.pi), 4.0, places=5)

        def test_heading_midpoint(self):
            # Halfway between index 0 and 1 -> (0+1)/2 = 0.5
            self.assertAlmostEqual(self._query(0.5, 0.5, jnp.pi / 8), 0.5, places=5)

        def test_circular_wrap(self):
            # Halfway between index 7 and 0 -> (7+0)/2 = 3.5
            self.assertAlmostEqual(
                self._query(0.5, 0.5, 2 * jnp.pi - jnp.pi / 8), 3.5, places=5
            )
            self.assertAlmostEqual(
                self._query(0.5, 0.5, -jnp.pi / 8), 3.5, places=5
            )

        def test_spatial_with_heading(self):
            # Between col 0 and col 1 at heading 0: (0 + 8) / 2 = 4.0
            self.assertAlmostEqual(self._query(1.0, 0.5, 0.0), 4.0, places=5)

        def test_extra_channel_dims(self):
            # Add 4-channel depth: (3, 2, 8, 4)
            layer = jnp.arange(192).reshape(3, 2, 8, 4).astype(jnp.float32)
            gm = GridMap(
                origin=jnp.array([0.0, 0.0]),
                resolution=1.0,
                layers={"data": layer},
            )
            pts = jnp.array([[0.5, 0.5, 0.0]])
            result = query_bilinear_heading(gm, "data", pts)
            self.assertEqual(result.shape, (1, 4))
            self.assertTrue(jnp.allclose(result[0], layer[0, 0, 0]))

    unittest.main()
