from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

import gridmap


@jax.tree_util.register_dataclass
@dataclass
class DynamicsInfo:
    """Dynamics information for cost computation."""

    dt: jnp.ndarray  # time step [s]
    pose2d: jnp.ndarray  # (x, y, theta)
    # cost map query (layer_name, query_pts, query_pts_vels, query_pts_scale)
    cost_queries: tuple[tuple[str, jnp.ndarray, jnp.ndarray, jnp.ndarray]]
    # collision map query (layer_name, query_pts, query_pts_vels)
    collision_queries: tuple[tuple[str, jnp.ndarray, jnp.ndarray]]
    # velocity limit query (layer_name, query_pts, query_pts_vels)
    vel_limits: tuple[tuple[str, jnp.ndarray, jnp.ndarray]]


"""
Dynamic Horizon Terminal Cost

Costs components:
- cost per step t: c(x_t)
- value function: V(x_k)

Costs are summed up to step k, at which point the value funciton is queried.
    C(x, k) = sum_{t=0}^k c(x_t) + V(x_k)

The time horizon k is optimized:
    C(x) = min_k C(x, k) - long_horizon_bias * k

Additionally, constraints are ususally enforced for the full horizon.
- constraint step cost: s(x_t)
- constraint terminal cost: S(x_T)

 C_full = C(x) + sum_{t=0}^T s(x_t) + S(x_T)
"""


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class DynamicHorizonTerminalCostState:
    i: int
    best_Ck: jnp.ndarray  # sum_c_k + V(x_k) - long_horizon_bias * k
    sum_c: jnp.ndarray

    def create(self):
        return DynamicHorizonTerminalCostState(
            i=0, best_Ck=jnp.array(jnp.inf), sum_c=jnp.zeros(())
        )


def dynamic_horizon_terminal_cost_step_cost(
    cost_state: DynamicHorizonTerminalCostState,
    cost_i: jnp.ndarray,
    value_i: jnp.ndarray,
    long_horizon_bias: jnp.ndarray,
) -> tuple[DynamicHorizonTerminalCostState, jnp.ndarray]:
    sum_c = cost_state.sum_c + cost_i
    Ck = sum_c + value_i - long_horizon_bias * cost_state.i
    best_Ck = jnp.minimum(cost_state.best_Ck, Ck)
    return DynamicHorizonTerminalCostState(
        i=cost_state.i + 1, best_Ck=best_Ck, sum_c=sum_c
    ), jnp.zeros(())


def dynamic_horizon_terminal_cost_terminal_cost(
    cost_state: DynamicHorizonTerminalCostState,
) -> jnp.ndarray:
    return cost_state.best_Ck


@jax.tree_util.register_dataclass
@dataclass
class GridMapCostContext:
    cost_to_go_map: gridmap.GridMap
    traversability_map: gridmap.GridMap
    # cost parameters
    long_horizon_bias: jnp.ndarray
    collision_constraint_cost: jnp.ndarray
    velocity_constraint_cost: jnp.ndarray


@jax.tree_util.register_dataclass
@dataclass
class GridMapCostState:
    dyn_horizon: DynamicHorizonTerminalCostState


def gridmap_cost(
    cost_state: GridMapCostState,
    dyn_state: Any,
    action: jax.Array,
    goal_context: GridMapCostContext,
):
    dyn = dyn_state.get_dynamics_info()
    step_cost = jnp.zeros(())
    for layer_name, query_pts, query_pts_vels, scale in dyn.cost_queries:
        costs = gridmap.query_bilinear(
            goal_context.traversability_map, layer_name, query_pts
        )
        step_cost += jnp.sum(costs * scale * query_pts_vels * dyn.dt)

    value = gridmap.query_bilinear_heading(
        goal_context.cost_to_go_map, "cost_to_go", dyn.pose2d
    )
    cost_state_dyn_horizon, step_cost = dynamic_horizon_terminal_cost_step_cost(
        cost_state.dyn_horizon,
        step_cost,
        value,
        goal_context.long_horizon_bias * dyn.dt,
    )
    next_cost_state = GridMapCostState(dyn_horizon=cost_state_dyn_horizon)

    constraint_cost = 0
    for layer_name, query_pts, query_pts_vels in dyn.collision_queries:
        collisions = gridmap.query_bilinear(
            goal_context.traversability_map, layer_name, query_pts
        )
        constraint_cost += (
            jnp.sum(collisions * query_pts_vels * dyn.dt * (1 + query_pts_vels))
            * goal_context.collision_constraint_cost
        )
    for layer_name, query_pts, query_pts_vels in dyn.vel_limits:
        vel_limits = gridmap.query_bilinear(
            goal_context.traversability_map, layer_name, query_pts
        )
        vel_exceeded = jnp.maximum(query_pts_vels - vel_limits, 0)
        constraint_cost += (
            jnp.sum(vel_exceeded * query_pts_vels * dyn.dt)
            * goal_context.velocity_constraint_cost
        )
    return next_cost_state, step_cost + constraint_cost


def gridmap_terminal_cost(cost_state, dyn_state, goal_context):
    return dynamic_horizon_terminal_cost_terminal_cost(cost_state.dyn_horizon)
