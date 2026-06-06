
#include <algorithm>
#include <functional>
#include <limits>
#include <queue>
#include <stdbool.h>
#include <stdio.h>
#include <vector>

/**
 * @brief Solves the shortest path problem on a lattice graph using Dijkstra's
 * algorithm.
 *
 * @param edge_costs Array of shape (num_states, num_edges) containing the cost
 *                   for each transition from a given state to its neighbors.
 * @param next_states Array of shape (num_states, num_edges) containing the flat
 *                    indices of the neighboring states for each state.
 * @param dist_out Output array of shape (num_states) that will be populated
 * with the minimum cost to reach each state.
 * @param num_states Total number of states in the lattice graph.
 * @param num_edges Number of outgoing edges (neighbors) per state.
 * @param start_states Array of shape (num_start_states) containing the indices
 *                     of the initial states.
 * @param start_values Array of shape (num_start_states) containing the values
 *                     of the initial states.
 * @param num_start_states Number of initial states.
 * @param terminal_state The index of the goal/terminal state.
 *
 * @return true if the terminal state was successfully reached, false otherwise.
 */
extern "C" bool solve(float *edge_costs, int *next_states, float *dist_out,
                      int num_states, int num_edges, int *start_states,
                      float *start_values, int num_start_states,
                      int terminal_state) {

  // Initialize distances to infinity
  std::fill(dist_out, dist_out + num_states,
            std::numeric_limits<float>::infinity());

  // Min-heap priority queue storing pairs of (distance, state_index)
  std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>,
                      std::greater<std::pair<float, int>>>
      pq;

  // Initialize queue with start states using the provided start_values
  for (int i = 0; i < num_start_states; ++i) {
    int s = start_states[i];
    if (s >= 0 && s < num_states) {
      float sv = start_values[i];
      dist_out[s] = sv;
      pq.push({sv, s});
    }
  }

  bool found_terminal = false;

  while (!pq.empty()) {
    float d = pq.top().first;
    int u = pq.top().second;
    pq.pop();

    // Lazy deletion: skip if we've already found a shorter path to this state
    if (d > dist_out[u])
      continue;

    // Early exit if we've settled the terminal state
    if (u == terminal_state) {
      found_terminal = true;
      break;
    }

    // Explore all outgoing edges (row-major layout)
    for (int e = 0; e < num_edges; ++e) {
      int v = next_states[u * num_edges + e];
      float cost = edge_costs[u * num_edges + e];

      // Bounds check to safely handle clipped boundary indices from JAX
      if (v >= 0 && v < num_states) {
        float new_dist = d + cost;
        if (new_dist < dist_out[v]) {
          dist_out[v] = new_dist;
          pq.push({new_dist, v});
        }
      }
    }
  }

  return found_terminal;
}
