from __future__ import annotations

import heapq
from typing import TYPE_CHECKING, Tuple

import numpy as np

from config import Config
from models import Agent, Hazard, Obstacle, Observation, Task, TaskStatus

if TYPE_CHECKING:
    from environment import Environment


class FlowField:
    """Adaptive cost-to-go flow field for decentralized swarm navigation.

    The field combines:
    - obstacle hard constraints,
    - hazard costs,
    - congestion/stigmergic costs,
    - shortest-path distance to active task goals,
    - deterministic local gradient following.
    """

    def __init__(self, env: "Environment", config: Config):
        self.env = env
        self.config = config
        self.version = 0
        self._build_initial_field()

    def _build_initial_field(self) -> None:
        shape = (self.env.height, self.env.width)

        self.cost = np.full(
            shape,
            self.config.base_cost,
            dtype=np.float32,
        )
        self.obstacle_mask = self.env.obstacle_map.copy()
        self.hazard_map = self.env.hazard_map.copy()
        self.congestion = np.zeros(shape, dtype=np.float32)
        self.visit_heat = np.zeros(shape, dtype=np.float32)
        self.flow_dx = np.zeros(shape, dtype=np.float32)
        self.flow_dy = np.zeros(shape, dtype=np.float32)
        self.distance_to_goal = np.full(
            shape,
            np.inf,
            dtype=np.float32,
        )

        self._compute_goals()
        self._update_cost_field()
        self._compute_distance_to_goal()
        self._compute_flow_vectors()

    def _compute_goals(self) -> None:
        """Create deterministic initial task goals when none exist."""
        if self.env.tasks:
            return

        num_tasks = min(20, max(1, self.env.width * self.env.height // 100))

        candidates = [
            (x, y)
            for y in range(self.env.height)
            for x in range(self.env.width)
            if self.env.obstacle_map[y, x] == 0
        ]

        if not candidates:
            return

        rng = np.random.default_rng(getattr(self.config, "random_seed", 42))
        rng.shuffle(candidates)

        for task_id, location in enumerate(candidates[:num_tasks]):
            self.env.tasks[task_id] = Task(
                id=task_id,
                location=location,
                priority=int(rng.integers(1, 10)),
                reward=1.0,
                status=TaskStatus.UNASSIGNED,
            )

    def _update_cost_field(self) -> None:
        """Build the adaptive movement-cost field."""
        self.obstacle_mask = self.env.obstacle_map.copy()
        self.hazard_map = self.env.hazard_map.copy()

        self.cost.fill(self.config.base_cost)

        # Stigmergic congestion penalty.
        self.cost += self.congestion

        # Historical visit heat provides a small adaptive avoidance signal.
        self.cost += 0.1 * self.visit_heat

        # Hazards increase traversal cost.
        self.cost += self.hazard_map

        # Obstacles are hard barriers.
        self.cost[self.obstacle_mask > 0] = np.inf

    def update(self) -> None:
        """Update the adaptive field with periodic global replanning."""
        occupancy = self.env.agent_occupancy

        occupied = occupancy > 0
        self.congestion[occupied] += float(
            self.config.congestion_deposit
        )

        decay = float(self.config.congestion_decay)
        self.congestion *= decay

        self.visit_heat *= decay
        self.visit_heat[occupied] += 0.05

        self.hazard_map.fill(0.0)

        for hazard in self.env.hazards:
            hx = int(hazard.position[0])
            hy = int(hazard.position[1])

            if 0 <= hx < self.env.width and 0 <= hy < self.env.height:
                self.hazard_map[hy, hx] = float(hazard.cost)

        self._update_cost_field()

        interval = max(
            1,
            int(getattr(self.config, "flow_recompute_interval", 5)),
        )

        should_recompute = (
            self.version == 0
            or self.version % interval == 0
        )

        if should_recompute:
            self._compute_distance_to_goal()
            self._compute_flow_vectors()

        self.version += 1

    def _active_goals(self) -> list[Tuple[int, int]]:
        """Return currently actionable task locations."""
        goals: list[Tuple[int, int]] = []

        for task in self.env.tasks.values():
            if task.status in (
                TaskStatus.UNASSIGNED,
                TaskStatus.IN_PROGRESS,
            ):
                goals.append(
                    (
                        int(task.location[0]),
                        int(task.location[1]),
                    )
                )

        return goals

    def _neighbors(self, x: int, y: int):
        """Yield valid 8-connected neighboring cells."""
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue

                nx = x + dx
                ny = y + dy

                if not (0 <= nx < self.env.width):
                    continue
                if not (0 <= ny < self.env.height):
                    continue
                if self.obstacle_mask[ny, nx] > 0:
                    continue

                yield nx, ny

    def _compute_distance_to_goal(self) -> None:
        """Compute weighted shortest-path distance to the nearest goal.

        This is a multi-source Dijkstra search. Each task goal is inserted
        into the priority queue with zero cost, allowing all agents to use
        the same distributed-style cost-to-go field.
        """
        self.distance_to_goal.fill(np.inf)

        goals = self._active_goals()
        if not goals:
            return

        queue: list[tuple[float, int, int]] = []

        for gx, gy in goals:
            if not (0 <= gx < self.env.width and 0 <= gy < self.env.height):
                continue

            if self.obstacle_mask[gy, gx] > 0:
                continue

            self.distance_to_goal[gy, gx] = 0.0
            heapq.heappush(queue, (0.0, gx, gy))

        while queue:
            current_distance, x, y = heapq.heappop(queue)

            if current_distance > float(self.distance_to_goal[y, x]) + 1e-6:
                 continue

            for nx, ny in self._neighbors(x, y):
                diagonal = nx != x and ny != y
                movement_cost = 1.41421356 if diagonal else 1.0

                cell_cost = float(self.cost[ny, nx])
                if not np.isfinite(cell_cost):
                    continue

                new_distance = (
                    current_distance
                    + movement_cost
                    + cell_cost
                )

                if new_distance < float(self.distance_to_goal[ny, nx]):
                    self.distance_to_goal[ny, nx] = new_distance
                    heapq.heappush(
                        queue,
                        (new_distance, nx, ny),
                    )

    def _compute_flow_vectors(self) -> None:
        """Generate normalized flow directions toward lower cost-to-go."""
        self.flow_dx.fill(0.0)
        self.flow_dy.fill(0.0)

        for y in range(self.env.height):
            for x in range(self.env.width):
                if self.obstacle_mask[y, x] > 0:
                    continue

                current = float(self.distance_to_goal[y, x])

                if not np.isfinite(current):
                    continue

                best = None

                for nx, ny in self._neighbors(x, y):
                    neighbor_distance = float(
                        self.distance_to_goal[ny, nx]
                    )

                    if not np.isfinite(neighbor_distance):
                        continue

                    candidate = (
                        neighbor_distance,
                        ny,
                        nx,
                    )

                    if neighbor_distance < current and (
                        best is None or candidate < best
                    ):
                        best = candidate

                if best is None:
                    continue

                _, ny, nx = best

                dx = float(nx - x)
                dy = float(ny - y)
                magnitude = float(np.hypot(dx, dy))

                if magnitude > 0.0:
                    self.flow_dx[y, x] = dx / magnitude
                    self.flow_dy[y, x] = dy / magnitude

    def get_flow_vector(
        self,
        x: float,
        y: float,
    ) -> Tuple[float, float]:
        """Return the normalized local flow direction."""
        cx = int(x)
        cy = int(y)

        if not self.env.is_inside_grid(cx, cy):
            return 0.0, 0.0

        if self.obstacle_mask[cy, cx] > 0:
            return 0.0, 0.0

        return (
            float(self.flow_dx[cy, cx]),
            float(self.flow_dy[cy, cx]),
        )

    def get_flow_field(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return cost, congestion, and directional flow fields."""
        return (
            self.cost,
            self.congestion,
            self.flow_dx,
            self.flow_dy,
        )