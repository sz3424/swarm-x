from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Optional
from config import Config
from models import Agent, Obstacle, Hazard, Task, TaskStatus, Observation


class FlowField:
    def __init__(self, env: "Environment", config: Config):
        self.env = env
        self.config = config
        self.version = 0
        self._build_initial_field()

    def _build_initial_field(self) -> None:
        # Initialize cost field
        self.cost = np.ones((self.env.height, self.env.width), dtype=np.float32) * self.config.base_cost
        self.obstacle_mask = self.env.obstacle_map.copy()
        self.hazard_map = self.env.hazard_map.copy()
        self.congestion = np.zeros((self.env.height, self.env.width), dtype=np.float32)
        self.visit_heat = np.zeros((self.env.height, self.env.width), dtype=np.float32)
        self.flow_dx = np.zeros((self.env.height, self.env.width), dtype=np.float32)
        self.flow_dy = np.zeros((self.env.height, self.env.width), dtype=np.float32)
        self.distance_to_goal = np.full((self.env.height, self.env.width), np.inf, dtype=np.float32)
        self._compute_goals()

    def _compute_goals(self) -> None:
        # Place tasks randomly in the environment
        num_tasks = 20
        for i in range(num_tasks):
            x = np.random.randint(0, self.env.width)
            y = np.random.randint(0, self.env.height)
            # Avoid placing near edges or obstacles
            if self.env.is_inside_grid(x, y) and self.env.obstacle_map[y, x] == 0:
                self.env.tasks[i] = Task(
                    id=i,
                    location=(x, y),
                    priority=np.random.randint(1, 10),
                    reward=1.0,
                    status=TaskStatus.UNASSIGNED
                )
            self._compute_distance_to_goal()

    def update(self) -> None:
        """Update the flow field based on current environment state"""
        # Update congestion based on agent occupancy
        for x in range(self.env.width):
            for y in range(self.env.height):
                if self.env.agent_occupancy[y, x] > 0:
                    self.congestion[y, x] += 0.1
        self.congestion *= self.config.congestion_decay

        # Update hazard costs
        for h in self.env.hazards:
            hx, hy = int(h.position[0]), int(h.position[1])
            if 0 <= hx < self.env.width and 0 <= hy < self.env.height:
                self.hazard_map[hy, hx] = h.cost

        # Compute distance to goal for each cell
        self._compute_distance_to_goal()

    def _compute_distance_to_goal(self) -> None:
        # Simple Euclidean distance to nearest task goal
        for y in range(self.env.height):
            for x in range(self.env.width):
                min_dist = np.inf
                for task in self.env.tasks.values():
                    if task.status == TaskStatus.UNASSIGNED or task.status == TaskStatus.IN_PROGRESS:
                        dist = np.sqrt((x - task.location[0]) ** 2 + (y - task.location[1]) ** 2)
                        if dist < min_dist:
                            min_dist = dist
                self.distance_to_goal[y, x] = min_dist if min_dist != np.inf else np.inf

    def get_flow_vector(self, x: float, y: float) -> Tuple[float, float]:
        """Get normalized flow vector at position (x, y)"""
        if not self.env.is_inside_grid(int(x), int(y)):
            return (0.0, 0.0)
        
        cx, cy = int(x), int(y)
        # Find valid neighbors
        neighbors = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.env.width and 0 <= ny < self.env.height:
                    if self.env.obstacle_map[ny, nx] == 0:
                        neighbors.append((nx, ny))
        
        if not neighbors:
            return (0.0, 0.0)
        
        # Weighted sum of neighbor distances
        total_dist = 0.0
        for nx, ny in neighbors:
            dist = np.sqrt((x - nx) ** 2 + (y - ny) ** 2)
            total_dist += dist
        
        # Normalize by number of neighbors
        avg_dist = total_dist / len(neighbors)
        
        # Direction toward closest neighbor(s)
        min_dist = min(np.sqrt((x - nx) ** 2 + (y - ny) ** 2) for nx, ny in neighbors)
        candidates = [(nx, ny) for nx, ny in neighbors if np.sqrt((x - nx) ** 2 + (y - ny) ** 2) == min_dist]
        
        # Pick first candidate (deterministic)
        chosen = candidates[0]
        
        # Calculate direction vector
        dx = chosen[0] - x
        dy = chosen[1] - y
        dist = np.sqrt(dx ** 2 + dy ** 2)
        
        if dist > 0:
            fx = dx / dist
            fy = dy / dist
        else:
            fx, fy = 0.0, 0.0
        
        return (fx, fy)

    def get_flow_field(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return cost, congestion, and flow vectors"""
        return self.cost, self.congestion, self.flow_dx, self.flow_dy
