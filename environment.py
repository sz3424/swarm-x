from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Set
from config import Config
from models import Agent, Obstacle, Hazard, Task


class Environment:
    def __init__(self, config: Config):
        self.config = config
        self.width = config.grid_width
        self.height = config.grid_height

        self.base_cost = np.ones((config.grid_height, config.grid_width), dtype=np.float32)
        self.obstacle_map = np.zeros((config.grid_height, config.grid_width), dtype=np.int32)
        self.hazard_map = np.zeros((config.grid_height, config.grid_width), dtype=np.float32)
        self.congestion = np.zeros((config.grid_height, config.grid_width), dtype=np.float32)
        self.visit_heat = np.zeros((config.grid_height, config.grid_width), dtype=np.float32)
        self.agent_occupancy = np.zeros((config.grid_height, config.grid_width), dtype=np.int32)

        self.flow_dx = np.zeros((config.grid_height, config.grid_width), dtype=np.float32)
        self.flow_dy = np.zeros((config.grid_height, config.grid_width), dtype=np.float32)
        self.distance_to_goal = np.full((config.grid_height, config.grid_width), np.inf, dtype=np.float32)

        self.dynamic_obstacles: List[Obstacle] = []
        self.hazards: List[Hazard] = []

        self.agents: Dict[int, Agent] = {}
        self.tasks: Dict[int, Task] = {}

        self.initialized = False

    def create_environment(self) -> None:
        if not self.initialized:
            self._initialize_grids()
            self._add_static_obstacles()
            self.initialized = True

    def _initialize_grids(self) -> None:
        self.base_cost = np.ones((self.height, self.width), dtype=np.float32)
        self.obstacle_map = np.zeros((self.height, self.width), dtype=np.int32)
        self.hazard_map = np.zeros((self.height, self.width), dtype=np.float32)
        self.congestion = np.zeros((self.height, self.width), dtype=np.float32)
        self.visit_heat = np.zeros((self.height, self.width), dtype=np.float32)
        self.agent_occupancy = np.zeros((self.height, self.width), dtype=np.int32)

    def _add_static_obstacles(self) -> None:
        obstacle_list = [
            ((10, 10), 1.5), ((90, 90), 1.5),
            ((10, 90), 1.5), ((90, 10), 1.5),
            ((50, 50), 3.0), ((30, 70), 2.0),
            ((70, 30), 2.0), ((20, 20), 1.0),
            ((80, 80), 1.0), ((70, 70), 1.0),
        ]
        for pos, radius in obstacle_list:
            self.add_obstacle(pos[0], pos[1], radius)

    def add_obstacle(self, x: float, y: float, radius: float) -> None:
        x_int, y_int = int(x), int(y)
        for dy in range(-int(radius), int(radius) + 1):
            for dx in range(-int(radius), int(radius) + 1):
                nx, ny = x_int + dx, y_int + dy
                if self.is_inside_grid(nx, ny):
                    dist = np.sqrt((dx) ** 2 + (dy) ** 2)
                    if dist <= radius:
                        self.obstacle_map[ny, nx] = 1
                        self.base_cost[ny, nx] = np.inf

    def remove_obstacle(self, x: float, y: float, radius: float) -> None:
        x_int, y_int = int(x), int(y)
        for dy in range(-int(radius), int(radius) + 1):
            for dx in range(-int(radius), int(radius) + 1):
                nx, ny = x_int + dx, y_int + dy
                if self.is_inside_grid(nx, ny):
                    if self.obstacle_map[ny, nx] > 0:
                        dist = np.sqrt((dx) ** 2 + (dy) ** 2)
                        if dist <= radius:
                            self.obstacle_map[ny, nx] = 0
                            self.base_cost[ny, nx] = self.config.base_cost

    def add_dynamic_obstacle(self, x: float, y: float, radius: float, velocity: Tuple[float, float]) -> None:
        obstacle_id = len(self.dynamic_obstacles) + 1
        obstacle = Obstacle(
            id=obstacle_id,
            position=(x, y),
            radius=radius,
            dynamic=True,
            velocity=velocity,
            active=True
        )
        self.dynamic_obstacles.append(obstacle)
        self.add_obstacle(x, y, radius)

    def update_dynamic_obstacles(self) -> None:
        for obstacle in self.dynamic_obstacles:
            if obstacle.active:
                x, y = obstacle.position
                obstacle.position = (
                    x + obstacle.velocity[0],
                    y + obstacle.velocity[1]
                )
                self.remove_obstacle(x, y, obstacle.radius)
                self.add_obstacle(obstacle.position[0], obstacle.position[1], obstacle.radius)

    def add_hazard(self, x: float, y: float, radius: float = 2.0) -> None:
        hazard_id = len(self.hazards) + 1
        hazard = Hazard(
            id=hazard_id,
            position=(x, y),
            radius=radius,
            cost=self.config.hazard_weight,
            active=True
        )
        self.hazards.append(hazard)
        x_int, y_int = int(x), int(y)
        for dy in range(-int(radius), int(radius) + 1):
            for dx in range(-int(radius), int(radius) + 1):
                nx, ny = x_int + dx, y_int + dy
                if self.is_inside_grid(nx, ny):
                    dist = np.sqrt((dx) ** 2 + (dy) ** 2)
                    if dist <= radius:
                        self.hazard_map[ny, nx] = hazard.cost

    def update_congestion(self, decay: bool = True) -> None:
        if decay:
            self.congestion *= self.config.congestion_decay

    def add_agent(self, agent: Agent) -> None:
        self.agents[agent.id] = agent
        x, y = agent.position
        x_int, y_int = int(x), int(y)
        if self.is_inside_grid(x_int, y_int):
            self.agent_occupancy[y_int, x_int] += 1

    def remove_agent(self, agent_id: int) -> None:
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            x, y = agent.position
            x_int, y_int = int(x), int(y)
            if self.is_inside_grid(x_int, y_int):
                self.agent_occupancy[y_int, x_int] = max(0, self.agent_occupancy[y_int, x_int] - 1)
            del self.agents[agent_id]

    def get_local_observation(self, x: int, y: int, sensing_radius: int) -> Dict:
        x_min = max(0, x - sensing_radius)
        x_max = min(self.width - 1, x + sensing_radius)
        y_min = max(0, y - sensing_radius)
        y_max = min(self.height - 1, y + sensing_radius)

        observation = {
            'agent_id': -1,
            'position': (x, y),
            'sensing_radius': sensing_radius,
            'nearby_obstacles': [],
            'nearby_agents': [],
            'local_congestion': {},
            'local_hazards': [],
            'local_flow': {},
            'nearby_tasks': [],
            'communication_state': True,
            'grid_bounds': (x_min, x_max, y_min, y_max)
        }

        for ny in range(y_min, y_max + 1):
            for nx in range(x_min, x_max + 1):
                if self.obstacle_map[ny, nx] > 0:
                    obs_x = nx + 0.5
                    obs_y = ny + 0.5
                    observation['nearby_obstacles'].append(
                        Obstacle(id=0, position=(obs_x, obs_y), radius=0.5)
                    )

                if self.hazard_map[ny, nx] > 0:
                    haz_x = nx + 0.5
                    haz_y = ny + 0.5
                    observation['local_hazards'].append(
                        Hazard(id=0, position=(haz_x, haz_y), radius=0.5, cost=self.hazard_map[ny, nx])
                    )

                observation['local_congestion'][(nx, ny)] = self.congestion[ny, nx]
                observation['local_flow'][(nx, ny)] = (self.flow_dx[ny, nx], self.flow_dy[ny, nx])

        for agent_id, agent in self.agents.items():
            ax, ay = agent.position
            dist = np.sqrt((ax - x) ** 2 + (ay - y) ** 2)
            if dist <= sensing_radius:
                observation['nearby_agents'].append(agent_id)

        return observation

    def is_walkable(self, x: float, y: float) -> bool:
        x_int, y_int = int(x), int(y)
        return self.is_inside_grid(x_int, y_int) and self.base_cost[y_int, x_int] < np.inf

    def is_inside_grid(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def is_obstacle(self, x: int, y: int) -> bool:
        return self.is_inside_grid(x, y) and self.obstacle_map[y, x] > 0

    def environment_changed(self) -> bool:
        for obstacle in self.dynamic_obstacles:
            if obstacle.active:
                return True
        return False

    def add_agent_occupancy(self, x: int, y: int) -> None:
        if self.is_inside_grid(x, y):
            self.agent_occupancy[y, x] += 1

    def remove_agent_occupancy(self, x: int, y: int) -> None:
        if self.is_inside_grid(x, y):
            self.agent_occupancy[y, x] = max(0, self.agent_occupancy[y, x] - 1)
