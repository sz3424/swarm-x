from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple
from config import Config
from models import Agent


class SafetyFilter:
    def __init__(self, config: Config, env):
        self.config = config
        self.env = env

    def validate_move(self, agent: Agent, target_x: float, target_y: float) -> bool:
        if not self.env.is_inside_grid(int(target_x), int(target_y)):
            return False

        if self.env.is_obstacle(int(target_x), int(target_y)):
            return False

        if self.env.obstacle_map[int(target_y), int(target_x)] > 0:
            return False

        dist = self.distance_to_obstacle(agent, target_x, target_y)
        if dist < self.config.obstacle_margin:
            return False

        if not self._agents_clear(agent, target_x, target_y):
            return False

        return True

    def distance_to_obstacle(self, agent: Agent, target_x: float, target_y: float) -> float:
        ox, oy = int(target_x), int(target_y)
        if self.env.is_inside_grid(ox, oy):
            if self.env.obstacle_map[oy, ox] > 0:
                return 0.0
        return 1.0

    def _agents_clear(self, agent: Agent, target_x: float, target_y: float) -> bool:
        for other_id, other_agent in self.env.agents.items():
            if other_id == agent.id:
                continue

            other_x, other_y = other_agent.position
            dist = np.sqrt((other_x - target_x) ** 2 + (other_y - target_y) ** 2)

            if dist < self.config.min_agent_distance:
                return False

        return True
