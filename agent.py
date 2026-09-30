from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple
from config import Config
from models import (
    Agent, Obstacle, Hazard, Task, TaskStatus,
    Observation, MessageType, Message, AgentStatus
)
from environment import Environment
from flow_field import FlowField
from safety import SafetyFilter


class AgentManager:
    def __init__(self, config: Config, env: Environment, flow_field: FlowField):
        self.config = config
        self.env = env
        self.flow_field = flow_field
        self.agents: Dict[int, Agent] = {}
        self._next_id = 0

    def add_agent(self) -> Agent:
        agent_id = self._next_id
        agent = Agent(
            id=agent_id,
            position=(np.random.uniform(10, 90), np.random.uniform(10, 90)),
            goal=(np.random.randint(0, self.env.width), np.random.randint(0, self.env.height)),
            status=AgentStatus.IDLE,
            communication_available=True
        )
        self.agents[agent_id] = agent
        self.env.add_agent(agent)
        self._next_id += 1
        return agent

    def initialize_agents(self, count: int) -> None:
        for _ in range(count):
            self.add_agent()

    def get_agent(self, agent_id: int) -> Optional[Agent]:
        return self.agents.get(agent_id)

    def update_agent_position(self, agent_id: int, new_x: float, new_y: float) -> bool:
        agent = self.get_agent(agent_id)
        if not agent:
            return False

        old_x, old_y = agent.position
        if not self.env.is_walkable(new_x, new_y):
            return False

        safety = SafetyFilter(self.config, self.env)
        if not safety.validate_move(agent, new_x, new_y):
            return False

        self.env.remove_agent_occupancy(int(old_x), int(old_y))
        self.env.add_agent_occupancy(int(new_x), int(new_y))

        agent.position = (new_x, new_y)
        agent.velocity = (
            new_x - old_x,
            new_y - old_y
        )

        if agent.current_task_id is not None:
            task = self.env.tasks.get(agent.current_task_id)
            if task:
                tx, ty = task.location
                dist = np.sqrt((new_x - tx) ** 2 + (new_y - ty) ** 2)
                agent.total_distance += dist

        return True

    def move_agent(self, agent_id: int, dx: float, dy: float) -> bool:
        agent = self.get_agent(agent_id)
        if not agent:
            return False

        new_x = agent.position[0] + dx
        new_y = agent.position[1] + dy
        return self.update_agent_position(agent_id, new_x, new_y)

    def observe(self, agent_id: int) -> Observation:
        agent = self.get_agent(agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")

        x, y = agent.position
        observation = self.env.get_local_observation(int(x), int(y), self.config.sensing_radius)
        observation['agent_id'] = agent_id
        agent.local_observation = observation
        return observation

    def decide_movement(self, agent_id: int) -> Dict:
        agent = self.get_agent(agent_id)
        if not agent:
            return {'valid': False, 'reason': 'Agent not found'}

        obs = self.observe(agent_id)
        flow_dx, flow_dy = self.flow_field.get_flow_vector(agent.position[0], agent.position[1])

        decision = {
            'agent_id': agent_id,
            'position': agent.position,
            'goal': agent.goal,
            'current_task_id': agent.current_task_id,
            'battery': agent.battery,
            'status': agent.status.value,
            'neighbors': obs['nearby_agents'],
            'communication_state': obs['communication_state'],
            'current_flow': (flow_dx, flow_dy),
            'chosen_action': (0.0, 0.0),
            'safety_result': False,
            'decision_latency': 0,
            'stalled_ticks': agent.stalled_ticks,
            'reason': ''
        }

        if agent.status == AgentStatus.IDLE or agent.status == AgentStatus.MOVING:
            dx, dy = self._select_movement(agent, obs, flow_dx, flow_dy)
            decision['chosen_action'] = (dx, dy)

            safety = SafetyFilter(self.config, self.env)
            decision['safety_result'] = safety.validate_move(agent, agent.position[0] + dx, agent.position[1] + dy)

            if decision['safety_result']:
                self.move_agent(agent_id, dx, dy)
                self.env.add_agent_occupancy(int(agent.position[0]), int(agent.position[1]))

                if agent.stalled_ticks > 0:
                    agent.stalled_ticks -= 1

        decision['reason'] = self._generate_reason(agent, obs, decision)
        agent.last_decision = decision

        return decision

    def _select_movement(self, agent: Agent, obs: Observation, flow_dx: float, flow_dy: float) -> Tuple[float, float]:
        speed = self.config.agent_speed

        candidates = []
        for dx in (-speed, 0, speed):
            for dy in (-speed, 0, speed):
                if dx == 0 and dy == 0:
                    continue

                nx, ny = agent.position[0] + dx, agent.position[1] + dy

                if not self.env.is_walkable(nx, ny):
                    continue

                dist = np.sqrt(dx ** 2 + dy ** 2)
                candidates.append((dist, dx, dy, nx, ny))

        if not candidates:
            return (0.0, 0.0)

        candidates.sort()

        for dist, dx, dy, nx, ny in candidates:
            if self.env.is_walkable(nx, ny):
                return (dx, dy)

        return (0.0, 0.0)

    def _generate_reason(self, agent: Agent, obs: Observation, decision: Dict) -> str:
        dx, dy = decision['chosen_action']
        if dx == 0 and dy == 0:
            return "No valid movement found"

        direction = ""
        if abs(dx) > abs(dy):
            direction = "east" if dx > 0 else "west"
        else:
            direction = "north" if dy > 0 else "south"

        return f"Selected {direction}ward movement because it was the valid direction with minimum cost and passed safety checks."

    def update_agents(self) -> None:
        for agent_id in list(self.agents.keys()):
            agent = self.agents[agent_id]
            if agent.status == AgentStatus.MOVING:
                obs = self.observe(agent_id)
                stalled = True

                for neighbor_id in obs['nearby_agents']:
                    neighbor = self.agents.get(neighbor_id)
                    if neighbor and neighbor.status == AgentStatus.MOVING:
                        dist = np.sqrt(
                            (agent.position[0] - neighbor.position[0]) ** 2 +
                            (agent.position[1] - neighbor.position[1]) ** 2
                        )
                        if dist > self.config.min_agent_distance:
                            stalled = False

                if stalled:
                    agent.stalled_ticks += 1
                else:
                    agent.stalled_ticks = 0

    def get_active_agents(self) -> List[Agent]:
        return [a for a in self.agents.values() if a.status not in [AgentStatus.FAILED, AgentStatus.COMPLETED]]
