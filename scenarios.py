from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple
from models import Config, Task, TaskStatus, Agent, AgentStatus


class Scenario:
    def __init__(self, name: str, config: Config):
        self.name = name
        self.config = config
        self.agents_count: int = 20
        self.dynamic_obstacles: bool = False
        self.communication_loss: float = 0.0
        self.agent_failures: bool = False
        self.high_congestion: bool = False
        self.perturbations: List[Tuple[int, str]] = []

    def apply(self, swarm) -> None:
        if self.dynamic_obstacles:
            self._add_dynamic_obstacles(swarm)
        if self.agent_failures:
            self._inject_failures(swarm)
        if self.high_congestion:
            self._increase_congestion(swarm)
        self._apply_perturbations(swarm)

    def _add_dynamic_obstacles(self, swarm) -> None:
        for i in range(5):
            x = np.random.uniform(10, 90)
            y = np.random.uniform(10, 90)
            vx = np.random.uniform(-0.5, 0.5)
            vy = np.random.uniform(-0.5, 0.5)
            swarm.env.add_dynamic_obstacle(x, y, 1.5, (vx, vy))

    def _inject_failures(self, swarm) -> None:
        agents = list(swarm.agent_manager.agents.values())
        if len(agents) > 5:
            for agent in agents[:5]:
                agent.failed = True
                agent.status = AgentStatus.FAILED
                if agent.current_task_id is not None:
                    task = swarm.env.tasks.get(agent.current_task_id)
                    if task:
                        task.status = TaskStatus.UNASSIGNED
                        task.assigned_agent_id = None
                        agent.current_task_id = None

    def _increase_congestion(self, swarm) -> None:
        swarm.env.congestion *= 2.0

    def _apply_perturbations(self, swarm) -> None:
        for tick, perturbation in self.perturbations:
            if swarm.tick == tick:
                if perturbation == "ADD OBSTACLE":
                    x = np.random.uniform(10, 90)
                    y = np.random.uniform(10, 90)
                    swarm.env.add_obstacle(x, y, 2.0)
                elif perturbation == "FAIL AGENT":
                    agents = list(swarm.agent_manager.agents.values())
                    active = [a for a in agents if not a.failed]
                    if active:
                        agent = active[0]
                        agent.failed = True
                        agent.status = AgentStatus.FAILED
                        if agent.current_task_id is not None:
                            task = swarm.env.tasks.get(agent.current_task_id)
                            if task:
                                task.status = TaskStatus.UNASSIGNED
                                task.assigned_agent_id = None
                                agent.current_task_id = None


class Round1Scenario(Scenario):
    def __init__(self, config: Config):
        super().__init__("round1", config)
        self.agents_count = 25
        self.dynamic_obstacles = False
        self.communication_loss = 0.0
        self.agent_failures = False
        self.high_congestion = False


class Round2Scenario(Scenario):
    def __init__(self, config: Config):
        super().__init__("round2", config)
        self.agents_count = 30
        self.dynamic_obstacles = True
        self.communication_loss = 0.2
        self.agent_failures = False
        self.high_congestion = False


class FinalScenario(Scenario):
    def __init__(self, config: Config):
        super().__init__("final", config)
        self.agents_count = 50
        self.dynamic_obstacles = True
        self.communication_loss = 0.25
        self.agent_failures = True
        self.high_congestion = True
        self.perturbations = [
            (25, "ADD OBSTACLE"),
            (50, "FAIL AGENT"),
            (75, "FAIL AGENT"),
            (100, "ADD OBSTACLE"),
        ]


def create_scenario(name: str, config: Config) -> Scenario:
    scenarios = {
        "round1": Round1Scenario,
        "round2": Round2Scenario,
        "final": FinalScenario,
    }
    if name not in scenarios:
        raise ValueError(f"Unknown scenario: {name}")
    return scenarios[name](config)