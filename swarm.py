from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Optional
from config import Config
from models import Agent, Task, TaskStatus, AgentStatus, MessageType
from environment import Environment
from flow_field import FlowField
from agent import AgentManager
from safety import SafetyFilter
from communication import CommunicationManager


class Swarm:
    def __init__(self, config: Config, env: Environment):
        self.config = config
        self.env = env
        self.flow_field = FlowField(env, config)
        self.agent_manager = AgentManager(config, env, self.flow_field)
        self.safety = SafetyFilter(config, env)
        self.communication = CommunicationManager(config, env)
        self.tick = 0
        self.current_scenario = None
        self.running = False
        self.paused = False

    def initialize(self, num_agents: int = 20) -> None:
        self.agent_manager.initialize_agents(num_agents)

    def _assign_initial_tasks(self) -> None:
        for agent in self.agent_manager.agents.values():
            available_tasks = [t for t in self.env.tasks.values() if t.status == TaskStatus.UNASSIGNED]
            if available_tasks:
                task = min(available_tasks, key=lambda t: np.sqrt(
                    (agent.position[0] - t.location[0]) ** 2 + 
                    (agent.position[1] - t.location[1]) ** 2
                ))
                task.status = TaskStatus.ASSIGNED
                task.assigned_agent_id = agent.id
                agent.current_task_id = task.id
                agent.goal = task.location
                agent.status = AgentStatus.MOVING

    def observe(self) -> Dict[int, dict]:
        observations = {}
        for agent_id in self.agent_manager.agents:
            observations[agent_id] = self.agent_manager.observe(agent_id)
        return observations

    def allocate_tasks(self) -> None:
        available_tasks = [t for t in self.env.tasks.values() if t.status == TaskStatus.UNASSIGNED]
        idle_agents = [a for a in self.agent_manager.agents.values() if a.status == AgentStatus.IDLE]

        for task in available_tasks:
            best_agent = None
            best_bid = -1

            for agent in idle_agents:
                bid = self._calculate_bid(agent, task)
                if bid > best_bid:
                    best_bid = bid
                    best_agent = agent

            if best_agent:
                task.status = TaskStatus.ASSIGNED
                task.assigned_agent_id = best_agent.id
                best_agent.current_task_id = task.id
                best_agent.goal = task.location
                best_agent.status = AgentStatus.MOVING

    def _calculate_bid(self, agent: Agent, task: Task) -> float:
        dist = np.sqrt(
            (agent.position[0] - task.location[0]) ** 2 + 
            (agent.position[1] - task.location[1]) ** 2
        )
        energy_cost = dist * self.config.movement_cost
        
        if agent.battery < energy_cost:
            return -1

        congestion = self.env.congestion[task.location[1], task.location[0]]
        bid = task.reward / (dist + 1) - congestion * self.config.congestion_weight - energy_cost * self.config.energy_weight
        
        return bid

    def plan_movements(self) -> Dict[int, Tuple[float, float]]:
        movements = {}
        for agent_id in self.agent_manager.agents:
            decision = self.agent_manager.decide_movement(agent_id)
            if decision.get('safety_result', False):
                movements[agent_id] = decision['chosen_action']
        return movements

    def resolve_conflicts(self, movements: Dict[int, Tuple[float, float]]) -> Dict[int, Tuple[float, float]]:
        target_cells: Dict[Tuple[int, int], List[int]] = {}

        for agent_id, (dx, dy) in movements.items():
            agent = self.agent_manager.get_agent(agent_id)
            if not agent:
                continue

            new_x = agent.position[0] + dx
            new_y = agent.position[1] + dy
            target = (int(new_x), int(new_y))

            if target not in target_cells:
                target_cells[target] = []
            target_cells[target].append(agent_id)

        resolved = {}
        for target, agents in target_cells.items():
            if len(agents) == 1:
                resolved[agents[0]] = movements[agents[0]]
            else:
                winner = self._resolve_conflict(agents)
                resolved[winner] = movements[winner]
                for agent_id in agents:
                    if agent_id != winner:
                        agent = self.agent_manager.get_agent(agent_id)
                        if agent:
                            agent.status = AgentStatus.WAITING

        return resolved

    def _resolve_conflict(self, agents: List[int]) -> int:
        best_agent = agents[0]
        best_priority = -1
        for agent_id in agents:
            agent = self.agent_manager.get_agent(agent_id)
            if agent and agent.current_task_id:
                task = self.env.tasks.get(agent.current_task_id)
                if task and task.priority > best_priority:
                    best_priority = task.priority
                    best_agent = agent_id
                elif task and task.priority == best_priority and agent_id < best_agent:
                    best_agent = agent_id
        return best_agent

    def execute_movements(self, movements: Dict[int, Tuple[float, float]]) -> None:
        for agent_id, (dx, dy) in movements.items():
            self.agent_manager.move_agent(agent_id, dx, dy)

    def step(self) -> Dict:
        self.tick += 1

        self.env.update_dynamic_obstacles()
        self.flow_field.update()

        self.allocate_tasks()
        movements = self.plan_movements()
        movements = self.resolve_conflicts(movements)
        self.execute_movements(movements)

        self.env.update_congestion()
        self.agent_manager.update_agents()
        self.communication.update()

        return {
            'tick': self.tick,
            'active_agents': len(self.agent_manager.get_active_agents()),
            'completed_tasks': sum(1 for t in self.env.tasks.values() if t.status == TaskStatus.COMPLETED)
        }

    def is_complete(self) -> bool:
        return all(t.status == TaskStatus.COMPLETED for t in self.env.tasks.values())
