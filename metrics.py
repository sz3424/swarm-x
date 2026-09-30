from __future__ import annotations

import json
import time
from typing import Dict, List, Any
from dataclasses import dataclass, field, asdict
from models import Config, Agent, Task, TaskStatus, AgentStatus


@dataclass
class Metrics:
    total_tasks: int = 0
    completed_tasks: int = 0
    task_completion_rate: float = 0.0
    throughput: float = 0.0
    coverage: float = 0.0
    total_distance: float = 0.0
    average_path_length: float = 0.0
    energy_consumed: float = 0.0
    collision_count: int = 0
    minimum_clearance: float = float('inf')
    deadlock_count: int = 0
    mean_tick_latency_ms: float = 0.0
    p95_tick_latency_ms: float = 0.0
    p99_tick_latency_ms: float = 0.0
    max_tick_latency_ms: float = 0.0
    deadline_violations: int = 0
    communication_loss_rate: float = 0.0
    failed_agents: int = 0
    recovered_tasks: int = 0
    recovery_time: float = 0.0
    fitness: float = 0.0
    safety_status: str = "SAFE"


class MetricsCollector:
    def __init__(self, config: Config):
        self.config = config
        self.metrics = Metrics()
        self.tick_latencies: List[float] = []
        self.tick_start_time: float = 0.0
        self.tick_count = 0
        self.start_time = time.perf_counter()

    def start_tick(self) -> None:
        self.tick_start_time = time.perf_counter_ns()

    def end_tick(self) -> None:
        end_time = time.perf_counter_ns()
        latency_ms = (end_time - self.tick_start_time) / 1_000_000
        self.tick_latencies.append(latency_ms)
        self.tick_count += 1

    def record_metrics(self, swarm: "Swarm") -> None:
        env = swarm.env
        agents = swarm.agent_manager.agents.values()

        self.metrics.total_tasks = len(env.tasks)
        self.metrics.completed_tasks = sum(1 for t in env.tasks.values() if t.status == TaskStatus.COMPLETED)
        self.metrics.task_completion_rate = self.metrics.completed_tasks / max(1, self.metrics.total_tasks)

        elapsed = time.perf_counter() - self.start_time
        self.metrics.throughput = self.metrics.completed_tasks / max(elapsed, 1e-6)

        required_tasks = sum(1 for t in env.tasks.values() if t.required)
        completed_required = sum(1 for t in env.tasks.values() if t.status == TaskStatus.COMPLETED and t.required)
        self.metrics.coverage = completed_required / max(1, required_tasks)

        self.metrics.total_distance = sum(a.total_distance for a in agents)
        active_agents = [a for a in agents if a.status != AgentStatus.FAILED]
        if active_agents:
            self.metrics.average_path_length = self.metrics.total_distance / len(active_agents)

        self.metrics.energy_consumed = sum(a.energy_consumed for a in agents)

        self.metrics.collision_count = self._count_collisions(agents)
        self.metrics.minimum_clearance = self._min_clearance(agents)

        self.metrics.deadlock_count = sum(1 for a in agents if a.stalled_ticks > self.config.deadlock_threshold)

        if self.tick_latencies:
            self.metrics.mean_tick_latency_ms = sum(self.tick_latencies) / len(self.tick_latencies)
            sorted_latencies = sorted(self.tick_latencies)
            n = len(sorted_latencies)
            self.metrics.p95_tick_latency_ms = sorted_latencies[int(0.95 * n)] if n > 0 else 0
            self.metrics.p99_tick_latency_ms = sorted_latencies[int(0.99 * n)] if n > 0 else 0
            self.metrics.max_tick_latency_ms = max(sorted_latencies) if n > 0 else 0

        self.metrics.failed_agents = sum(1 for a in agents if a.failed)
        self.metrics.recovered_tasks = sum(1 for t in env.tasks.values() if t.status == TaskStatus.COMPLETED and t.assigned_agent_id != t.id)

        self._calculate_fitness()

    def _count_collisions(self, agents) -> int:
        count = 0
        agent_list = list(agents)
        for i in range(len(agent_list)):
            for j in range(i + 1, len(agent_list)):
                a1, a2 = agent_list[i], agent_list[j]
                if a1.failed or a2.failed:
                    continue
                dist = ((a1.position[0] - a2.position[0]) ** 2 + (a1.position[1] - a2.position[1]) ** 2) ** 0.5
                if dist < self.config.min_agent_distance:
                    count += 1
        return count

    def _min_clearance(self, agents) -> float:
        min_dist = float('inf')
        agent_list = list(agents)
        for i in range(len(agent_list)):
            for j in range(i + 1, len(agent_list)):
                a1, a2 = agent_list[i], agent_list[j]
                if a1.failed or a2.failed:
                    continue
                dist = ((a1.position[0] - a2.position[0]) ** 2 + (a1.position[1] - a2.position[1]) ** 2) ** 0.5
                if dist < min_dist:
                    min_dist = dist
        return min_dist

    def _calculate_fitness(self) -> None:
        if self.metrics.collision_count > 0:
            self.metrics.safety_status = "SAFETY_VIOLATION"
            self.metrics.fitness = 0.0
            return

        self.metrics.safety_status = "SAFE"

        throughput_score = min(1.0, self.metrics.throughput / 0.1)
        coverage_score = self.metrics.coverage
        task_completion_score = self.metrics.task_completion_rate
        resilience_score = 1.0 - (self.metrics.failed_agents / max(1, len(self.env.agents))) if hasattr(self, 'env') else 1.0
        energy_score = 1.0 - (self.metrics.energy_consumed / max(1, self.metrics.total_distance * 10)) if self.metrics.total_distance > 0 else 1.0
        path_length_score = 1.0 - min(1.0, self.metrics.average_path_length / 100.0) if self.metrics.average_path_length > 0 else 1.0
        deadlock_score = 1.0 - min(1.0, self.metrics.deadlock_count / max(1, len(self.env.agents))) if hasattr(self, 'env') else 1.0

        self.metrics.fitness = (
            self.config.fitness_throughput_weight * throughput_score +
            self.config.fitness_coverage_weight * coverage_score +
            self.config.fitness_task_completion_weight * task_completion_score +
            self.config.fitness_resilience_weight * resilience_score -
            self.config.fitness_energy_weight * (1 - energy_score) -
            self.config.fitness_path_length_weight * (1 - path_length_score) -
            self.config.fitness_deadlock_weight * (1 - deadlock_score)
        )

    def to_json(self) -> Dict[str, Any]:
        return asdict(self.metrics)

    def save(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_json(), f, indent=2)