from dataclasses import dataclass
from typing import Tuple


@dataclass
class Config:
    grid_width: int = 100
    grid_height: int = 100
    max_ticks: int = 1000

    base_cost: float = 1.0
    congestion_weight: float = 2.5
    visit_weight: float = 1.0
    hazard_weight: float = 10.0
    energy_weight: float = 0.5

    congestion_deposit: float = 0.5
    congestion_decay: float = 0.95

    sensing_radius: int = 5
    min_agent_distance: float = 1.5
    obstacle_margin: float = 1.0
    deadlock_threshold: int = 10

    agent_speed: float = 1.0
    max_battery: float = 100.0
    movement_cost: float = 1.0

    communication_loss_rate: float = 0.0
    message_ttl: int = 3

    fitness_throughput_weight: float = 0.30
    fitness_coverage_weight: float = 0.20
    fitness_task_completion_weight: float = 0.15
    fitness_resilience_weight: float = 0.10
    fitness_energy_weight: float = 0.10
    fitness_path_length_weight: float = 0.05
    fitness_deadlock_weight: float = 0.10

    optimization_max_iterations: int = 50

    seed: int = 42


DEFAULT_CONFIG = Config()
