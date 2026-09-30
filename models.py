from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class AgentStatus(Enum):
    IDLE = "IDLE"
    MOVING = "MOVING"
    WAITING = "WAITING"
    REPLANNING = "REPLANNING"
    FAILED = "FAILED"
    COMPLETED = "COMPLETED"


class TaskStatus(Enum):
    UNASSIGNED = "UNASSIGNED"
    ASSIGNED = "ASSIGNED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class MessageType(Enum):
    POSITION_UPDATE = "POSITION_UPDATE"
    TASK_ANNOUNCEMENT = "TASK_ANNOUNCEMENT"
    TASK_BID = "TASK_BID"
    TASK_CLAIM = "TASK_CLAIM"
    TRAJECTORY_INTENT = "TRAJECTORY_INTENT"
    FAILURE_NOTICE = "FAILURE_NOTICE"


@dataclass
class Obstacle:
    id: int
    position: Tuple[float, float]
    radius: float = 1.0
    dynamic: bool = False
    velocity: Tuple[float, float] = (0.0, 0.0)
    occupied_cells: Optional[List[Tuple[int, int]]] = None
    active: bool = True


@dataclass
class Hazard:
    id: int
    position: Tuple[float, float]
    radius: float = 2.0
    cost: float = 10.0
    active: bool = True


@dataclass
class Task:
    id: int
    location: Tuple[int, int]
    priority: int = 1
    reward: float = 1.0
    status: TaskStatus = TaskStatus.UNASSIGNED
    assigned_agent_id: Optional[int] = None
    creation_tick: int = 0
    completion_tick: Optional[int] = None
    required: bool = True


@dataclass
class Agent:
    id: int
    position: Tuple[float, float]
    goal: Tuple[int, int]
    velocity: Tuple[float, float] = (0.0, 0.0)
    battery: float = 100.0
    status: AgentStatus = AgentStatus.IDLE
    current_task_id: Optional[int] = None
    trajectory: List[Tuple[float, float]] = field(default_factory=list)
    stalled_ticks: int = 0
    communication_available: bool = True
    last_known_neighbors: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    total_distance: float = 0.0
    energy_consumed: float = 0.0
    completed_tasks: int = 0
    failed: bool = False
    last_seen_flow_field_version: int = 0
    local_observation: Optional[Dict] = None
    last_decision: Optional[Dict] = None
    waiting_since: Optional[int] = None


@dataclass
class Message:
    sender_id: int
    receiver_id: int
    timestamp: int
    message_type: MessageType
    sequence_number: int
    payload: Dict
    ttl: int = 3


@dataclass
class Observation:
    agent_id: int
    position: Tuple[float, float]
    sensing_radius: int
    nearby_obstacles: List[Obstacle]
    nearby_agents: List[int]
    local_congestion: Dict[Tuple[int, int], float]
    local_hazards: List[Hazard]
    local_flow: Dict[Tuple[int, int], Tuple[float, float]]
    nearby_tasks: List[Task]
    communication_state: bool
    grid_bounds: Tuple[int, int, int, int]