import pytest
import numpy as np
from config import Config
from models import Agent, Task, AgentStatus


class TestAgents:
    def test_agent_initialization(self):
        config = Config()
        agent = Agent(
            id=1,
            position=(10.0, 10.0),
            goal=(50, 50),
            status=AgentStatus.IDLE
        )
        
        assert agent.id == 1
        assert agent.status == AgentStatus.IDLE
        assert agent.position == (10.0, 10.0)
        assert agent.goal == (50, 50)
        assert agent.battery == 100.0
        assert agent.failed == False
        assert agent.completed_tasks == 0
        assert agent.total_distance == 0.0
        assert agent.energy_consumed == 0.0
        assert agent.current_task_id is None
        assert agent.trajectory == []
        assert agent.stalled_ticks == 0

    def test_agent_status_transitions(self):
        config = Config()
        agent = Agent(
            id=1,
            position=(10.0, 10.0),
            goal=(50, 50),
            status=AgentStatus.IDLE
        )
        
        agent.status = AgentStatus.MOVING
        assert agent.status == AgentStatus.MOVING
        
        agent.status = AgentStatus.WAITING
        assert agent.status == AgentStatus.WAITING
        
        agent.status = AgentStatus.REPLANNING
        assert agent.status == AgentStatus.REPLANNING
        
        agent.failed = True
        assert agent.failed == True
        
        agent.failed = False
        agent.status = AgentStatus.COMPLETED
        assert agent.status == AgentStatus.COMPLETED

    def test_agent_default_values(self):
        config = Config()
        agent = Agent(id=1, position=(0.0, 0.0), goal=(10, 10))
        
        assert agent.velocity == (0.0, 0.0)
        assert agent.communication_available == True
        assert agent.last_known_neighbors == {}
        assert agent.trajectory == []
        assert agent.waiting_since is None

    def test_agent_with_task(self):
        config = Config()
        task = Task(id=1, location=(50, 50), priority=2, reward=10.0)
        
        agent = Agent(id=1, position=(10.0, 10.0), goal=(50, 50))
        agent.current_task_id = task.id
        agent.goal = task.location
        agent.status = AgentStatus.MOVING
        agent.communication_available = True
        
        assert agent.current_task_id == task.id
        assert agent.goal == (50, 50)
        assert agent.status == AgentStatus.MOVING
        assert agent.battery == 100.0

    def test_agent_trajectory(self):
        config = Config()
        agent = Agent(
            id=1,
            position=(10.0, 10.0),
            goal=(50, 50),
            status=AgentStatus.MOVING
        )
        
        agent.trajectory.append((10.0, 10.0))
        agent.trajectory.append((11.0, 11.0))
        agent.trajectory.append((12.0, 12.0))
        
        assert len(agent.trajectory) == 3
        assert agent.trajectory[0] == (10.0, 10.0)
        assert agent.trajectory[-1] == (12.0, 12.0)
