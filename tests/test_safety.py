import pytest
import numpy as np
from config import Config
from models import Agent, AgentStatus


class TestSafety:
    def test_agent_cannot_enter_obstacle(self):
        from safety import SafetyFilter
        from environment import Environment
        
        config = Config()
        env = Environment(config)
        env.create_environment()
        env.add_obstacle(50, 50, 2.0)
        
        safety = SafetyFilter(config, env)
        agent = Agent(
            id=1,
            position=(50.0, 50.0),
            goal=(60, 60),
            status=AgentStatus.MOVING
        )
        
        assert safety.validate_move(agent, 50.0, 50.0) == False

    def test_agents_cannot_violate_minimum_separation(self):
        from safety import SafetyFilter
        from environment import Environment
        
        config = Config()
        env = Environment(config)
        env.create_environment()
        
        safety = SafetyFilter(config, env)
        
        agent1 = Agent(
            id=1,
            position=(10.0, 10.0),
            goal=(20, 20),
            status=AgentStatus.MOVING
        )
        
        agent2 = Agent(
            id=2,
            position=(11.0, 10.0),
            goal=(30, 30),
            status=AgentStatus.MOVING
        )
        
        env.agents[1] = agent1
        env.agents[2] = agent2
        
        result = safety._agents_clear(agent1, 11.0, 10.0)
        assert result == False

    def test_boundary_violations_rejected(self):
        from safety import SafetyFilter
        from environment import Environment
        
        config = Config()
        env = Environment(config)
        env.create_environment()
        
        safety = SafetyFilter(config, env)
        agent = Agent(
            id=1,
            position=(0.0, 0.0),
            goal=(10, 10),
            status=AgentStatus.MOVING
        )
        
        assert safety.validate_move(agent, -1.0, 0.0) == False
        assert safety.validate_move(agent, 0.0, -1.0) == False

    def test_distance_to_obstacle_zero_at_obstacle(self):
        from safety import SafetyFilter
        from environment import Environment
        
        config = Config()
        env = Environment(config)
        env.create_environment()
        env.add_obstacle(50, 50, 2.0)
        
        safety = SafetyFilter(config, env)
        agent = Agent(
            id=1,
            position=(50.0, 50.0),
            goal=(60, 60),
            status=AgentStatus.MOVING
        )
        
        dist = safety.distance_to_obstacle(agent, 50.0, 50.0)
        assert dist == 0.0