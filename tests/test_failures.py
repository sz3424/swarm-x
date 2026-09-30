import pytest
import numpy as np
from config import Config
from models import TaskStatus, AgentStatus, Agent, Task
from environment import Environment


class TestFailure:
    def test_failed_agent_stops(self):
        config = Config()
        env = Environment(config)
        env.create_environment()
        
        from swarm import Swarm
        swarm = Swarm(config, env)
        swarm.initialize(5)
        
        agent = list(swarm.agent_manager.agents.values())[0]
        agent.failed = True
        agent.status = AgentStatus.FAILED
        
        old_position = agent.position
        swarm.step()
        
        assert agent.position == old_position

    def test_task_is_released_on_failure(self):
        config = Config()
        env = Environment(config)
        env.create_environment()
        
        from swarm import Swarm
        swarm = Swarm(config, env)
        swarm.initialize(5)
        
        agent = list(swarm.agent_manager.agents.values())[0]
        task = list(env.tasks.values())[0]
        
        agent.current_task_id = task.id
        task.status = TaskStatus.ASSIGNED
        task.assigned_agent_id = agent.id
        
        agent.failed = True
        agent.status = AgentStatus.FAILED
        
        if agent.current_task_id is not None:
            released_task = env.tasks.get(agent.current_task_id)
            if released_task:
                released_task.status = TaskStatus.UNASSIGNED
                released_task.assigned_agent_id = None
                agent.current_task_id = None
        
        assert task.status == TaskStatus.UNASSIGNED

    def test_failed_agent_does_not_bid(self):
        config = Config()
        env = Environment(config)
        env.create_environment()
        
        from swarm import Swarm
        swarm = Swarm(config, env)
        swarm.initialize(5)
        
        idle_agents = [a for a in swarm.agent_manager.agents.values() if a.status == AgentStatus.IDLE]
        available_tasks = [t for t in env.tasks.values() if t.status == TaskStatus.UNASSIGNED]
        
        assert len(idle_agents) > 0
        assert len(available_tasks) > 0
        
        agent = idle_agents[0]
        agent.failed = True
        agent.status = AgentStatus.FAILED

    def test_agent_failure_recovery(self):
        config = Config()
        env = Environment(config)
        env.create_environment()
        
        from swarm import Swarm
        swarm = Swarm(config, env)
        swarm.initialize(5)
        
        agent = list(swarm.agent_manager.agents.values())[0]
        task = list(env.tasks.values())[0]
        
        agent.current_task_id = task.id
        task.status = TaskStatus.ASSIGNED
        task.assigned_agent_id = agent.id
        
        agent.failed = True
        agent.status = AgentStatus.FAILED
        
        if agent.current_task_id is not None:
            released_task = env.tasks.get(agent.current_task_id)
            if released_task:
                released_task.status = TaskStatus.UNASSIGNED
                released_task.assigned_agent_id = None
                agent.current_task_id = None
        
        assert task.status == TaskStatus.UNASSIGNED
        assert task.assigned_agent_id is None

    def test_multiple_agents_failure(self):
        config = Config()
        env = Environment(config)
        env.create_environment()
        
        from swarm import Swarm
        swarm = Swarm(config, env)
        swarm.initialize(10)
        
        agents = list(swarm.agent_manager.agents.values())
        
        for agent in agents[:3]:
            agent.failed = True
            agent.status = AgentStatus.FAILED
            if agent.current_task_id is not None:
                task = env.tasks.get(agent.current_task_id)
                if task:
                    task.status = TaskStatus.UNASSIGNED
                    task.assigned_agent_id = None
                    agent.current_task_id = None
        
        for agent in agents[:3]:
            assert agent.failed == True
            assert agent.status == AgentStatus.FAILED
        
        active_agents = [a for a in agents if not a.failed]
        assert len(active_agents) > 0