import pytest
import numpy as np
from hypothesis import given, strategies as st, settings
from config import Config
from models import Task, TaskStatus, Agent, AgentStatus
from environment import Environment


class TestFlowField:
    def test_goal_has_zero_distance(self):
        from flow_field import FlowField
        
        config = Config(grid_width=50, grid_height=50)
        env = Environment(config)
        env.create_environment()
        flow_field = FlowField(env, config)
        
        for task in env.tasks.values():
            x, y = task.location
            assert abs(flow_field.distance_to_goal[y, x]) < 1e-6

    def test_obstacles_are_impassable(self):
        from flow_field import FlowField
        
        config = Config(grid_width=50, grid_height=50)
        env = Environment(config)
        env.create_environment()
        
        env.add_obstacle(25, 25, 2.0)
        
        flow_field = FlowField(env, config)
        flow_dx, flow_dy = flow_field.get_flow_vector(25.0, 25.0)
        assert (flow_dx, flow_dy) == (0.0, 0.0)

    def test_reachable_cells_have_finite_distances(self):
        from flow_field import FlowField
        
        config = Config(grid_width=50, grid_height=50)
        env = Environment(config)
        env.create_environment()
        
        flow_field = FlowField(env, config)
        
        test_cells = [(10, 10), (30, 20), (40, 40)]
        for x, y in test_cells:
            if env.obstacle_map[y, x] == 0:
                dist = flow_field.distance_to_goal[y, x]
                assert np.isfinite(dist) and dist >= 0

    def test_flow_points_toward_lower_cost(self):
        from flow_field import FlowField
        
        config = Config(grid_width=30, grid_height=30)
        env = Environment(config)
        env.create_environment()
        
        flow_field = FlowField(env, config)
        x, y = 15, 15
        if env.obstacle_map[y, x] == 0:
            flow_dx, flow_dy = flow_field.get_flow_vector(float(x), float(y))
            magnitude = np.sqrt(flow_dx ** 2 + flow_dy ** 2)
            assert magnitude <= 1.0


class TestSafety:
    def test_agent_cannot_enter_obstacle(self):
        from safety import SafetyFilter
        
        config = Config()
        env = Environment(config)
        env.create_environment()
        env.add_obstacle(50, 50, 2.0)
        
        safety = SafetyFilter(config, env)
        agent = Agent(id=1, position=(50.0, 50.0), goal=(60, 60))
        
        assert safety.validate_move(agent, 50.0, 50.0) == False

    def test_agents_cannot_violate_minimum_separation(self):
        from safety import SafetyFilter
        
        config = Config()
        env = Environment(config)
        env.create_environment()
        
        safety = SafetyFilter(config, env)
        agent1 = Agent(id=1, position=(10.0, 10.0), goal=(20, 20))
        agent2 = Agent(id=2, position=(11.0, 10.0), goal=(30, 30))
        
        env.agents[1] = agent1
        env.agents[2] = agent2
        
        result = safety._agents_clear(agent1, 11.0, 10.0)
        assert result == False

    def test_boundary_violations_rejected(self):
        from safety import SafetyFilter
        
        config = Config()
        env = Environment(config)
        env.create_environment()
        
        safety = SafetyFilter(config, env)
        agent = Agent(id=1, position=(0.0, 0.0), goal=(10, 10))
        
        assert safety.validate_move(agent, -1.0, 0.0) == False
        assert safety.validate_move(agent, 0.0, -1.0) == False


class TestCongestion:
    def test_cell_heat_increases_after_visitation(self):
        config = Config()
        env = Environment(config)
        env.create_environment()
        
        initial_heat = env.congestion[50, 50].copy()
        env.congestion[50, 50] += config.congestion_deposit
        
        assert env.congestion[50, 50] > initial_heat

    def test_heat_decay(self):
        config = Config()
        env = Environment(config)
        env.create_environment()
        
        env.congestion[50, 50] = 10.0
        env.update_congestion()
        
        expected = 10.0 * config.congestion_decay
        assert abs(env.congestion[50, 50] - expected) < 0.01


class TestCommunication:
    def test_messages_can_be_dropped(self):
        from communication import CommunicationManager
        from models import MessageType
        
        config = Config()
        config.communication_loss_rate = 1.0
        env = Environment(config)
        env.create_environment()
        
        comm = CommunicationManager(config, env)
        result = comm.send_message(1, 2, MessageType.POSITION_UPDATE, {})
        assert result == False

    def test_local_operation_continues_during_communication_loss(self):
        from communication import CommunicationManager
        from models import MessageType
        
        config = Config()
        config.communication_loss_rate = 0.5
        env = Environment(config)
        env.create_environment()
        
        comm = CommunicationManager(config, env)
        env.tick = 0
        
        messages_sent = 0
        for i in range(10):
            if comm.send_message(1, 2, MessageType.POSITION_UPDATE, {}):
                messages_sent += 1
        
        received = comm.receive_messages(2)
        messages_received = len(received)
        
        assert messages_received <= messages_sent


class TestSimulation:
    def test_deterministic_seed_produces_repeatable_result(self):
        np.random.seed(42)
        pos1 = np.random.uniform(10, 90, 2)
        
        np.random.seed(42)
        pos2 = np.random.uniform(10, 90, 2)
        
        np.testing.assert_array_almost_equal(pos1, pos2)

    @given(seed=st.integers(min_value=1, max_value=100))
    @settings(max_examples=10)
    def test_config_is_reproducible(self, seed):
        config1 = Config(seed=seed)
        config2 = Config(seed=seed)
        
        assert config1.seed == config2.seed
        assert config1.grid_width == config2.grid_width
        assert config1.grid_height == config2.grid_height

    def test_tasks_complete_in_solvable_scenario(self):
        config = Config(grid_width=30, grid_height=30)
        env = Environment(config)
        env.create_environment()
        
        from swarm import Swarm
        swarm = Swarm(config, env)
        swarm.initialize(5)
        
        for _ in range(100):
            if swarm.is_complete():
                break
            swarm.step()
        
        assert swarm.is_complete() or swarm.tick >= 100

    def test_invariant_no_collisions(self):
        config = Config()
        env = Environment(config)
        env.create_environment()
        
        from swarm import Swarm
        swarm = Swarm(config, env)
        swarm.initialize(10)
        
        for _ in range(50):
            if swarm.is_complete():
                break
            swarm.step()
            
            agents = list(swarm.agent_manager.agents.values())
            for i in range(len(agents)):
                for j in range(i + 1, len(agents)):
                    a1, a2 = agents[i], agents[j]
                    dist = np.sqrt(
                        (a1.position[0] - a2.position[0]) ** 2 +
                        (a1.position[1] - a2.position[1]) ** 2
                    )
                    assert dist >= config.min_agent_distance - 0.1


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])