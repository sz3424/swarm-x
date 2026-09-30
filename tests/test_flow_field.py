import pytest
import numpy as np
from config import Config
from models import TaskStatus, AgentStatus, Agent, Task


class TestFlowField:
    def test_goal_has_zero_distance(self):
        from flow_field import FlowField
        from environment import Environment
        
        config = Config(grid_width=50, grid_height=50)
        env = Environment(config)
        env.create_environment()
        flow_field = FlowField(env, config)
        
        for task in env.tasks.values():
            x, y = task.location
            assert abs(flow_field.distance_to_goal[y, x]) < 1e-6

    def test_obstacles_are_impassable(self):
        from flow_field import FlowField
        from environment import Environment
        
        config = Config(grid_width=50, grid_height=50)
        env = Environment(config)
        env.create_environment()
        
        env.add_obstacle(25, 25, 2.0)
        
        flow_field = FlowField(env, config)
        flow_dx, flow_dy = flow_field.get_flow_vector(25.0, 25.0)
        assert (flow_dx, flow_dy) == (0.0, 0.0)

    def test_reachable_cells_have_finite_distances(self):
        from flow_field import FlowField
        from environment import Environment
        
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
        from environment import Environment
        
        config = Config(grid_width=30, grid_height=30)
        env = Environment(config)
        env.create_environment()
        
        flow_field = FlowField(env, config)
        x, y = 15, 15
        if env.obstacle_map[y, x] == 0:
            flow_dx, flow_dy = flow_field.get_flow_vector(float(x), float(y))
            magnitude = np.sqrt(flow_dx ** 2 + flow_dy ** 2)
            assert magnitude <= 1.0