from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Optional
from config import Config
from models import Agent, Obstacle, Hazard, Task, TaskStatus, AgentStatus
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class Visualization:
    def __init__(self, config: Config):
        self.config = config
        self.colors = {
            'agent_active': '#1f77b4',
            'agent_failed': '#d62728',
            'agent_completed': '#2ca02c',
            'goal': '#ff7f0e',
            'obstacle': '#000000',
            'dynamic_obstacle': '#8c564b',
            'hazard': '#e377c2',
            'congestion_low': '#ffffcc',
            'congestion_high': '#ff0000',
            'flow_field': '#17becf'
        }

    def create_simulation_map(self, 
                            agents: Dict[int, Agent], 
                            obstacles: List[Obstacle],
                            dynamic_obstacles: List[Obstacle],
                            hazards: List[Hazard],
                            tasks: Dict[int, Task],
                            congestion: np.ndarray,
                            flow_dx: np.ndarray,
                            flow_dy: np.ndarray,
                            width: int,
                            height: int) -> go.Figure:
        """Create the main simulation visualization"""
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Swarm Simulation', 'Metrics Overview'),
            specs=[[{'type': 'scatter'}, {'type': 'table'}]]
        )
        
        # Main simulation plot
        self._add_grid_background(fig, width, height, row=1, col=1)
        self._add_congestion_heatmap(fig, congestion, width, height, row=1, col=1)
        self._add_obstacles(fig, obstacles, dynamic_obstacles, width, height, row=1, col=1)
        self._add_hazards(fig, hazards, width, height, row=1, col=1)
        self._add_tasks(fig, tasks, width, height, row=1, col=1)
        self._add_agents(fig, agents, width, height, row=1, col=1)
        self._add_flow_vectors(fig, flow_dx, flow_dy, width, height, row=1, col=1)
        
        # Update layout
        fig.update_layout(
            title_text="SWARM-X Simulation",
            showlegend=True,
            height=600,
            width=1200
        )
        
        return fig

    def _add_grid_background(self, fig: go.Figure, width: int, height: int, row: int, col: int) -> None:
        fig.add_shape(
            type="rect",
            x0=0, y0=0, x1=width, y1=height,
            line=dict(color="LightGray", width=1),
            fillcolor="white",
            row=row, col=col
        )

    def _add_congestion_heatmap(self, fig: go.Figure, congestion: np.ndarray, width: int, height: int, row: int, col: int) -> None:
        # Normalize congestion for visualization
        max_congestion = np.max(congestion)
        if max_congestion > 0:
            normalized_congestion = congestion / max_congestion
        else:
            normalized_congestion = congestion
            
        fig.add_trace(
            go.Heatmap(
                z=normalized_congestion.T,
                colorscale=[[0, 'rgba(255,255,255,0)'], [1, 'rgba(255,0,0,0.3)']],
                showscale=False,
                hoverinfo='skip'
            ),
            row=row, col=col
        )

    def _add_obstacles(self, fig: go.Figure, obstacles: List[Obstacle], dynamic_obstacles: List[Obstacle], width: int, height: int, row: int, col: int) -> None:
        # Static obstacles
        static_x, static_y = [], []
        for obs in obstacles:
            if obs.active:
                static_x.append(obs.position[0])
                static_y.append(obs.position[1])
        
        if static_x:
            fig.add_trace(
                go.Scatter(
                    x=static_x, y=static_y,
                    mode='markers',
                    marker=dict(
                        symbol='square',
                        size=10,
                        color=self.colors['obstacle'],
                        line=dict(width=1, color='DarkSlateGray')
                    ),
                    name='Static Obstacles',
                    hovertemplate='Static Obstacle<br>x: %{x}<br>y: %{y}<extra></extra>'
                ),
                row=row, col=col
            )
        
        # Dynamic obstacles
        dynamic_x, dynamic_y = [], []
        for obs in dynamic_obstacles:
            if obs.active:
                dynamic_x.append(obs.position[0])
                dynamic_y.append(obs.position[1])
        
        if dynamic_x:
            fig.add_trace(
                go.Scatter(
                    x=dynamic_x, y=dynamic_y,
                    mode='markers',
                    marker=dict(
                        symbol='diamond',
                        size=12,
                        color=self.colors['dynamic_obstacle'],
                        line=dict(width=1, color='DarkSlateGray')
                    ),
                    name='Dynamic Obstacles',
                    hovertemplate='Dynamic Obstacle<br>x: %{x}<br>y: %{y}<extra></extra>'
                ),
                row=row, col=col
            )

    def _add_hazards(self, fig: go.Figure, hazards: List[Hazard], width: int, height: int, row: int, col: int) -> None:
        hazard_x, hazard_y = [], []
        for haz in hazards:
            if haz.active:
                hazard_x.append(haz.position[0])
                hazard_y.append(haz.position[1])
        
        if hazard_x:
            fig.add_trace(
                go.Scatter(
                    x=hazard_x, y=hazard_y,
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up',
                        size=8,
                        color=self.colors['hazard'],
                        line=dict(width=1, color='DarkSlateGray')
                    ),
                    name='Hazards',
                    hovertemplate='Hazard<br>x: %{x}<br>y: %{y}<extra></extra>'
                ),
                row=row, col=col
            )

    def _add_tasks(self, fig: go.Figure, tasks: Dict[int, Task], width: int, height: int, row: int, col: int) -> None:
        unassigned_x, unassigned_y = [], []
        assigned_x, assigned_y = [], []
        completed_x, completed_y = [], []
        
        for task in tasks.values():
            x, y = task.location
            if task.status == TaskStatus.UNASSIGNED:
                unassigned_x.append(x)
                unassigned_y.append(y)
            elif task.status == TaskStatus.ASSIGNED or task.status == TaskStatus.IN_PROGRESS:
                assigned_x.append(x)
                assigned_y.append(y)
            elif task.status == TaskStatus.COMPLETED:
                completed_x.append(x)
                completed_y.append(y)
        
        if unassigned_x:
            fig.add_trace(
                go.Scatter(
                    x=unassigned_x, y=unassigned_y,
                    mode='markers',
                    marker=dict(
                        symbol='circle-open',
                        size=8,
                        color=self.colors['goal'],
                        line=dict(width=2, color='DarkOrange')
                    ),
                    name='Unassigned Tasks',
                    hovertemplate='Task<br>x: %{x}<br>y: %{y}<extra></extra>'
                ),
                row=row, col=col
            )
        
        if assigned_x:
            fig.add_trace(
                go.Scatter(
                    x=assigned_x, y=assigned_y,
                    mode='markers',
                    marker=dict(
                        symbol='circle',
                        size=8,
                        color=self.colors['goal'],
                        line=dict(width=2, color='DarkOrange')
                    ),
                    name='Assigned Tasks',
                    hovertemplate='Task<br>x: %{x}<br>y: %{y}<extra></extra>'
                ),
                row=row, col=col
            )
        
        if completed_x:
            fig.add_trace(
                go.Scatter(
                    x=completed_x, y=completed_y,
                    mode='markers',
                    marker=dict(
                        symbol='circle',
                        size=8,
                        color=self.colors['goal'],
                        line=dict(width=2, color='DarkGreen')
                    ),
                    name='Completed Tasks',
                    hovertemplate='Task<br>x: %{x}<br>y: %{y}<extra></extra>'
                ),
                row=row, col=col
            )

    def _add_agents(self, fig: go.Figure, agents: Dict[int, Agent], width: int, height: int, row: int, col: int) -> None:
        active_x, active_y = [], []
        failed_x, failed_y = [], []
        completed_x, completed_y = [], []
        waiting_x, waiting_y = [], []
        
        for agent in agents.values():
            x, y = agent.position
            if agent.status == AgentStatus.FAILED:
                failed_x.append(x)
                failed_y.append(y)
            elif agent.status == AgentStatus.COMPLETED:
                completed_x.append(x)
                completed_y.append(y)
            elif agent.status == AgentStatus.WAITING:
                waiting_x.append(x)
                waiting_y.append(y)
            else:  # IDLE or MOVING
                active_x.append(x)
                active_y.append(y)
        
        if active_x:
            fig.add_trace(
                go.Scatter(
                    x=active_x, y=active_y,
                    mode='markers',
                    marker=dict(
                        symbol='circle',
                        size=6,
                        color=self.colors['agent_active'],
                        line=dict(width=1, color='DarkBlue')
                    ),
                    name='Active Agents',
                    hovertemplate='Agent<br>x: %{x}<br>y: %{y}<br>status: %{customdata}<extra></extra>',
                    customdata=['Active'] * len(active_x)
                ),
                row=row, col=col
            )
        
        if failed_x:
            fig.add_trace(
                go.Scatter(
                    x=failed_x, y=failed_y,
                    mode='markers',
                    marker=dict(
                        symbol='x',
                        size=8,
                        color=self.colors['agent_failed'],
                        line=dict(width=2, color='DarkRed')
                    ),
                    name='Failed Agents',
                    hovertemplate='Agent<br>x: %{x}<br>y: %{y}<br>status: %{customdata}<extra></extra>',
                    customdata=['Failed'] * len(failed_x)
                ),
                row=row, col=col
            )
        
        if completed_x:
            fig.add_trace(
                go.Scatter(
                    x=completed_x, y=completed_y,
                    mode='markers',
                    marker=dict(
                        symbol='circle',
                        size=6,
                        color=self.colors['agent_completed'],
                        line=dict(width=1, color='DarkGreen')
                    ),
                    name='Completed Agents',
                    hovertemplate='Agent<br>x: %{x}<br>y: %{y}<br>status: %{customdata}<extra></extra>',
                    customdata=['Completed'] * len(completed_x)
                ),
                row=row, col=col
            )
        
        if waiting_x:
            fig.add_trace(
                go.Scatter(
                    x=waiting_x, y=waiting_y,
                    mode='markers',
                    marker=dict(
                        symbol='square',
                        size=6,
                        color=self.colors['agent_active'],
                        line=dict(width=1, color='DarkBlue')
                    ),
                    name='Waiting Agents',
                    hovertemplate='Agent<br>x: %{x}<br>y: %{y}<br>status: %{customdata}<extra></extra>',
                    customdata=['Waiting'] * len(waiting_x)
                ),
                row=row, col=col
            )

    def _add_flow_vectors(self, fig: go.Figure, flow_dx: np.ndarray, flow_dy: np.ndarray, width: int, height: int, row: int, col: int) -> None:
        # Sample flow vectors to avoid overcrowding
        step = max(1, min(width, height) // 20)
        x_samples = np.arange(0, width, step)
        y_samples = np.arange(0, height, step)
        
        u_samples = flow_dx[np.ix_(y_samples, x_samples)]
        v_samples = flow_dy[np.ix_(y_samples, x_samples)]
        
        # Normalize for display
        magnitude = np.sqrt(u_samples**2 + v_samples**2)
        max_mag = np.max(magnitude)
        if max_mag > 0:
            u_samples = u_samples / max_mag * 0.3
            v_samples = v_samples / max_mag * 0.3
        
        xx, yy = np.meshgrid(x_samples, y_samples)
        
        fig.add_trace(
            go.Quiver(
                x=xx.flatten(),
                y=yy.flatten(),
                u=u_samples.flatten(),
                v=v_samples.flatten(),
                scale=1,
                scale_units='inches',
                line=dict(width=1, color=self.colors['flow_field']),
                name='Flow Field',
                hoverinfo='skip'
            ),
            row=row, col=col
        )

    def create_metrics_dashboard(self, metrics_history: List[Dict]) -> go.Figure:
        """Create a dashboard showing metrics over time"""
        if not metrics_history:
            # Return empty figure if no history
            fig = go.Figure()
            fig.add_annotation(
                text="No metrics data available yet",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        # Extract time series data
        ticks = [m.get('tick', i) for i, m in enumerate(metrics_history)]
        fitness = [m.get('fitness', 0.0) for m in metrics_history]
        throughput = [m.get('throughput', 0.0) for m in metrics_history]
        coverage = [m.get('coverage', 0.0) for m in metrics_history]
        collisions = [m.get('collision_count', 0) for m in metrics_history]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Fitness Over Time', 'Throughput Over Time', 
                          'Coverage Over Time', 'Collisions Over Time'),
            vertical_spacing=0.12
        )
        
        fig.add_trace(
            go.Scatter(x=ticks, y=fitness, mode='lines', name='Fitness',
                      line=dict(color='blue')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=ticks, y=throughput, mode='lines', name='Throughput',
                      line=dict(color='green')),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(x=ticks, y=coverage, mode='lines', name='Coverage',
                      line=dict(color='orange')),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=ticks, y=collisions, mode='lines+markers', name='Collisions',
                      line=dict(color='red')),
            row=2, col=2
        )
        
        fig.update_layout(
            height=500,
            showlegend=False,
            title_text="SWARM-X Metrics Dashboard"
        )
        
        return fig

    def create_decision_trace(self, agent: Agent) -> go.Figure:
        """Create a visualization of an agent's decision process"""
        fig = go.Figure()
        
        # Agent trajectory
        if agent.trajectory:
            traj_x, traj_y = zip(*agent.trajectory)
            fig.add_trace(
                go.Scatter(
                    x=traj_x, y=traj_y,
                    mode='lines+markers',
                    name='Trajectory',
                    line=dict(color='blue', width=2),
                    marker=dict(size=4)
                )
            )
        
        # Current position
        fig.add_trace(
            go.Scatter(
                x=[agent.position[0]], y=[agent.position[1]],
                mode='markers',
                name='Current Position',
                marker=dict(
                    size=12,
                    color='red' if agent.failed else 'green',
                    symbol='circle'
                )
            )
        )
        
        # Goal position
        if hasattr(agent, 'goal'):
            fig.add_trace(
                go.Scatter(
                    x=[agent.goal[0]], y=[agent.goal[1]],
                    mode='markers',
                    name='Goal',
                    marker=dict(
                        size=10,
                        color='orange',
                        symbol='star'
                    )
                )
            )
        
        fig.update_layout(
            title=f"Agent {agent.id} Decision Trace",
            xaxis_title="X Position",
            yaxis_title="Y Position",
            showlegend=True,
            width=600,
            height=400
        )
        
        return fig

    def create_event_log(self, events: List[Dict]) -> go.Figure:
        """Create an event log visualization"""
        fig = go.Figure()
        
        if not events:
            fig.add_annotation(
                text="No events recorded",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        # Create table-like display
        event_texts = []
        for event in events[-20:]:  # Show last 20 events
            timestamp = event.get('tick', '?')
            event_type = event.get('type', 'Unknown')
            description = event.get('description', '')
            event_texts.append(f"Tick {timestamp}: {event_type} - {description}")
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['Event Log (Most Recent)'],
                    fill_color='paleturquoise',
                    align='left'
                ),
                cells=dict(
                    values=[event_texts[::-1]],  # Reverse to show most recent first
                    fill_color='lavender',
                    align='left'
                )
            )
        )
        
        fig.update_layout(
            title="SWARM-X Event Log",
            height=400
        )
        
        return fig