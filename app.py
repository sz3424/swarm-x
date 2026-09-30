from __future__ import annotations

import argparse
import json
import sys
import os
from datetime import datetime
from typing import Dict, Any

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import Config, DEFAULT_CONFIG
from environment import Environment
from swarm import Swarm
from metrics import MetricsCollector
from scenarios import create_scenario, Round1Scenario, Round2Scenario, FinalScenario
from visualization import Visualization


def init_session_state() -> None:
    """Initialize Streamlit session state"""
    defaults = {
        'initialized': False,
        'swarm': None,
        'metrics': None,
        'current_scenario': None,
        'metrics_history': [],
        'events': [],
        'running': False,
        'paused': False,
        'tick': 0,
        'config': DEFAULT_CONFIG.copy() if hasattr(DEFAULT_CONFIG, 'copy') else DEFAULT_CONFIG,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def setup_page_config() -> None:
    """Set up the Streamlit page configuration"""
    st.set_page_config(
        page_title="SWARM-X: Adaptive Stigmergic Flow-Field Navigation",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def render_kpi_bar(swarm: Swarm, metrics: MetricsCollector) -> None:
    """Render the top KPI bar"""
    # Get current state
    agents = swarm.agent_manager.get_active_agents() if swarm else []
    active_count = len(agents)
    
    # Calculate metrics
    completed_tasks = sum(1 for t in swarm.env.tasks.values() if t.status == TaskStatus.COMPLETED)
    total_tasks = len(swarm.env.tasks)
    task_completion = completed_tasks / max(1, total_tasks)
    
    # Throughput
    elapsed = max(1, swarm.tick / 60)  # approximate
    throughput = completed_tasks / elapsed if elapsed > 0 else 0
    
    # Coverage
    required_tasks = sum(1 for t in swarm.env.tasks.values() if t.required)
    completed_required = sum(1 for t in swarm.env.tasks.values() 
                           if t.status == TaskStatus.COMPLETED and t.required)
    coverage = completed_required / max(1, required_tasks)
    
    # Collisions
    collision_count = sum(1 for t in swarm.env.tasks.values() if t.status == TaskStatus.FAILED)
    
    # Deadlocks
    deadlock_count = sum(1 for a in agents if a.stalled_ticks >= swarm.config.deadlock_threshold)
    
    # P95 Latency
    if metrics and metrics.tick_latencies:
        sorted_latencies = sorted(metrics.tick_latencies)
        p95_idx = int(0.95 * len(sorted_latencies))
        p95_latency = sorted_latencies[p95_idx] if sorted_latencies else 0
    else:
        p95_latency = 0
    
    # Energy
    energy = sum(a.energy_consumed for a in swarm.agent_manager.agents) if swarm else 0
    
    # Fitness
    fitness = metrics.metrics.fitness if metrics and metrics.metrics.fitness else 0.0
    safety_status = metrics.metrics.safety_status if metrics else "SAFE"
    
    # Display KPI bar
    col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = st.columns(10)
    
    with col1:
        st.metric("SWARM HEALTH", f"{active_count}/{len(swarm.agent_manager.agents)}")
    
    with col2:
        st.metric("ACTIVE AGENTS", active_count)
    
    with col3:
        st.metric("TASK COMPLETION", f"{task_completion:.1%}")
    
    with col4:
        st.metric("THROUGHPUT", f"{throughput:.2f}/tick")
    
    with col5:
        st.metric("COVERAGE", f"{coverage:.1%}")
    
    with col6:
        st.metric("COLLISIONS", collision_count)
    
    with col7:
        st.metric("DEADLOCKS", deadlock_count)
    
    with col8:
        st.metric("P95 LATENCY", f"{p95_latency:.1f}ms")
    
    with col9:
        st.metric("ENERGY", f"{energy:.1f}")
    
    with col10:
        safety_color = "normal" if safety_status == "SAFE" else "inverse"
        st.metric("FITNESS", f"{fitness:.3f}", delta=safety_status)


def render_simulation_map(visualizer: Visualization, swarm: Swarm) -> go.Figure:
    """Render the main simulation map"""
    env = swarm.env
    
    fig = visualizer.create_simulation_map(
        agents={a.id: a for a in swarm.agent_manager.agents.values()},
        obstacles=[],
        dynamic_obstacles=env.dynamic_obstacles,
        hazards=env.hazards,
        tasks=env.tasks,
        congestion=env.congestion,
        flow_dx=np.zeros((env.height, env.width)),
        flow_dy=np.zeros((env.height, env.width)),
        width=env.width,
        height=env.height
    )
    
    return fig


def render_event_log(events: List[Dict]) -> None:
    """Render the event log"""
    if not events:
        st.info("No events recorded yet")
        return
    
    # Show last 20 events
    recent_events = events[-20:]
    
    # Convert to table
    event_texts = []
    for event in reversed(recent_events):
        tick = event.get('tick', '?')
        event_type = event.get('type', 'Unknown')
        desc = event.get('description', '')
        event_texts.append(f"Tick {tick}: {event_type} - {desc}")
    
    fig = go.Figure(data=[go.Table(
        header=dict(values=['SWARM-X Events'], fill_color='paleturquoise', align='left'),
        cells=dict(values=[event_texts], fill_color='lavender', align='left')
    )])
    
    st.plotly_chart(fig, use_container_width=True)


def render_convergence_chart(metrics_history: List[Dict]) -> None:
    """Render the convergence visualization"""
    if not metrics_history:
        st.info("No metrics data available for convergence chart")
        return
    
    ticks = [m.get('tick', i) for i, m in enumerate(metrics_history[-50:])]
    fitness = [m.get('fitness', 0.0) for m in metrics_history[-50:]]
    throughput = [m.get('throughput', 0.0) for m in metrics_history[-50:]]
    coverage = [m.get('coverage', 0.0) for m in metrics_history[-50:]]
    
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=('Fitness vs Iteration', 'Throughput vs Iteration',
                                       'Coverage vs Iteration', 'Latency vs Iteration'))
    
    fig.add_trace(go.Scatter(x=ticks, y=fitness, mode='lines', name='Fitness', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=ticks, y=throughput, mode='lines', name='Throughput', line=dict(color='green')), row=1, col=2)
    fig.add_trace(go.Scatter(x=ticks, y=coverage, mode='lines', name='Coverage', line=dict(color='orange')), row=2, col=1)
    
    # Add latency if available
    if len(metrics_history) > 1 and 'mean_tick_latency_ms' in metrics_history[0]:
        latency = [m.get('mean_tick_latency_ms', 0) for m in metrics_history[-50:]]
        fig.add_trace(go.Scatter(x=ticks, y=latency, mode='lines', name='Latency p95', line=dict(color='red')), row=2, col=2)
    
    fig.update_layout(height=600, showlegend=False, title_text="Convergence Visualization")
    st.plotly_chart(fig, use_container_width=True)


def handle_perturbation(perturbation: str, swarm: Swarm) -> None:
    """Handle interactive perturbation from control panel"""
    if perturbation == "ADD OBSTACLE":
        x = np.random.uniform(10, 90)
        y = np.random.uniform(10, 90)
        swarm.env.add_obstacle(x, y, 2.0)
        swarm.env.environment_changed = True
        swarm.flow_field.version += 1
        st.toast("Obstacle added at ({:.1f}, {:.1f})".format(x, y))
    
    elif perturbation == "MOVE OBSTACLE":
        for obs in swarm.env.dynamic_obstacles:
            if obs.active:
                obs.position = (
                    obs.position[0] + np.random.uniform(-1, 1),
                    obs.position[1] + np.random.uniform(-1, 1)
                )
        swarm.env.environment_changed = True
        swarm.flow_field.version += 1
        st.toast("Obstacle moved")
    
    elif perturbation == "FAIL AGENT":
        agents = [a for a in swarm.agent_manager.agents.values() if not a.failed]
        if agents:
            agent = agents[0]
            agent.failed = True
            agent.status = 'FAILED'
            if agent.current_task_id is not None:
                task = swarm.env.tasks.get(agent.current_task_id)
                if task:
                    task.status = TaskStatus.UNASSIGNED
                    task.assigned_agent_id = None
                    agent.current_task_id = None
            # Re-bid tasks to nearby agents
            nearby_agents = [a for a in swarm.agent_manager.agents.values() 
                           if not a.failed and a.current_task_id is None]
            for a in nearby_agents[:3]:
                bid_result = swarm._calculate_bid(a, task) if 'task' in dir() else 0
            swarm.env.environment_changed = True
            swarm.flow_field.version += 1
            st.toast(f"Agent {agent.id} failed")
    
    elif perturbation == "DROP COMMUNICATION":
        for agent in swarm.agent_manager.agents.values():
            agent.communication_available = not agent.communication_available
        st.toast("Communication state toggled")


def run_tick(swarm: Swarm, metrics: MetricsCollector, events: List[Dict]) -> None:
    """Execute one simulation tick"""
    if swarm.running and not swarm.paused:
        start_time = __import__('time').perf_counter_ns()
        
        result = swarm.step()
        
        # Record metrics
        metrics.end_tick()
        metrics.record_metrics(swarm)
        
        # Record events
        tick_events = {
            'tick': swarm.tick,
            'events': []
        }
        
        # Check for task completions
        for task in swarm.env.tasks.values():
            if task.status == TaskStatus.COMPLETED:
                tick_events['events'].append({
                    'type': 'TASK_COMPLETED',
                    'description': f'Task {task.id} completed',
                    'tick': swarm.tick
                })
        
        # Check for agent failures
        for agent in swarm.agent_manager.agents.values():
            if agent.failed and agent.stalled_ticks >= swarm.config.deadlock_threshold:
                tick_events['events'].append({
                    'type': 'DEADLOCK_DETECTED',
                    'description': f'Agnet {agent.id} stalled {agent.stalled_ticks} ticks',
                    'tick': swarm.tick
                })
        
        events.extend(tick_events.get('events', []))
        
        elapsed_ms = (time.perf_counter_ns() - start_time) / 1_000_000
        metrics.tick_latencies.append(elapsed_ms)
        metrics.tick_count += 1
        
        st.session_state.tick = swarm.tick


def render_control_panel(swarm: Swarm, metrics: MetricsCollector, 
                         events: List[Dict], visualization: Visualization) -> None:
    """Render the control panel sidebar"""
    st.sidebar.header("SWARM-X Control Panel")
    
    # Scenario selection
    scenario_name = st.sidebar.selectbox(
        "Scenario",
        ["round1", "round2", "final"],
        index=["round1", "round2", "final"].index(swarm.current_scenario.name)
if swarm and swarm.current_scenario else 0
    )
    
    # Agent count
    agent_count = st.sidebar.slider(
        "Agent Count",
        min_value=5,
        max_value=100,
        value=len(swarm.agent_manager.agents) if swarm else 20
    )
    
    # Grid size
    grid_size = st.sidebar.slider(
        "Grid Size",
        min_value=30,
        max_value=200,
        value=swarm.env.width if swarm else 100
    )
    
    # Communication loss
    comm_loss = st.sidebar.slider(
        "Communication Loss Rate",
        min_value=0.0,
        max_value=1.0,
        value=swarm.config.communication_loss_rate if swarm else 0.0,
        step=0.01
    )
    
    # Congestion weight
    congestion_weight = st.sidebar.slider(
        "Congestion Weight",
        min_value=0.1,
        max_value=10.0,
        value=swarm.config.congestion_weight if swarm else 2.5,
        step=0.1
    )
    
    # Hazard weight
    hazard_weight = st.sidebar.slider(
        "Hazard Weight",
        min_value=0.1,
        max_value=50.0,
        value=swarm.config.hazard_weight if swarm else 10.0,
        step=0.5
    )
    
    # Simulation speed
    sim_speed = st.sidebar.slider(
        "Simulation Speed",
        min_value=0,
        max_value=100,
        value=60,
        step=10
    )
    
    # Parameter optimization
    if st.sidebar.button("Run Parameter Optimization"):
        with st.spinner("Running optimization..."):
            from optimizer import coordinate_search_optimize
            coord_config = Config(
                seed=swarm.config.seed if swarm else 42,
                congestion_weight=congestion_weight,
                hazard_weight=hazard_weight,
                # Keep other parameters
                **{k: v for k, v in vars(swarm.config).items() 
                   if k not in ['congestion_weight', 'hazard_weight']}
            )
            results = coordinate_search_optimize(scenario_name, coord_config)
            st.toast("Optimization complete")
    
    # Perturbation controls
    st.sidebar.subheader("Perturbation Controls")
    
    col_a, col_b = st.sidebar.columns(2)
    with col_a:
        if st.button("ADD OBSTACLE"):
            handle_perturbation("ADD OBSTACLE", swarm)
            events.append({
                'tick': swarm.tick,
                'type': 'OBSTACLE_ADDED',
                'description': 'New obstacle added interactively'
            })
    
    with col_b:
        if st.button("FAIL AGENT"):
            handle_perturbation("FAIL AGENT", swarm)
            events.append({
                'tick': swarm.tick,
                'type': 'AGENT_FAILED',
                'description': 'Agent failed interactively'
            })
    
    # Start/Pause/Reset buttons
    col_start, col_pause, col_reset = st.sidebar.columns(3)
    
    with col_start:
        if st.button("START") and not swarm.running:
            swarm.running = True
            swarm.paused = False
            st.toast("Simulation started")
    
    with col_pause:
        if st.button("PAUSE") and swarm.running:
            swarm.paused = True
            st.toast("Simulation paused")
    
    with col_reset:
        if st.button("RESET"):
            # Reset everything
            from config import DEFAULT_CONFIG
            from environment import Environment
            from swarm import Swarm
            
            env = Environment(DEFAULT_CONFIG)
            env.create_environment()
            
            scenario = create_scenario(scenario_name, DEFAULT_CONFIG)
            swarm = Swarm(DEFAULT_CONFIG, env)
            swarm.initialize(agent_count)
            
            metrics = MetricsCollector(DEFAULT_CONFIG)
            metrics.tick_latencies = []
            metrics.tick_count = 0
            metrics.start_time = __import__('time').perf_counter()
            
            events.clear()
            swarm.running = False
            swarm.paused = False
            swarm.tick = 0
            
            st.session_state.swarm = swarm
            st.session_state.metrics = metrics
            st.session_state.events = events
            st.session_state.tick = 0
            st.session_state.current_scenario = scenario
            st.toast("Simulation reset")
    
    # Apply communication loss if configured
    if swarm and comm_loss != swarm.config.communication_loss_rate:
        swarm.config.communication_loss_rate = comm_loss
        for agent in swarm.agent_manager.agents.values():
            agent.communication_available = False  # Simplified


def main() -> None:
    """Main application entry point"""
    setup_page_config()
    init_session_state()
    
    # Header
    st.title("🤖 SWARM-X: Adaptive Stigmergic Flow-Field Navigation")
    st.caption("Decentralized Adaptive Flow-Field Swarm Coordination Engine")
    
    # Initialize session state if needed
    if not st.session_state.initialized:
        init_session_state()
    
    swarm = st.session_state.swarm
    metrics = st.session_state.metrics
    events = st.session_state.events
    viz = Visualization(DEFAULT_CONFIG if not swarm else swarm.config)
    
    # Render KPI bar (full width at top)
    if swarm:
        render_kpi_bar(swarm, metrics)
    
    # Main content area
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("Simulation Map")
        if swarm:
            fig = render_simulation_map(viz, swarm)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No simulation running. Configure parameters and click START.")
    
    with col_right:
        st.subheader("Event Log")
        render_event_log(events)
        
        st.subheader("Convergence")
        render_convergence_chart(metrics.tick_latencies if hasattr(metrics, 'tick_latencies') else [])
    
    # Control panel
    render_control_panel(swarm, metrics, events, viz)
    
    # Run simulation if active
    if swarm and swarm.running and not swarm.paused:
        # Run multiple ticks per render cycle based on speed
        speed = st.session_state.get('sim_speed', 60)
        ticks_to_run = max(1, min(5, speed // 10))
        
        for _ in range(ticks_to_run):
            if not swarm.paused and swarm.running:
                run_tick(swarm, metrics, events)
                if swarm.is_complete():
                    break
    
    # Auto-rerun
    time_sleep = max(0.01, 1.0 / st.session_state.get('sim_speed', 60))
    # Don't auto-render continuously - let user control
    # The simulation runs on button press


if __name__ == "__main__":
    main()