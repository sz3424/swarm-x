# SWARM-X: Adaptive Stigmergic Flow-Field Swarm Coordination Engine

A decentralized autonomous swarm coordination system for dynamic environments with obstacles, communication failures, and agent failures.

## Problem

Multi-agent systems must navigate shared environments with:
- Dynamic obstacles
- Resource contention
- Communication failures  
- Agent failures

The challenge is maintaining collective throughput, coverage, and safety while adapting to changes without a central controller.

## Architecture

```
swarm_x/
├── app.py              # Streamlit dashboard
├── config.py           # Configuration parameters
├── models.py           # Domain models (Agent, Task, Obstacle, Hazard)
├── environment.py      # Grid world with obstacles/hazards
├── flow_field.py       # Dijkstra-based navigation field
├── agent.py            # Agent decision-making
├── swarm.py            # Swarm coordination
├── safety.py           # Hard constraint validation
├── communication.py     # Simulated P2P messaging
├── metrics.py          # Performance measurement
├── scenarios.py        # Benchmark scenarios
├── benchmark.py        # Evaluation harness
├── optimizer.py        # Parameter optimization
└── visualization.py    # Plotly visualizations
```

## Why Flow Field

Flow fields pre-compute cost-to-go information for all cells, enabling any agent to obtain a velocity direction without running its own pathfinder. This allows thousands of agents to share the same navigation field.

## Why Dijkstra Instead of Independent A*

Running individual A* searches for each agent scales poorly. Dijkstra computes a single distance field from the goal(s) that all agents can share. This reduces computation from O(n_agents × n_cells) to O(n_cells) for field generation.

## Stigmergic Congestion Mechanism

When an agent enters a cell:
1. Cell congestion increases by `CONGESTION_DEPOSIT`
2. Congestion decays by `CONGESTION_DECAY` each tick
3. High congestion increases traversal cost

This creates indirect swarm coordination through the environment itself, without explicit traffic management.

## Decentralization

- Agents maintain local state only
- Decisions based on local observations (sensing radius)
- Peer-to-peer communication for coordination
- Task allocation via local bidding

## Safety Layer

Safety is a **hard constraint**, not an optimization penalty:
- `MIN_AGENT_DISTANCE = 1.5` units between agents
- `OBSTACLE_MARGIN = 1.0` units from obstacles
- Movements violating these bounds are rejected

## Communication Failure Handling

When communication fails:
- Agent switches to local mode
- Uses cached neighbor positions
- Continues navigating using local flow field
- No central server required

## Agent Failure Recovery

Failed agents:
- Stop moving
- Release their tasks (task becomes UNASSIGNED)
- Agents nearby detect and rebid on released tasks

Metrics tracked:
- `failure_count`
- `tasks_recovered`
- `recovery_time`

## Deadlock Detection

Each agent tracks `stalled_ticks`:
- Incremented when no progress is made
- If `stalled_ticks >= DEADLOCK_THRESHOLD`, deadlock is declared
- Recovery: increase congestion penalty, select alternate direction

## Fitness Function

```
fitness = 0.30 × throughput_score
        + 0.20 × coverage_score
        + 0.15 × task_completion_score
        + 0.10 × resilience_score
        - 0.10 × energy_score
        - 0.05 × path_length_score
        - 0.10 × deadlock_score
```

**Safety is a gate**: If `collision_count > 0`, fitness = 0 and safety_status = "SAFETY_VIOLATION"

## Benchmark Scenarios

| Scenario | Agents | Dynamic Obstacles | Comm Loss | Failures | Purpose |
|----------|--------|-------------------|-----------|----------|---------|
| Round 1 | 25-50 | No | 0% | No | Baseline |
| Round 2 | 25-50 | Yes | 10-30% | No | Adaptation |
| Final | 50-75 | Yes | 20-30% | Yes | Stress test |

## How to Run

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run benchmarks
```bash
python -m benchmark --scenario round1 --seed 42
python -m benchmark --scenario round2 --seed 42
python -m benchmark --scenario final --seed 42
```

### Run parameter optimization
```bash
python -m optimizer --scenario round2 --seed 42
```

### Launch dashboard
```bash
streamlit run app.py
```

### Run tests
```bash
pytest tests/
```

## Results

Generated in `results/`:
- `round1_metrics.json`, `round2_metrics.json`, `final_metrics.json`
- `convergence.csv` (optimization trajectory)
- `benchmark_summary.json`

Each JSON includes: scenario, seed, configuration, metrics, fitness, safety_status, runtime, timestamp

## Limitations

1. 2D grid only (no 3D navigation)
2. Discrete position representation
3. Simple energy model (no realistic battery physics)
4. Limited communication model (no real networking)
5. Static dynamic obstacles (bounded velocity)

## Future Improvements

1. Anisotropic cost fields (different costs per axis)
2. Multi-goal assignment
3. Predictive obstacle avoidance
4. Hierarchical flow fields
5. Real-time replanning triggers
6. Advanced stigmergic mechanisms