from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, List

import numpy as np

from config import Config
from environment import Environment
from swarm import Swarm
from metrics import MetricsCollector
from scenarios import create_scenario


def run_benchmark(scenario_name: str, seed: int = 42) -> Dict[str, Any]:
    """Run a benchmark scenario and collect metrics"""
    
    # Load configuration
    config = Config(seed=seed)
    
    # Create environment
    env = Environment(config)
    env.create_environment()
    
    # Create scenario
    scenario = create_scenario(scenario_name, config)
    
    # Create swarm
    swarm = Swarm(config, env)
    swarm.current_scenario = scenario
    swarm.initialize(scenario.agents_count)
    
    # Initialize metrics collector
    metrics_collector = MetricsCollector(config)
    metrics_collector.start_time = time.perf_counter()
    
    # Run simulation
    start_time = time.perf_counter()
    while True:
        # Perform one tick
        metrics_collector.start_tick()
        result = swarm.step()
        metrics_collector.end_tick()
        
        # Stop if all tasks completed
        if result["completed_tasks"] >= scenario.agents_count:
            break
            
        # Small delay to prevent excessive CPU usage
        time.sleep(0.001)
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    # Record final metrics
    metrics_collector.record_metrics(swarm)
    
    # Generate results
    results = {
        "scenario": scenario_name,
        "seed": seed,
        "configuration": {
            "grid_width": config.grid_width,
            "grid_height": config.grid_height,
            "max_ticks": config.max_ticks,
            "base_cost": config.base_cost,
            "congestion_weight": config.congestion_weight,
            "visit_weight": config.visit_weight,
            "hazard_weight": config.hazard_weight,
            "energy_weight": config.energy_weight,
            "min_agent_distance": config.min_agent_distance,
            "obstacle_margin": config.obstacle_margin,
            "deadlock_threshold": config.deadlock_threshold,
            "agent_speed": config.agent_speed,
            "max_battery": config.max_battery,
            "movement_cost": config.movement_cost,
            "communication_loss_rate": config.communication_loss_rate,
            "fitness_throughput_weight": config.fitness_throughput_weight,
            "fitness_coverage_weight": config.fitness_coverage_weight,
            "fitness_task_completion_weight": config.fitness_task_completion_weight,
            "fitness_resilience_weight": config.fitness_resilience_weight,
            "optimization_max_iterations": config.optimization_max_iterations,
        },
        "metrics": metrics_collector.to_json(),
        "runtime_seconds": total_time,
        "throughput": metrics_collector.metrics.throughput,
        "completion_rate": metrics_collector.metrics.task_completion_rate,
        "coverage": metrics_collector.metrics.coverage,
        "energy_consumed": metrics_collector.metrics.energy_consumed,
        "collision_count": metrics_collector.metrics.collision_count,
        "deadlock_count": metrics_collector.metrics.deadlock_count,
        "failure_count": metrics_collector.metrics.failed_agents,
        "recovery_time": metrics_collector.metrics.recovery_time,
        "fitness": metrics_collector.metrics.fitness,
        "safety_status": metrics_collector.metrics.safety_status,
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run SWARM-X benchmark")
    parser.add_argument("--scenario", type=str, required=True, 
                        choices=["round1", "round2", "final"],
                        help="Benchmark scenario to run")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible results")
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print(f"Running {args.scenario} benchmark with seed {args.seed}...")
    print(f"Results will be saved to: {args.output}")
    
    # Run benchmark
    results = run_benchmark(args.scenario, args.seed)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save individual scenario results
    scenario_file = os.path.join(args.output, f"{args.scenario}_metrics.json")
    with open(scenario_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved results to: {scenario_file}")
    
    # Save summary if first run
    summary_file = os.path.join(args.output, "benchmark_summary.json")
    if not os.path.exists(summary_file):
        summary = {
            "timestamp": timestamp,
            "scenarios": [],
            "configuration": results["configuration"]
        }
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
    
    # Update summary
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    summary["scenarios"].append({
        "timestamp": timestamp,
        "scenario": args.scenario,
        "seed": args.seed,
        "results": results
    })
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Updated summary at: {summary_file}")
    
    # Print summary
    print("\n=== BENCHMARK RESULTS ===")
    print(f"Scenario: {results['scenario']}")
    print(f"Seed: {results['seed']}")
    print(f"Fitness: {results['fitness']:.4f}")
    print(f"Safety Status: {results['safety_status']}")
    print(f"Throughput: {results['throughput']:.2f} tasks/tick")
    print(f"Coverage: {results['coverage']:.2%}")
    print(f"Energy Consumed: {results['energy_consumed']:.2f}")
    print(f"Collisions: {results['collision_count']}")
    print(f"Deadlocks: {results['deadlock_count']}")
    print(f"Failures: {results['failure_count']}")
    print(f"Runtime: {results['runtime_seconds']:.2f} seconds")


if __name__ == "__main__":
    main()