import argparse
import os
import sys
import csv
from typing import Dict, List
from config import Config


def run_benchmark_for_config(config: Config, scenario_name: str, seed: int) -> Dict:
    """Run benchmark with specific configuration"""
    from benchmark import run_benchmark
    
    # Use the provided config's seed but allow override
    test_seed = seed
    try:
        results = run_benchmark(scenario_name, test_seed)
        return results
    except Exception as e:
        print(f"Error running benchmark: {e}")
        return {
            "fitness": 0.0,
            "throughput": 0.0,
            "coverage": 0.0,
            "energy_consumed": 0.0,
            "deadlock_count": 0,
            "mean_tick_latency_ms": 0.0,
            "safety_status": "ERROR"
        }


def coordinate_search_optimize(scenario_name: str, config: Config) -> None:
    """Perform coordinate search optimization on key parameters"""
    
    # Parameters to optimize with their bounds
    param_bounds = {
        "congestion_weight": (0.5, 5.0),
        "visit_weight": (0.1, 3.0),
        "hazard_weight": (1.0, 20.0),
        "energy_weight": (0.1, 2.0),
        "deadlock_threshold": (5, 20),
        "congestion_decay": (0.8, 0.99),
        "congestion_deposit": (0.1, 2.0)
    }
    
    # Ensure results directory exists
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    convergence_file = os.path.join(results_dir, "convergence.csv")
    
    # Initialize CSV file
    with open(convergence_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "iteration", "parameter_changed", "parameter_value", 
            "fitness", "throughput", "coverage", "energy", 
            "deadlocks", "latency", "safety_status"
        ])
    
    # Start with baseline parameters
    best_config = Config(
        seed=config.seed,
        congestion_weight=config.congestion_weight,
        visit_weight=config.visit_weight,
        hazard_weight=config.hazard_weight,
        energy_weight=config.energy_weight,
        deadlock_threshold=config.deadlock_threshold,
        congestion_decay=config.congestion_decay,
        congestion_deposit=config.congestion_deposit
    )
    best_fitness = -float('inf')
    iteration = 0
    
    print(f"Starting optimization for scenario: {scenario_name}")
    print(f"Baseline fitness: {best_fitness}")
    
    # For each parameter, try different values
    for param_name, (min_val, max_val) in param_bounds.items():
        print(f"\nOptimizing parameter: {param_name} [{min_val}, {max_val}]")
        
        # Try 5 different values for this parameter
        if param_name in ["deadlock_threshold"]:
            values = [int(min_val + i * (max_val - min_val) / 4) for i in range(5)]
        else:
            values = [min_val + i * (max_val - min_val) / 4 for i in range(5)]
        
        for value in values:
            iteration += 1
            
            # Create test config with this parameter value
            test_config = Config(
                seed=config.seed,
                congestion_weight=config.congestion_weight,
                visit_weight=config.visit_weight,
                hazard_weight=config.hazard_weight,
                energy_weight=config.energy_weight,
                deadlock_threshold=config.deadlock_threshold,
                congestion_decay=config.congestion_decay,
                congestion_deposit=config.congestion_deposit
            )
            
            # Set the parameter being optimized
            setattr(test_config, param_name, value)
            
            # Run benchmark
            print(f"  Iteration {iteration}: Testing {param_name} = {value:.3f}")
            try:
                results = run_benchmark_for_config(test_config, scenario_name, config.seed)
                fitness = results.get("fitness", 0.0)
                
                print(f"    Fitness: {fitness:.4f}")
                
                # Check if this is better
                if fitness > best_fitness:
                    print(f"    Improvement! {fitness:.4f} > {best_fitness:.4f}")
                    best_fitness = fitness
                    best_config = test_config
                    improved = True
                else:
                    improved = False
                    
                # Save to convergence file
                with open(convergence_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        iteration,
                        param_name,
                        value,
                        fitness,
                        results.get("throughput", 0.0),
                        results.get("coverage", 0.0),
                        results.get("energy_consumed", 0.0),
                        results.get("deadlock_count", 0),
                        results.get("mean_tick_latency_ms", 0.0),
                        results.get("safety_status", "UNKNOWN")
                    ])
                    
            except Exception as e:
                print(f"    Error running benchmark: {e}")
                # Save error result
                with open(convergence_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        iteration,
                        param_name,
                        value,
                        0.0,  # fitness
                        0.0,  # throughput
                        0.0,  # coverage
                        0.0,  # energy
                        0,    # deadlocks
                        0.0,  # latency
                        "ERROR"  # safety_status
                    ])
    
    # Save best configuration
    best_config_file = os.path.join(results_dir, f"best_config_{scenario_name}.py")
    with open(best_config_file, "w") as f:
        f.write(f"# Best configuration for {scenario_name}\n")
        f.write(f"# Fitness: {best_fitness}\n")
        f.write("from config import Config\n\n")
        f.write(f"BEST_CONFIG = Config(\n")
        for field_name in vars(best_config):
            if not field_name.startswith("_"):
                value = getattr(best_config, field_name)
                if isinstance(value, str):
                    f.write(f"    {field_name}='{value}',\n")
                else:
                    f.write(f"    {field_name}={value},\n")
        f.write(")\n")
    
    print(f"\nOptimization complete!")
    print(f"Best fitness: {best_fitness:.4f}")
    print(f"Results saved to: {convergence_file}")
    print(f"Best configuration saved to: {best_config_file}")


def main():
    parser = argparse.ArgumentParser(description="Run SWARM-X parameter optimization")
    parser.add_argument("--scenario", type=str, required=True, choices=["round1", "round2", "final"])
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    config = Config(seed=args.seed)
    coordinate_search_optimize(args.scenario, config)


if __name__ == "__main__":
    main()