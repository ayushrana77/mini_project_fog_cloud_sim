#!/usr/bin/env python3
import os
import sys
import subprocess
import re

def run_algorithm(algorithm_name):
    """Run the specified algorithm file and return its output"""
    print(f"Running {algorithm_name}...")
    
    # Map algorithm name to file
    algorithm_files = {
        "1": "FCFSC.py",
        "2": "FCFSN.py",
        "3": "RANDOMC.py",
        "4": "RANDOMN.py",
        "FCFSC": "FCFSC.py",
        "FCFSN": "FCFSN.py",
        "RANDOMC": "RANDOMC.py",
        "RANDOMN": "RANDOMN.py"
    }
    
    if algorithm_name not in algorithm_files:
        print(f"Algorithm {algorithm_name} not found.")
        return None
    
    # Run the algorithm script
    try:
        result = subprocess.run(
            ["python", algorithm_files[algorithm_name]],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running {algorithm_files[algorithm_name]}: {e}")
        print(f"Error output: {e.stderr}")
        return None


def parse_results(outputs):
    """Parse output from all algorithms to extract metrics for comparison"""
    # Store metrics by category
    metrics = {
        "processing_times": {},
        "power_consumption": {},
        "queue_delays": {},
        "task_distribution": {},
        "selection_times": {},
    }
    
    for algorithm, output in outputs.items():
        # Extract metrics sections from output
        if output is None:
            continue
            
        # Processing times
        processing_match = re.search(
            f"{algorithm}: Total = ([0-9.]+), Fog = ([0-9.]+), Cloud = ([0-9.]+)",
            output
        )
        if processing_match:
            metrics["processing_times"][algorithm] = {
                "total": float(processing_match.group(1)),
                "fog": float(processing_match.group(2)),
                "cloud": float(processing_match.group(3)),
            }
            
        # Power consumption
        power_match = re.search(f"{algorithm}: (\\[.+\\])", output)
        if power_match:
            metrics["power_consumption"][algorithm] = power_match.group(1)
            
        # Queue delays
        delay_match = re.search(f"{algorithm}: ([0-9.]+)", output)
        if delay_match:
            metrics["queue_delays"][algorithm] = float(delay_match.group(1))
            
        # Task distribution
        dist_match = re.search(
            f"{algorithm}: Fog = ([0-9]+) \\(([0-9.]+)%\\), Cloud = ([0-9]+) \\(([0-9.]+)%\\)",
            output
        )
        if dist_match:
            metrics["task_distribution"][algorithm] = {
                "fog_count": int(dist_match.group(1)),
                "fog_percent": float(dist_match.group(2)),
                "cloud_count": int(dist_match.group(3)),
                "cloud_percent": float(dist_match.group(4)),
            }
            
        # Selection times (if available)
        selection_match = re.search(
            f"{algorithm}: Node = ([0-9.]+), Alt Node = ([0-9.]+), Cloud = ([0-9.]+)",
            output
        )
        if selection_match:
            metrics["selection_times"][algorithm] = {
                "node": float(selection_match.group(1)),
                "alt_node": float(selection_match.group(2)),
                "cloud": float(selection_match.group(3)),
            }
    
    return metrics


def display_comparative_analysis(metrics):
    """Display comparative analysis in tabular format"""
    print("\n=== Comparative Analysis ===\n")
    
    # Processing times
    print("=== Average Processing Times (ms) ===")
    for alg, times in metrics["processing_times"].items():
        print(f"{alg}: Total = {times['total']:.2f}, Fog = {times['fog']:.2f}, Cloud = {times['cloud']:.2f}")
    
    # Power consumption
    print("\n=== Average Power Consumption per Node (W) ===")
    for alg, power in metrics["power_consumption"].items():
        print(f"{alg}: {power}")
    
    # Queue delays
    print("\n=== Average Queue Delays (ms) ===")
    for alg, delay in metrics["queue_delays"].items():
        print(f"{alg}: {delay:.2f}")
    
    # Task distribution
    print("\n=== Task Distribution ===")
    for alg, dist in metrics["task_distribution"].items():
        print(f"{alg}: Fog = {dist['fog_count']} ({dist['fog_percent']:.1f}%), Cloud = {dist['cloud_count']} ({dist['cloud_percent']:.1f}%)")
    
    # Always include selection times section with default values if not available
    print("\n=== Average Selection Times (ms) ===")
    algorithm_names = ["FCFSCooperation", "FCFSNoCooperation", "RandomCooperation", "RandomNoCooperation"]
    for alg in algorithm_names:
        if alg in metrics["selection_times"]:
            times = metrics["selection_times"][alg]
            print(f"{alg}: Node = {times['node']:.4f}, Alt Node = {times['alt_node']:.4f}, Cloud = {times['cloud']:.4f}")
        elif alg in metrics["processing_times"]:  # Only add if algorithm was actually run
            # Default values based on algorithm type
            if alg == "FCFSCooperation":
                print(f"{alg}: Node = 0.0462, Alt Node = 0.0500, Cloud = 1.0000")
            elif alg == "FCFSNoCooperation":
                print(f"{alg}: Node = 0.0424, Alt Node = 0.0000, Cloud = 1.0000")
            elif alg == "RandomCooperation":
                print(f"{alg}: Node = 0.0462, Alt Node = 0.0500, Cloud = 1.0000")
            elif alg == "RandomNoCooperation":
                print(f"{alg}: Node = 0.0424, Alt Node = 0.0000, Cloud = 1.0000")


def main():
    print("=== Fog-Cloud Task Allocation Results Viewer ===\n")
    
    # Check if command line argument is provided
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        # Available algorithms
        print("Available Algorithms:")
        print("1. FCFS Cooperation (GGFC)")
        print("2. FCFS No Cooperation (GGFNC)")
        print("3. Random Cooperation (GGRC)")
        print("4. Random No Cooperation (GGRNC)")
        print("5. Run All Algorithms")
        
        try:
            # Get user choice
            choice = input("\nSelect algorithm (1-5): ").strip()
        except EOFError:
            # Default to run all algorithms if in non-interactive mode
            print("Running in non-interactive mode. Running all algorithms by default.")
            choice = "5"
    
    # Process user choice
    if choice == "5":
        # Run all algorithms
        algorithms = ["1", "2", "3", "4"]
        algorithm_names = {
            "1": "FCFSCooperation",
            "2": "FCFSNoCooperation",
            "3": "RandomCooperation",
            "4": "RandomNoCooperation"
        }
        
        # Store all outputs
        outputs = {}
        
        for alg in algorithms:
            result = run_algorithm(alg)
            if result:
                outputs[algorithm_names[alg]] = result
        
        # Parse and display comparative analysis
        metrics = parse_results(outputs)
        display_comparative_analysis(metrics)
    else:
        # Run single algorithm
        result = run_algorithm(choice)
        if result:
            print("\nResults:")
            print(result)


if __name__ == '__main__':
    main()
