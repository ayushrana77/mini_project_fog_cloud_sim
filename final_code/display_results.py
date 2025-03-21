#!/usr/bin/env python3
import os
import sys
import subprocess
import re
import matplotlib.pyplot as plt
import numpy as np

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
            # Extract the power consumption string and convert to list of floats
            power_str = power_match.group(1)
            # Remove brackets and split by comma
            power_values = [float(val.strip().strip("'")) for val in power_str.strip('[]').split(',')]
            metrics["power_consumption"][algorithm] = power_values
            
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
    # Processing Times
    print("\n=== Average Processing Times (ms) ===")
    for algorithm, times in metrics["processing_times"].items():
        print(f"{algorithm}: Total = {times['total']:.2f}, Fog = {times['fog']:.2f}, Cloud = {times['cloud']:.2f}")
    
    # Power Consumption
    print("\n=== Average Power Consumption per Node (W) ===")
    for algorithm, power in metrics["power_consumption"].items():
        power_str = str(power)
        print(f"{algorithm}: {power_str}")
    
    # Queue Delays
    print("\n=== Average Queue Delays (ms) ===")
    for algorithm, delay in metrics["queue_delays"].items():
        print(f"{algorithm}: {delay:.2f}")
    
    # Task Distribution
    print("\n=== Task Distribution ===")
    for algorithm, dist in metrics["task_distribution"].items():
        print(f"{algorithm}: Fog = {dist['fog_count']} ({dist['fog_percent']:.1f}%), Cloud = {dist['cloud_count']} ({dist['cloud_percent']:.1f}%)")
    
    # Selection Times
    print("\n=== Average Selection Times (ms) ===")
    for algorithm, times in metrics["selection_times"].items():
        print(f"{algorithm}: Node = {times['node']:.4f}, Alt Node = {times['alt_node']:.4f}, Cloud = {times['cloud']:.4f}")


def plot_processing_times(metrics):
    """Plot processing times comparison"""
    algorithms = list(metrics["processing_times"].keys())
    
    # Extract data
    total_times = [metrics["processing_times"][alg]["total"] for alg in algorithms]
    fog_times = [metrics["processing_times"][alg]["fog"] for alg in algorithms]
    cloud_times = [metrics["processing_times"][alg]["cloud"] for alg in algorithms]
    
    # Create labels with algorithm names
    labels = []
    for alg in algorithms:
        if alg == "FCFSCooperation":
            labels.append("FCFS-C")
        elif alg == "FCFSNoCooperation":
            labels.append("FCFS-NC")
        elif alg == "RandomCooperation":
            labels.append("Random-C")
        elif alg == "RandomNoCooperation":
            labels.append("Random-NC")
        else:
            labels.append(alg)
    
    # Set width of bars
    barWidth = 0.25
    
    # Set positions of bar on X axis
    r1 = np.arange(len(labels))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.bar(r1, total_times, width=barWidth, edgecolor='grey', label='Total')
    plt.bar(r2, fog_times, width=barWidth, edgecolor='grey', label='Fog')
    plt.bar(r3, cloud_times, width=barWidth, edgecolor='grey', label='Cloud')
    
    # Add labels
    plt.xlabel('Algorithm', fontweight='bold')
    plt.ylabel('Processing Time (ms)', fontweight='bold')
    plt.title('Average Processing Times by Algorithm')
    plt.xticks([r + barWidth for r in range(len(labels))], labels)
    plt.legend()
    
    plt.savefig('processing_times.png')
    print("Processing times plot saved as 'processing_times.png'")


def plot_task_distribution(metrics):
    """Plot task distribution between fog and cloud"""
    algorithms = list(metrics["task_distribution"].keys())
    
    # Extract data
    fog_percents = [metrics["task_distribution"][alg]["fog_percent"] for alg in algorithms]
    cloud_percents = [metrics["task_distribution"][alg]["cloud_percent"] for alg in algorithms]
    
    # Create labels with algorithm names
    labels = []
    for alg in algorithms:
        if alg == "FCFSCooperation":
            labels.append("FCFS-C")
        elif alg == "FCFSNoCooperation":
            labels.append("FCFS-NC")
        elif alg == "RandomCooperation":
            labels.append("Random-C")
        elif alg == "RandomNoCooperation":
            labels.append("Random-NC")
        else:
            labels.append(alg)
    
    # Create stacked bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(labels, fog_percents, label='Fog')
    plt.bar(labels, cloud_percents, bottom=fog_percents, label='Cloud')
    
    plt.xlabel('Algorithm', fontweight='bold')
    plt.ylabel('Percentage of Tasks (%)', fontweight='bold')
    plt.title('Task Distribution Between Fog and Cloud')
    plt.legend()
    
    plt.savefig('task_distribution.png')
    print("Task distribution plot saved as 'task_distribution.png'")


def plot_queue_delays(metrics):
    """Plot queue delays comparison"""
    algorithms = list(metrics["queue_delays"].keys())
    
    # Extract data
    delays = [metrics["queue_delays"][alg] for alg in algorithms]
    
    # Create labels with algorithm names
    labels = []
    for alg in algorithms:
        if alg == "FCFSCooperation":
            labels.append("FCFS-C")
        elif alg == "FCFSNoCooperation":
            labels.append("FCFS-NC")
        elif alg == "RandomCooperation":
            labels.append("Random-C")
        elif alg == "RandomNoCooperation":
            labels.append("Random-NC")
        else:
            labels.append(alg)
    
    # Create bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(labels, delays)
    
    plt.xlabel('Algorithm', fontweight='bold')
    plt.ylabel('Queue Delay (ms)', fontweight='bold')
    plt.title('Average Queue Delays by Algorithm')
    
    plt.savefig('queue_delays.png')
    print("Queue delays plot saved as 'queue_delays.png'")


def plot_total_time_comparison(metrics):
    """Plot total processing time comparison"""
    algorithms = list(metrics["processing_times"].keys())
    
    # Extract data
    total_times = [metrics["processing_times"][alg]["total"] for alg in algorithms]
    
    # Create labels with algorithm names
    labels = []
    for alg in algorithms:
        if alg == "FCFSCooperation":
            labels.append("FCFS-C")
        elif alg == "FCFSNoCooperation":
            labels.append("FCFS-NC")
        elif alg == "RandomCooperation":
            labels.append("Random-C")
        elif alg == "RandomNoCooperation":
            labels.append("Random-NC")
        else:
            labels.append(alg)
    
    # Create bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, total_times)
    
    # Add values on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    plt.xlabel('Algorithm', fontweight='bold')
    plt.ylabel('Total Processing Time (ms)', fontweight='bold')
    plt.title('Total Processing Time Comparison')
    
    plt.savefig('total_time_comparison.png')
    print("Total time comparison plot saved as 'total_time_comparison.png'")


def plot_power_consumption(metrics):
    """Plot power consumption comparison"""
    algorithms = list(metrics["power_consumption"].keys())
    
    # Create labels with algorithm names
    labels = []
    for alg in algorithms:
        if alg == "FCFSCooperation":
            labels.append("FCFS-C")
        elif alg == "FCFSNoCooperation":
            labels.append("FCFS-NC")
        elif alg == "RandomCooperation":
            labels.append("Random-C")
        elif alg == "RandomNoCooperation":
            labels.append("Random-NC")
        else:
            labels.append(alg)
    
    # Calculate average power per algorithm
    avg_power = []
    for alg in algorithms:
        power_values = metrics["power_consumption"][alg]
        avg = sum(power_values) / len(power_values) if power_values else 0
        avg_power.append(avg)
    
    # Create bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, avg_power)
    
    # Add values on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    plt.xlabel('Algorithm', fontweight='bold')
    plt.ylabel('Average Power Consumption (W)', fontweight='bold')
    plt.title('Power Consumption Comparison')
    
    plt.savefig('power_consumption.png')
    print("Power consumption plot saved as 'power_consumption.png'")


def plot_total_power_consumption(metrics):
    """Plot total power consumption (power × nodes)"""
    algorithms = list(metrics["power_consumption"].keys())
    
    # Create labels with algorithm names
    labels = []
    for alg in algorithms:
        if alg == "FCFSCooperation":
            labels.append("FCFS-C")
        elif alg == "FCFSNoCooperation":
            labels.append("FCFS-NC")
        elif alg == "RandomCooperation":
            labels.append("Random-C")
        elif alg == "RandomNoCooperation":
            labels.append("Random-NC")
        else:
            labels.append(alg)
    
    # Calculate total power per algorithm
    total_power = []
    for alg in algorithms:
        power_values = metrics["power_consumption"][alg]
        total = sum(power_values) if power_values else 0
        total_power.append(total)
    
    # Create bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, total_power)
    
    # Add values on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    plt.xlabel('Algorithm', fontweight='bold')
    plt.ylabel('Total Power Consumption (W)', fontweight='bold')
    plt.title('Total Power Consumption Across All Nodes')
    
    plt.savefig('total_power_consumption.png')
    print("Total power consumption plot saved as 'total_power_consumption.png'")


def generate_all_graphs(metrics):
    """Generate all available graphs from metrics"""
    if len(metrics["processing_times"]) > 0:
        plot_processing_times(metrics)
        plot_total_time_comparison(metrics)
    
    if len(metrics["task_distribution"]) > 0:
        plot_task_distribution(metrics)
    
    if len(metrics["queue_delays"]) > 0:
        plot_queue_delays(metrics)
    
    if len(metrics["power_consumption"]) > 0:
        plot_power_consumption(metrics)
        plot_total_power_consumption(metrics)


def main():
    # Parse command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python display_results.py <algorithm_number>")
        print("  1: FCFSCooperation")
        print("  2: FCFSNoCooperation")
        print("  3: RandomCooperation")
        print("  4: RandomNoCooperation")
        print("  5: All algorithms")
        print("  6: Generate graphs (will recompute all results)")
        sys.exit(1)
    
    algorithm_arg = sys.argv[1]
    
    # Map numeric inputs to algorithm names
    algorithm_names = {
        "1": "FCFSCooperation",
        "2": "FCFSNoCooperation",
        "3": "RandomCooperation",
        "4": "RandomNoCooperation"
    }
    
    # Print header
    print("=== Fog-Cloud Task Allocation Results Viewer ===\n")
    
    outputs = {}
    
    # Option 6 now recomputes all results instead of using cached files
    if algorithm_arg == "6":
        print("Recomputing all results...")
        for num, name in algorithm_names.items():
            outputs[name] = run_algorithm(num)
            
        # Parse results and generate graphs
        metrics = parse_results(outputs)
        generate_all_graphs(metrics)
        sys.exit(0)
    
    # Run all algorithms or a specific one
    if algorithm_arg == "5":
        for num, name in algorithm_names.items():
            outputs[name] = run_algorithm(num)
    else:
        # Run a specific algorithm
        if algorithm_arg in algorithm_names:
            name = algorithm_names[algorithm_arg]
            outputs[name] = run_algorithm(algorithm_arg)
        else:
            print(f"Unknown algorithm: {algorithm_arg}")
            sys.exit(1)
    
    # Parse the results
    metrics = parse_results(outputs)
    
    # Display comparative analysis
    print("\n=== Comparative Analysis ===")
    display_comparative_analysis(metrics)
    
    # Generate graphs
    if algorithm_arg == "5":
        generate_all_graphs(metrics)


if __name__ == '__main__':
    main()
