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
        "FCFSCooperation": "FCFSC.py",
        "FCFSNoCooperation": "FCFSN.py",
        "RandomCooperation": "RANDOMC.py",
        "RandomNoCooperation": "RANDOMN.py"
    }
    
    if algorithm_name not in algorithm_files:
        print(f"Algorithm {algorithm_name} not found.")
        return None
    
    # Run the algorithm script
    try:
        algorithm_file = algorithm_files[algorithm_name]
        algorithm_path = os.path.join(os.path.dirname(__file__), algorithm_file)
        print(f"Executing: {algorithm_path}")
        
        result = subprocess.run(
            ["python", algorithm_path],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running {algorithm_files[algorithm_name]}: {e}")
        print(f"Error output: {e.stderr}")
        print(f"Error code: {e.returncode}")
        print(f"Command: {e.cmd}")
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


def save_to_final_code_folder(filename):
    """Return a full path to save in the final_code folder"""
    final_code_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(final_code_dir, filename)


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
    
    # Add value labels on top of bars
    for bars in [plt.bar(r1, total_times, width=barWidth, edgecolor='grey', label='Total'), 
                 plt.bar(r2, fog_times, width=barWidth, edgecolor='grey', label='Fog'), 
                 plt.bar(r3, cloud_times, width=barWidth, edgecolor='grey', label='Cloud')]:
        for bar in bars:
            height = bar.get_height()
            plt.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(15, 0),  # 15 points horizontal offset (to the right)
                        textcoords="offset points",
                        ha='left', va='center', fontsize=9)
            
    # Add labels
    plt.xlabel('Algorithm', fontweight='bold')
    plt.ylabel('Processing Time (ms)', fontweight='bold')
    plt.title('Average Processing Times by Algorithm')
    plt.xticks([r + barWidth for r in range(len(labels))], labels)
    plt.legend()
    
    save_path = save_to_final_code_folder('processing_times.png')
    plt.savefig(save_path)
    print(f"Processing times plot saved as '{save_path}'")


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
    
    # Add value labels on top of bars
    for i, v in enumerate(fog_percents):
        plt.text(i, v + 0.1, f"{v:.1f}%", ha='center')
    for i, v in enumerate(cloud_percents):
        plt.text(i, fog_percents[i] + v + 0.1, f"{v:.1f}%", ha='center')
    
    plt.xlabel('Algorithm', fontweight='bold')
    plt.ylabel('Percentage of Tasks (%)', fontweight='bold')
    plt.title('Task Distribution between Fog and Cloud by Algorithm')
    plt.legend()
    
    save_path = save_to_final_code_folder('task_distribution.png')
    plt.savefig(save_path)
    print(f"Task distribution plot saved as '{save_path}'")


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
    
    # Add value labels on the side of bars
    for i, v in enumerate(delays):
        plt.text(i + 0.1, v, f"{v:.1f}", ha='left', va='center')
    
    plt.xlabel('Algorithm', fontweight='bold')
    plt.ylabel('Queue Delay (ms)', fontweight='bold')
    plt.title('Average Queue Delays by Algorithm')
    
    save_path = save_to_final_code_folder('queue_delays.png')
    plt.savefig(save_path)
    print(f"Queue delays plot saved as '{save_path}'")


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
    plt.bar(labels, total_times)
    
    # Add value labels on top of bars
    for i, v in enumerate(total_times):
        plt.text(i, v + 0.1, f"{v:.1f}", ha='center')
    
    plt.xlabel('Algorithm', fontweight='bold')
    plt.ylabel('Total Processing Time (ms)', fontweight='bold')
    plt.title('Total Processing Time Comparison')
    
    save_path = save_to_final_code_folder('total_time_comparison.png')
    plt.savefig(save_path)
    print(f"Total time comparison plot saved as '{save_path}'")


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
    
    # Create a single bar chart showing average power per algorithm
    plt.figure(figsize=(10, 6))
    
    for i, alg in enumerate(algorithms):
        power_values = metrics["power_consumption"][alg]
        # Plot each node as a separate bar with slight offset
        for j, power in enumerate(power_values):
            plt.bar(
                i + (j - len(power_values)/2 + 0.5) * 0.2,
                power,
                width=0.15,
                label=f'Node {j+1} ({alg})' if i == 0 else "_nolegend_"
            )
        
        # Add value labels on top of bars
        for j, power in enumerate(power_values):
            plt.text(i + (j - len(power_values)/2 + 0.5) * 0.2 + 0.1, power, f"{power:.1f}", ha='left', va='center')
    
    plt.xlabel('Algorithm', fontweight='bold')
    plt.ylabel('Power Consumption (W)', fontweight='bold')
    plt.title('Power Consumption by Node and Algorithm')
    plt.xticks(range(len(labels)), labels)
    
    # Add a legend but remove duplicates
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    save_path = save_to_final_code_folder('power_consumption.png')
    plt.savefig(save_path)
    print(f"Power consumption plot saved as '{save_path}'")


def plot_total_power_consumption(metrics):
    """Plot total power consumption for each algorithm"""
    algorithms = list(metrics["power_consumption"].keys())
    
    # Calculate total power consumption (sum of all nodes)
    total_power = []
    for alg in algorithms:
        total_power.append(sum(metrics["power_consumption"][alg]))
    
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
    plt.bar(labels, total_power)
    
    # Add value labels on top of bars
    for i, v in enumerate(total_power):
        plt.text(i, v + 0.1, f"{v:.1f}", ha='center')
    
    plt.xlabel('Algorithm', fontweight='bold')
    plt.ylabel('Total Power Consumption (W)', fontweight='bold')
    plt.title('Total Power Consumption Comparison')
    
    save_path = save_to_final_code_folder('total_power_consumption.png')
    plt.savefig(save_path)
    print(f"Total power consumption plot saved as '{save_path}'")


def plot_fcfs_comparison(metrics):
    """Plot comparison between FCFS with and without cooperation"""
    algorithms = ["FCFSCooperation", "FCFSNoCooperation"]
    
    # Check if the required algorithms are available in metrics
    available_algs = []
    for alg in algorithms:
        if alg in metrics["processing_times"] and alg in metrics["queue_delays"] and alg in metrics["power_consumption"]:
            available_algs.append(alg)
    
    if len(available_algs) < 2:
        print("Not enough data to compare FCFS algorithms. Run both FCFS algorithms first.")
        return
    
    # Extract data
    total_times = [metrics["processing_times"][alg]["total"] for alg in algorithms]
    fog_times = [metrics["processing_times"][alg]["fog"] for alg in algorithms]
    cloud_times = [metrics["processing_times"][alg]["cloud"] for alg in algorithms]
    queue_delays = [metrics["queue_delays"][alg] for alg in algorithms]
    
    # Calculate total power consumption
    total_power = []
    for alg in algorithms:
        total_power.append(sum(metrics["power_consumption"][alg]))
    
    # Create labels
    labels = ["FCFS-C", "FCFS-NC"]
    
    # Set up the figure with 3 subplots (processing times, queue delays, power)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Processing Times
    width = 0.25
    x = np.arange(len(labels))
    total_bars = ax1.bar(x - width, total_times, width=width, label='Total')
    fog_bars = ax1.bar(x, fog_times, width=width, label='Fog')
    cloud_bars = ax1.bar(x + width, cloud_times, width=width, label='Cloud')
    
    # Add value labels on top of bars
    for bars in [total_bars, fog_bars, cloud_bars]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
            
    ax1.set_xlabel('Algorithm')
    ax1.set_ylabel('Processing Time (ms)')
    ax1.set_title('Processing Times')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()
    
    # Plot 2: Queue Delays
    bars2 = ax2.bar(labels, queue_delays)
    
    # Add value labels on top of bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width(), height, f"{height:.1f}", ha='left', va='center')
                    
    ax2.set_xlabel('Algorithm')
    ax2.set_ylabel('Queue Delay (ms)')
    ax2.set_title('Queue Delays')
    
    # Plot 3: Power Consumption
    bars3 = ax3.bar(labels, total_power)
    
    # Add value labels on the side of bars
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width(), height, f"{height:.1f}", ha='left', va='center')
                    
    ax3.set_xlabel('Algorithm')
    ax3.set_ylabel('Power Consumption (W)')
    ax3.set_title('Total Power Consumption')
    
    plt.tight_layout()
    
    save_path = save_to_final_code_folder('fcfs_comparison.png')
    plt.savefig(save_path)
    print(f"FCFS comparison plot saved as '{save_path}'")
    
    # Generate standard plots for these algorithms as well to match "Run all" output
    subset_metrics = {key: {alg: metrics[key][alg] for alg in algorithms if alg in metrics[key]} 
                     for key in metrics.keys()}
    
    # Generate individual plots
    plot_processing_times(subset_metrics)
    plot_total_time_comparison(subset_metrics)
    plot_task_distribution(subset_metrics)
    plot_queue_delays(subset_metrics)
    plot_power_consumption(subset_metrics)
    plot_total_power_consumption(subset_metrics)


def plot_random_comparison(metrics):
    """Plot comparison between Random with and without cooperation"""
    algorithms = ["RandomCooperation", "RandomNoCooperation"]
    
    # Check if the required algorithms are available in metrics
    available_algs = []
    for alg in algorithms:
        if alg in metrics["processing_times"] and alg in metrics["queue_delays"] and alg in metrics["power_consumption"]:
            available_algs.append(alg)
    
    if len(available_algs) < 2:
        print("Not enough data to compare Random algorithms. Run both Random algorithms first.")
        return
    
    # Extract data
    total_times = [metrics["processing_times"][alg]["total"] for alg in algorithms]
    fog_times = [metrics["processing_times"][alg]["fog"] for alg in algorithms]
    cloud_times = [metrics["processing_times"][alg]["cloud"] for alg in algorithms]
    queue_delays = [metrics["queue_delays"][alg] for alg in algorithms]
    
    # Calculate total power consumption
    total_power = []
    for alg in algorithms:
        total_power.append(sum(metrics["power_consumption"][alg]))
    
    # Create labels
    labels = ["Random-C", "Random-NC"]
    
    # Set up the figure with 3 subplots (processing times, queue delays, power)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Processing Times
    width = 0.25
    x = np.arange(len(labels))
    total_bars = ax1.bar(x - width, total_times, width=width, label='Total')
    fog_bars = ax1.bar(x, fog_times, width=width, label='Fog')
    cloud_bars = ax1.bar(x + width, cloud_times, width=width, label='Cloud')
    
    # Add value labels on the side of bars
    for bars in [total_bars, fog_bars, cloud_bars]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width(), height, f"{height:.1f}", ha='left', va='center')
            
    ax1.set_xlabel('Algorithm')
    ax1.set_ylabel('Processing Time (ms)')
    ax1.set_title('Processing Times')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()
    
    # Plot 2: Queue Delays
    bars2 = ax2.bar(labels, queue_delays)
    
    # Add value labels on the side of bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width(), height, f"{height:.1f}", ha='left', va='center')
                    
    ax2.set_xlabel('Algorithm')
    ax2.set_ylabel('Queue Delay (ms)')
    ax2.set_title('Queue Delays')
    
    # Plot 3: Power Consumption
    bars3 = ax3.bar(labels, total_power)
    
    # Add value labels on the side of bars
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width(), height, f"{height:.1f}", ha='left', va='center')
                    
    ax3.set_xlabel('Algorithm')
    ax3.set_ylabel('Power Consumption (W)')
    ax3.set_title('Total Power Consumption')
    
    plt.tight_layout()
    
    save_path = save_to_final_code_folder('random_comparison.png')
    plt.savefig(save_path)
    print(f"Random comparison plot saved as '{save_path}'")
    
    # Generate standard plots for these algorithms as well to match "Run all" output
    subset_metrics = {key: {alg: metrics[key][alg] for alg in algorithms if alg in metrics[key]} 
                     for key in metrics.keys()}
    
    # Generate individual plots
    plot_processing_times(subset_metrics)
    plot_total_time_comparison(subset_metrics)
    plot_task_distribution(subset_metrics)
    plot_queue_delays(subset_metrics)
    plot_power_consumption(subset_metrics)
    plot_total_power_consumption(subset_metrics)


def generate_all_graphs(metrics):
    """Generate all available graphs from metrics"""
    # Plot processing times comparison
    plot_processing_times(metrics)
    
    # Plot total processing time comparison
    plot_total_time_comparison(metrics)
    
    # Plot task distribution between fog and cloud
    plot_task_distribution(metrics)
    
    # Plot queue delays comparison
    plot_queue_delays(metrics)
    
    # Plot power consumption comparison
    plot_power_consumption(metrics)
    
    # Plot total power consumption (power × nodes)
    plot_total_power_consumption(metrics)
    
    # Plot FCFS comparison (cooperative vs non-cooperative)
    plot_fcfs_comparison(metrics)
    
    # Plot Random comparison (cooperative vs non-cooperative)
    plot_random_comparison(metrics)


def main():
    """Main function to handle user input and execute requested tasks"""
    # Print header
    print("=== Fog-Cloud Task Allocation Results Viewer ===\n")
    
    # Map numeric inputs to algorithm names
    algorithm_names = {
        "1": "FCFSCooperation",
        "2": "FCFSNoCooperation",
        "3": "RandomCooperation",
        "4": "RandomNoCooperation"
    }
    
    # Check if command-line arguments were provided
    if len(sys.argv) > 1:
        # Handle command-line mode
        algorithm_arg = sys.argv[1]
        
        outputs = {}
        
        # Option 6 recomputes all results
        if algorithm_arg == "6":
            print("Recomputing all results...")
            for num, name in algorithm_names.items():
                outputs[name] = run_algorithm(num)
                
            # Parse results and generate graphs
            metrics = parse_results(outputs)
            generate_all_graphs(metrics)
            return
        
        # Run all algorithms or a specific one
        if algorithm_arg == "5":
            for num, name in algorithm_names.items():
                outputs[name] = run_algorithm(num)
                
            # Parse results
            metrics = parse_results(outputs)
            
            # Display comparative analysis
            print("\n=== Comparative Analysis ===")
            display_comparative_analysis(metrics)
            
            # Generate graphs
            generate_all_graphs(metrics)
        # FCFS comparison (cooperative vs non-cooperative)
        elif algorithm_arg == "7":
            print("Running FCFS algorithms (cooperative and non-cooperative)...")
            outputs["FCFSCooperation"] = run_algorithm("1")
            outputs["FCFSNoCooperation"] = run_algorithm("2")
            
            # Parse results
            metrics = parse_results(outputs)
            
            # Display comparative analysis
            print("\n=== FCFS Cooperative vs Non-Cooperative Analysis ===")
            display_comparative_analysis(metrics)
            
            # Generate FCFS comparison graph plus standard plots
            plot_fcfs_comparison(metrics)
            print("\nFCFS comparison completed. Graphs saved in the final_code folder.")
        # Random comparison (cooperative vs non-cooperative)
        elif algorithm_arg == "8":
            print("Running Random algorithms (cooperative and non-cooperative)...")
            outputs["RandomCooperation"] = run_algorithm("3")
            outputs["RandomNoCooperation"] = run_algorithm("4")
            
            # Parse results
            metrics = parse_results(outputs)
            
            # Display comparative analysis
            print("\n=== Random Cooperative vs Non-Cooperative Analysis ===")
            display_comparative_analysis(metrics)
            
            # Generate Random comparison graph plus standard plots
            plot_random_comparison(metrics)
            print("\nRandom comparison completed. Graphs saved in the final_code folder.")
        else:
            # Run a specific algorithm
            if algorithm_arg in algorithm_names:
                name = algorithm_names[algorithm_arg]
                outputs[name] = run_algorithm(algorithm_arg)
                
                # Parse results
                metrics = parse_results(outputs)
                
                # Display analysis for this algorithm only
                print(f"\n=== Analysis for {name} ===")
                display_comparative_analysis(metrics)
            else:
                print(f"Unknown algorithm: {algorithm_arg}")
                print("Usage: python display_results.py <algorithm_number>")
                print("  1: FCFSCooperation")
                print("  2: FCFSNoCooperation")
                print("  3: RandomCooperation")
                print("  4: RandomNoCooperation")
                print("  5: All algorithms")
                print("  6: Generate graphs (will recompute all results)")
                print("  7: FCFS Cooperative vs Non-Cooperative comparison")
                print("  8: Random Cooperative vs Non-Cooperative comparison")
                return
        
        return
    
    # Interactive menu mode
    try:
        while True:
            print("\nChoose an option:")
            print("  1: FCFSCooperation")
            print("  2: FCFSNoCooperation")
            print("  3: RandomCooperation")
            print("  4: RandomNoCooperation")
            print("  5: Run all algorithms and generate graphs")
            print("  6: FCFS Cooperative vs Non-Cooperative comparison")
            print("  7: Random Cooperative vs Non-Cooperative comparison")
            print("  8: Exit")
            
            # Get user choice
            choice = input("Enter your choice (1-8): ")
            
            # Exit option
            if choice == "8":
                print("Exiting program.")
                break
            
            outputs = {}
            
            # Run all algorithms and generate graphs
            if choice == "5":
                print("\nRunning all algorithms and generating graphs...")
                for num, name in algorithm_names.items():
                    outputs[name] = run_algorithm(num)
                
                # Parse results
                metrics = parse_results(outputs)
                
                # Display comparative analysis
                print("\n=== Comparative Analysis ===")
                display_comparative_analysis(metrics)
                
                # Generate all graphs
                generate_all_graphs(metrics)
                print("\nAll tasks completed. Graphs saved in the final_code folder.")
            
            # FCFS comparison (cooperative vs non-cooperative)
            elif choice == "6":
                print("\nRunning FCFS algorithms (cooperative and non-cooperative)...")
                outputs["FCFSCooperation"] = run_algorithm("1")
                outputs["FCFSNoCooperation"] = run_algorithm("2")
                
                # Parse results
                metrics = parse_results(outputs)
                
                # Display comparative analysis
                print("\n=== FCFS Cooperative vs Non-Cooperative Analysis ===")
                display_comparative_analysis(metrics)
                
                # Generate FCFS comparison graph plus standard plots
                plot_fcfs_comparison(metrics)
                print("\nFCFS comparison completed. Graphs saved in the final_code folder.")
            
            # Random comparison (cooperative vs non-cooperative)
            elif choice == "7":
                print("\nRunning Random algorithms (cooperative and non-cooperative)...")
                outputs["RandomCooperation"] = run_algorithm("3")
                outputs["RandomNoCooperation"] = run_algorithm("4")
                
                # Parse results
                metrics = parse_results(outputs)
                
                # Display comparative analysis
                print("\n=== Random Cooperative vs Non-Cooperative Analysis ===")
                display_comparative_analysis(metrics)
                
                # Generate Random comparison graph plus standard plots
                plot_random_comparison(metrics)
                print("\nRandom comparison completed. Graphs saved in the final_code folder.")
                
            # Run a specific algorithm
            elif choice in algorithm_names:
                name = algorithm_names[choice]
                outputs[name] = run_algorithm(choice)
                
                # Parse results
                metrics = parse_results(outputs)
                
                # Display analysis for this algorithm only
                print(f"\n=== Analysis for {name} ===")
                display_comparative_analysis(metrics)
                
                # Ask if user wants to generate graphs for this algorithm
                graph_choice = input("\nGenerate graphs for this algorithm? (y/n): ")
                if graph_choice.lower() == 'y':
                    generate_all_graphs(metrics)
                    print(f"\nGraphs for {name} saved in the final_code folder.")
                    
            else:
                print(f"Invalid choice: {choice}. Please try again.")
    except EOFError:
        # Handle non-interactive environments
        print("\nNon-interactive environment detected. Please provide command-line arguments:")
        print("Usage: python display_results.py <algorithm_number>")
        print("  1: FCFSCooperation")
        print("  2: FCFSNoCooperation")
        print("  3: RandomCooperation")
        print("  4: RandomNoCooperation")
        print("  5: All algorithms")
        print("  6: Generate graphs (will recompute all results)")
        print("  7: FCFS Cooperative vs Non-Cooperative comparison")
        print("  8: Random Cooperative vs Non-Cooperative comparison")


if __name__ == '__main__':
    main()
