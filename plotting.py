import matplotlib.pyplot as plt
import numpy as np

def plot_processing_times(results):
    """Plot average processing times for different policies."""
    policy_names = list(results.keys())
    
    # Calculate average times
    fog_times = [np.mean(results[policy]['fog_times']) if results[policy]['fog_times'] else 0 for policy in policy_names]
    cloud_times = [np.mean(results[policy]['cloud_times']) if results[policy]['cloud_times'] else 0 for policy in policy_names]
    
    # Set up the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Fog times plot (with smaller scale)
    bar_positions = np.arange(len(policy_names))
    bars = ax1.bar(bar_positions, fog_times, color='skyblue')
    ax1.set_title('Average Fog Processing Times (ms)')
    ax1.set_xlabel('Policy')
    ax1.set_ylabel('Time (ms)')
    ax1.set_xticks(bar_positions)
    ax1.set_xticklabels(policy_names, rotation=45)
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    # Cloud times plot
    bars = ax2.bar(bar_positions, cloud_times, color='coral')
    ax2.set_title('Average Cloud Processing Times (ms)')
    ax2.set_xlabel('Policy')
    ax2.set_ylabel('Time (ms)')
    ax2.set_xticks(bar_positions)
    ax2.set_xticklabels(policy_names, rotation=45)
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('processing_times.png')
    plt.close()

def plot_task_distribution(results):
    """Plot task distribution between fog and cloud for different policies."""
    policy_names = list(results.keys())
    
    # Calculate task counts
    fog_tasks = [len(results[policy]['fog_times']) for policy in policy_names]
    cloud_tasks = [len(results[policy]['cloud_times']) for policy in policy_names]
    
    # Calculate percentages
    totals = [fog + cloud for fog, cloud in zip(fog_tasks, cloud_tasks)]
    fog_percents = [fog / total * 100 for fog, total in zip(fog_tasks, totals)]
    cloud_percents = [cloud / total * 100 for cloud, total in zip(cloud_tasks, totals)]
    
    # Set up the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Task count plot
    bar_width = 0.35
    bar_positions = np.arange(len(policy_names))
    ax1.bar(bar_positions - bar_width/2, fog_tasks, bar_width, label='Fog', color='skyblue')
    ax1.bar(bar_positions + bar_width/2, cloud_tasks, bar_width, label='Cloud', color='coral')
    ax1.set_title('Task Distribution (Count)')
    ax1.set_xlabel('Policy')
    ax1.set_ylabel('Number of Tasks')
    ax1.set_xticks(bar_positions)
    ax1.set_xticklabels(policy_names, rotation=45)
    ax1.legend()
    
    # Percentage plot
    ax2.bar(bar_positions, fog_percents, label='Fog', color='skyblue')
    ax2.bar(bar_positions, cloud_percents, bottom=fog_percents, label='Cloud', color='coral')
    ax2.set_title('Task Distribution (Percentage)')
    ax2.set_xlabel('Policy')
    ax2.set_ylabel('Percentage')
    ax2.set_xticks(bar_positions)
    ax2.set_xticklabels(policy_names, rotation=45)
    ax2.set_ylim(0, 100)
    
    # Add percentage labels
    for i, (f, c) in enumerate(zip(fog_percents, cloud_percents)):
        ax2.annotate(f'{f:.1f}%', 
                    xy=(i, f/2), 
                    ha='center', va='center')
        ax2.annotate(f'{c:.1f}%', 
                    xy=(i, f + c/2), 
                    ha='center', va='center')
    
    ax2.legend()
    plt.tight_layout()
    plt.savefig('task_distribution.png')
    plt.close()

def plot_queue_delays(results):
    """Plot average queue delays for different policies."""
    policy_names = list(results.keys())
    
    # Calculate average queue delays
    queue_delays = [np.mean(results[policy]['queue_delays']) if results[policy]['queue_delays'] else 0 
                   for policy in policy_names]
    
    # Set up the figure
    plt.figure(figsize=(10, 6))
    bars = plt.bar(policy_names, queue_delays, color='purple')
    plt.title('Average Queue Delays (ms)')
    plt.xlabel('Policy')
    plt.ylabel('Delay (ms)')
    plt.xticks(rotation=45)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
                    
    plt.tight_layout()
    plt.savefig('queue_delays.png')
    plt.close()

def plot_power_consumption(results):
    """Plot average power consumption for different policies."""
    policy_names = list(results.keys())
    
    # Get power consumption data with 2 nodes
    power_data = []
    for policy in policy_names:
        node_power = []
        for node in results[policy]['power']:
            node_power.append(np.mean(node) if node else 100)
        power_data.append(node_power)
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set the positions and width for the bars
    bar_width = 0.35
    pos = np.arange(len(policy_names))
    
    # Create bars for each node
    for i in range(len(power_data[0])):  # Assuming all policies have same number of nodes
        node_values = [policy_data[i] for policy_data in power_data]
        ax.bar(pos + i*bar_width/len(power_data[0]), node_values, bar_width/len(power_data[0]), 
               label=f'Node {i+1}')
    
    # Add labels and title
    ax.set_title('Average Power Consumption by Node (W)')
    ax.set_xlabel('Policy')
    ax.set_ylabel('Power (W)')
    ax.set_xticks(pos + bar_width/4)  # Center the ticks
    ax.set_xticklabels(policy_names, rotation=45)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('power_consumption.png')
    plt.close()

def plot_total_power_consumption(results):
    """Plot total power consumption across all nodes for different policies."""
    policy_names = list(results.keys())
    
    # Calculate total power consumption (average across all nodes)
    total_power = []
    for policy in policy_names:
        policy_total = 0
        num_nodes = 0
        for node in results[policy]['power']:
            if node:
                policy_total += np.mean(node)
                num_nodes += 1
        total_power.append(policy_total if num_nodes == 0 else policy_total/num_nodes)
    
    # Set up the figure
    plt.figure(figsize=(10, 6))
    bars = plt.bar(policy_names, total_power, color='green')
    plt.title('Average Total Power Consumption (W)')
    plt.xlabel('Policy')
    plt.ylabel('Power (W)')
    plt.xticks(rotation=45)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('total_power_consumption.png')
    plt.close()

def plot_total_time(results):
    """Plot average total processing time for different policies."""
    policy_names = list(results.keys())
    
    # Calculate total time (weighted average of fog and cloud times)
    total_times = []
    for policy in policy_names:
        fog_times = results[policy]['fog_times']
        cloud_times = results[policy]['cloud_times']
        all_times = fog_times + cloud_times
        
        if all_times:
            total_times.append(np.mean(all_times))
        else:
            total_times.append(0)
    
    # Set up the figure
    plt.figure(figsize=(10, 6))
    bars = plt.bar(policy_names, total_times, color='teal')
    plt.title('Average Total Processing Time (ms)')
    plt.xlabel('Policy')
    plt.ylabel('Time (ms)')
    plt.xticks(rotation=45)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('total_time.png')
    plt.close()

def plot_response_time(results):
    """Plot average response time (processing + queue delay) for different policies."""
    policy_names = list(results.keys())
    
    # Calculate processing times
    processing_times = []
    for policy in policy_names:
        fog_times = results[policy]['fog_times']
        cloud_times = results[policy]['cloud_times']
        all_times = fog_times + cloud_times
        
        if all_times:
            processing_times.append(np.mean(all_times))
        else:
            processing_times.append(0)
    
    # Calculate queue delays
    queue_delays = [np.mean(results[policy]['queue_delays']) if results[policy]['queue_delays'] else 0 
                   for policy in policy_names]
    
    # Calculate response time (processing + queue delay)
    response_times = [p + q for p, q in zip(processing_times, queue_delays)]
    
    # Set up the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # First subplot: Response time
    bars = ax1.bar(policy_names, response_times, color='darkblue')
    ax1.set_title('Average Response Time (ms)')
    ax1.set_xlabel('Policy')
    ax1.set_ylabel('Time (ms)')
    ax1.set_xticks(range(len(policy_names)))
    ax1.set_xticklabels(policy_names, rotation=45)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    # Second subplot: Stacked bar chart showing processing time and queue delay
    bar_width = 0.35
    bar_positions = np.arange(len(policy_names))
    
    # Plot processing time at the bottom
    ax2.bar(bar_positions, processing_times, bar_width, 
            label='Processing Time', color='lightblue')
    
    # Plot queue delay on top of processing time
    ax2.bar(bar_positions, queue_delays, bar_width,
            bottom=processing_times, label='Queue Delay', color='lightcoral')
    
    ax2.set_title('Response Time Components (ms)')
    ax2.set_xlabel('Policy')
    ax2.set_ylabel('Time (ms)')
    ax2.set_xticks(bar_positions)
    ax2.set_xticklabels(policy_names, rotation=45)
    ax2.legend()
    
    # Add percentage labels
    for i, (p, q) in enumerate(zip(processing_times, queue_delays)):
        # Only add labels if the values are significant
        if p > 10:
            ax2.annotate(f'{p:.1f}', 
                       xy=(i, p/2), 
                       ha='center', va='center')
        if q > 10:
            ax2.annotate(f'{q:.1f}', 
                       xy=(i, p + q/2), 
                       ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig('response_time.png')
    plt.close()

def plot_comparison_graph(results):
    """Create a comprehensive comparison graph showing multiple metrics."""
    policy_names = list(results.keys())
    
    # Calculate metrics
    fog_times = [np.mean(results[policy]['fog_times']) if results[policy]['fog_times'] else 0 for policy in policy_names]
    cloud_times = [np.mean(results[policy]['cloud_times']) if results[policy]['cloud_times'] else 0 for policy in policy_names]
    queue_delays = [np.mean(results[policy]['queue_delays']) if results[policy]['queue_delays'] else 0 for policy in policy_names]
    
    # Calculate task distribution percentages
    fog_tasks = [len(results[policy]['fog_times']) for policy in policy_names]
    cloud_tasks = [len(results[policy]['cloud_times']) for policy in policy_names]
    totals = [fog + cloud for fog, cloud in zip(fog_tasks, cloud_tasks)]
    fog_percents = [fog / total * 100 for fog, total in zip(fog_tasks, totals)]
    
    # Normalize for visualization on same scale (0-1)
    max_cloud = max(cloud_times)
    norm_cloud = [c/max_cloud for c in cloud_times]
    max_queue = max(queue_delays) if max(queue_delays) > 0 else 1
    norm_queue = [q/max_queue for q in queue_delays]
    norm_fog_percent = [f/100 for f in fog_percents]
    
    # Set up the figure with a radar plot
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))
    
    # Number of metrics
    N = 4
    
    # Angles for each metric (in radians)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Labels for each angle
    metric_labels = ['Fog Time\n(Lower is better)', 
                    'Cloud Time\n(Lower is better)', 
                    'Queue Delay\n(Lower is better)',
                    'Fog Usage %\n(Higher is better)']
    
    # Set the labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels)
    
    # Draw the polygons for each policy
    colors = ['blue', 'red', 'green', 'orange']
    for i, policy in enumerate(policy_names):
        # For 'Fog Percent', invert the normalization (higher is better)
        values = [fog_times[i]/max(fog_times), 
                 norm_cloud[i], 
                 norm_queue[i], 
                 1 - norm_fog_percent[i]]  # Invert fog percentage (lower is better for radar)
        values += values[:1]  # Close the loop
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=policy, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title("Policy Comparison (Lower values are better)")
    
    plt.tight_layout()
    plt.savefig('policy_comparison_radar.png')
    plt.close()

def plot_all(results):
    """Generate all plots."""
    plot_processing_times(results)
    plot_task_distribution(results)
    plot_queue_delays(results)
    plot_power_consumption(results)
    plot_total_power_consumption(results)
    plot_total_time(results)
    plot_response_time(results)
    plot_comparison_graph(results)
    print("All plots have been generated and saved as PNG files.")

# Add this to the main function in text.py:
# if results:
#     analyze_results(results)
#     from plotting import plot_all
#     plot_all(results)
# else:
#     print("No valid policy selected!") 