#!/usr/bin/env python3
import time
import os
import importlib
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import all four algorithm files as modules
import FCFSC
import FCFSN
import RANDOMC
import RANDOMN

# Algorithms with their friendly names for display
ALGORITHMS = {
    "FCFS with Cooperation": FCFSC,
    "FCFS without Cooperation": FCFSN,
    "Random with Cooperation": RANDOMC,
    "Random without Cooperation": RANDOMN
}

def run_comparative_analysis():
    """Run all four algorithms and compare their results"""
    print("=== Fog-Cloud Task Allocation Comparative Analysis ===\n")
    
    # Number of tasks to generate for testing
    num_tasks = 500  # Reduced from 100000 for faster testing
    
    print(f"Running comparative analysis with {num_tasks} tasks...")
    
    # Get Task and FogNode classes from one of the modules (should be same in all)
    # This avoids redefining them and ensures compatibility
    Task = FCFSC.Task
    FogNode = FCFSC.FogNode
    
    # Create standard fog nodes for all algorithms
    fog_nodes = []
    
    # Create Edge-Fog-01 (moderate capacity)
    fog1 = FogNode(
        name="Edge-Fog-01",
        num_devices=150,
        total_storage=2000,  # 2 TB
        total_ram=128,       # 128 GB
        mips_per_device=2000
    )
    fog_nodes.append(fog1)
    
    # Create Edge-Fog-02 (higher capacity)
    fog2 = FogNode(
        name="Edge-Fog-02",
        num_devices=200,
        total_storage=4000,  # 4 TB
        total_ram=256,       # 256 GB
        mips_per_device=2500
    )
    fog_nodes.append(fog2)
    
    # Create standard cloud service for all algorithms
    cloud_services = [{
        "name": "Cloud-DC-01",
        "location": (0, 0),
        "ram": 100000,  # 100 GB
        "mips": 50000,  # High compute
        "bw": 1000      # 1 Gbps
    }]
    
    # Generate tasks using the module's task generation function
    tasks = FCFSC.generate_tasks(num_tasks)
    
    # Store results for each algorithm
    results = {}
    
    # Run each algorithm
    for name, module in ALGORITHMS.items():
        print(f"\nRunning {name}...")
        
        # Get a fresh copy of fog nodes and tasks for each run
        local_fog_nodes = []
        for fog in fog_nodes:
            # Create a new fog node with the same parameters
            new_fog = FogNode(
                name=fog.name,
                num_devices=fog.num_devices,
                total_storage=fog.total_storage,
                total_ram=fog.total_ram,
                mips_per_device=fog.mips_per_device
            )
            local_fog_nodes.append(new_fog)
        
        # Create gateway with the algorithm's implementation
        gateway = module.FCFSCooperationGateway(local_fog_nodes, cloud_services)
        
        # Process all tasks
        for i, original_task in enumerate(tqdm(tasks, desc=f"Processing {name}")):
            # Create a fresh copy of the task for each run
            task = Task(
                task_id=original_task.id,
                data_type=original_task.data_type,
                size=original_task.size,
                arrival_time=original_task.arrival_time
            )
            
            # Process the task
            gateway.offload_task(task)
        
        # Store results
        results[name] = {
            'gateway': gateway,
            'metrics': gateway.metrics,
            'data_type_counts': gateway.data_type_counts
        }
    
    # Analyze and display results
    print("\n=== Comparative Analysis Results ===\n")
    
    # Processing times
    print("=== Average Processing Times (ms) ===")
    for name, result in results.items():
        # Calculate cloud and fog average processing times
        fog_times = result['metrics'].get('fog_times', [])
        cloud_times = result['metrics'].get('cloud_times', [])
        
        avg_fog = sum(fog_times) / len(fog_times) if fog_times else 0
        avg_cloud = sum(cloud_times) / len(cloud_times) if cloud_times else 0
        
        total_time = sum(fog_times) + sum(cloud_times)
        total_count = len(fog_times) + len(cloud_times)
        avg_total = total_time / total_count if total_count > 0 else 0
        
        print(f"{name}: Total = {avg_total:.2f}, Fog = {avg_fog:.2f}, Cloud = {avg_cloud:.2f}")
    
    # Queue delays
    print("\n=== Average Queue Delays (ms) ===")
    for name, result in results.items():
        queue_delays = result['metrics'].get('queue_delays', [])
        avg_delay = sum(queue_delays) / len(queue_delays) if queue_delays else 0
        print(f"{name}: {avg_delay:.2f}")
    
    # Task distribution
    print("\n=== Task Distribution ===")
    for name, result in results.items():
        data_type_counts = result['data_type_counts']
        fog_count = sum(counts['fog'] for counts in data_type_counts.values())
        cloud_count = sum(counts['cloud'] for counts in data_type_counts.values())
        
        total = fog_count + cloud_count
        print(f"{name}: Fog = {fog_count} ({fog_count/total*100:.1f}%), Cloud = {cloud_count} ({cloud_count/total*100:.1f}%)")
    
    # Selection times
    print("\n=== Average Selection Times (ms) ===")
    for name, result in results.items():
        metrics = result['metrics']
        
        selection_times = metrics.get('node_selection_time', [])
        avg_selection = sum(selection_times) / len(selection_times) if selection_times else 0
        
        # Some policies may not have alt_node_selection_time
        alt_selection_times = metrics.get('alt_node_selection_time', [])
        avg_alt_selection = sum(alt_selection_times) / len(alt_selection_times) if alt_selection_times else 0
        
        # Cloud selection time
        cloud_selection_times = metrics.get('cloud_selection_time', [])
        avg_cloud_selection = sum(cloud_selection_times) / len(cloud_selection_times) if cloud_selection_times else 0
        
        print(f"{name}: Node = {avg_selection:.4f}, Alt Node = {avg_alt_selection:.4f}, Cloud = {avg_cloud_selection:.4f}")

    print("\n=== Data Type Distribution ===")
    for name, result in results.items():
        print(f"\n{name} Distribution by Data Type:")
        data_type_counts = result['data_type_counts']
        print(f"{'Data Type':<15} | {'Fog Count':<10} | {'Cloud Count':<10} | {'Total':<10} | {'Fog %':<10}")
        print("-" * 70)
        
        for data_type, counts in data_type_counts.items():
            fog_count = counts.get('fog', 0)
            cloud_count = counts.get('cloud', 0)
            total = fog_count + cloud_count
            fog_percent = (fog_count / total * 100) if total > 0 else 0
            
            print(f"{data_type:<15} | {fog_count:<10} | {cloud_count:<10} | {total:<10} | {fog_percent:<10.1f}%")


if __name__ == "__main__":
    run_comparative_analysis()
