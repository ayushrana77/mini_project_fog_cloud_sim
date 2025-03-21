import json
import os
import numpy as np
import random
from collections import defaultdict
from dataclasses import dataclass
from math import radians, sin, cos, sqrt, atan2
from typing import Dict, List, Optional, Any
import time
from copy import deepcopy

# ========== Configuration Section ==========
# Batch Processing Configuration
BATCH_SIZE = 1000
BATCHES_BEFORE_RESET = 10
BATCH_RESET_DELAY = 100  # ms between batch resets

FOG_NODES = [
    {
        "name": "Edge-Fog-01",
        "location": (33.72, 72.85),
        "down_bw": 50000,
        "up_bw": 30000,
        "mips": 200000,
        "num_pes": 2400,
        "ram": 327680,
        "storage": 800000,
        "num_devices": 250
    },
    {
        "name": "Edge-Fog-02",
        "location": (34.12, 73.25),
        "down_bw": 60000,
        "up_bw": 40000,
        "mips": 250000,
        "num_pes": 3200,
        "ram": 491520,
        "storage": 1000000,
        "num_devices": 250
    }
]

CLOUD_SERVICES = [
    {
        "name": "USA-Service1",
        "location": (37.09, -95.71),
        "ram": 16384,
        "mips": 15000,
        "bw": 800
    },
    {
        "name": "Singapore-Service1",
        "location": (1.35, 103.82),
        "ram": 16384,
        "mips": 15000,
        "bw": 800
    }
]

# Helper function to load tasks
def load_tasks(filepath, limit=None):
    """Load tasks from JSON file with proper error handling."""
    try:
        print(f"Loading tasks from {filepath}...")
        
        if not os.path.exists(filepath):
            print(f"Error: File {filepath} not found")
            return []
            
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
            
        if not isinstance(data, list):
            print("Error: Expected JSON array at root level")
            return []
            
        print(f"Successfully loaded {len(data)} tasks from {filepath}")
        
        # Limit the number of tasks for testing if specified
        if limit:
            data = data[:limit]
            print(f"Using {len(data)} tasks for simulation")
            
        # Define the Task fields
        task_fields = {'id', 'size', 'name', 'mips', 'number_of_pes', 'ram', 'bw', 
                       'data_type', 'location', 'device_type', 'arrival_time', 
                       'fog_candidate'}
        
        tasks = []
        for item in data:
            try:
                # Set initial fog_candidate to True
                fog_candidate = True
                
                # Determine if task should be a fog candidate based on data type and size
                if item['DataType'] in ['Large', 'Bulk'] and int(item['Size']) > 250:
                    fog_candidate = False
                elif item['DataType'] in ['Abrupt', 'LocationBased', 'Medical', 'SmallTextual', 'Multimedia'] and int(item['Size']) > 230:
                    fog_candidate = False
                
                # Create Task object with properly mapped attributes
                task = Task(
                    id=item['ID'],
                    size=int(item['Size']),
                    name=item['Name'],
                    mips=float(item['MIPS']),
                    number_of_pes=item['NumberOfPes'],
                    ram=int(item['RAM']),
                    bw=item['BW'],
                    data_type=item['DataType'],
                    location=(
                        float(item['GeoLocation']['latitude']),
                        float(item['GeoLocation']['longitude'])
                    ),
                    device_type=item['DeviceType'],
                    arrival_time=random.uniform(0, 1000),
                    fog_candidate=fog_candidate
                )
                tasks.append(task)
                
            except Exception as e:
                print(f"Error creating task: {e}")
                continue
                
        print(f"Successfully created {len(tasks)} task objects")
        return tasks
        
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in {filepath}: {str(e)}")
        return []
    except Exception as e:
        print(f"Error loading tasks: {str(e)}")
        return []

def main():
    """Run the experiment"""
    print("\n=== Running all algorithms with limited task dataset ===")
    
    # Try to load tasks - first from Tuple10.json, then from tuples.json if needed
    tasks = []
    try:
        tasks = load_tasks("Tuple10.json", limit=500)
    except Exception as e:
        print(f"Failed to load Tuple10.json: {e}")
        
    if not tasks:
        try:
            tasks = load_tasks("tuples.json", limit=500)
        except Exception as e:
            print(f"Failed to load tuples.json: {e}")
            
    if not tasks:
        print("Failed to load any tasks. Exiting.")
        return
    
    # Sort tasks by arrival time
    sorted_tasks = sorted(tasks, key=lambda x: x.arrival_time)
    
    # Initialize fog nodes to get num_devices
    fog_nodes = [FogNode(config) for config in FOG_NODES]
    total_fog_capacity = sum(node.num_devices for node in fog_nodes)
    print(f"Total fog device capacity: {total_fog_capacity} devices")
    print(f"Average tasks per device: {len(sorted_tasks)/total_fog_capacity:.2f} tasks/device")
    
    # Define batch size based on node capacity
    batch_size = min(BATCH_SIZE, total_fog_capacity // 2)  # Use at most half capacity per batch
    print(f"Using batch size: {batch_size} (based on fog capacity)")
    
    # Define algorithms to run
    algorithms = ["FCFSCooperation", "FCFSNoCooperation", "RandomCooperation", "RandomNoCooperation"]
    
    # Run all algorithms
    for algorithm_name in algorithms:
        print(f"\n=== Running {algorithm_name} algorithm ===")
        start_time = time.time()
        
        # Initialize algorithm-specific gateway
        if algorithm_name == "FCFSCooperation":
            gateway = FCFSCooperationGateway(
                [FogNode(config) for config in FOG_NODES], 
                [CloudService(config) for config in CLOUD_SERVICES]
            )
        elif algorithm_name == "FCFSNoCooperation":
            gateway = FCFSGateway(
                [FogNode(config) for config in FOG_NODES], 
                [CloudService(config) for config in CLOUD_SERVICES]
            )
        elif algorithm_name == "RandomCooperation":
            gateway = RandomCooperationGateway(
                [FogNode(config) for config in FOG_NODES], 
                [CloudService(config) for config in CLOUD_SERVICES]
            )
        elif algorithm_name == "RandomNoCooperation":
            gateway = RandomGateway(
                [FogNode(config) for config in FOG_NODES], 
                [CloudService(config) for config in CLOUD_SERVICES]
            )
        else:
            print(f"Unknown algorithm: {algorithm_name}")
            continue
            
        # Set batch size and verbose mode
        gateway.batch_size = batch_size
        gateway.verbose_output = True
        
        # Process all tasks
        fog_count = 0
        cloud_count = 0
        remaining_tasks = sorted_tasks.copy()
        
        # Process tasks in batches
        while remaining_tasks:
            # Get next batch
            batch_size = min(gateway.batch_size, len(remaining_tasks))
            current_batch = remaining_tasks[:batch_size]
            remaining_tasks = remaining_tasks[batch_size:]
            
            # Process the batch
            batch_completion_time = gateway.process_batch(current_batch)
            
            # Count tasks by processing location
            for task in current_batch:
                if task.id in gateway.processed_tasks:
                    continue
                gateway.processed_tasks.add(task.id)
                
                if hasattr(task, 'is_cloud_served') and task.is_cloud_served:
                    cloud_count += 1
                else:
                    fog_count += 1
            
            # Reset nodes for the next batch
            gateway.reset_nodes()
        
        # Record metrics
        runtime = time.time() - start_time
        
        # Display algorithm summary
        total_tasks = fog_count + cloud_count
        fog_percent = (fog_count / total_tasks * 100) if total_tasks > 0 else 0
        cloud_percent = (cloud_count / total_tasks * 100) if total_tasks > 0 else 0
        
        print(f"\nAlgorithm: {algorithm_name}")
        print(f"Total execution time: {runtime:.2f} seconds")
        print(f"Total tasks processed: {total_tasks}")
        print(f"Fog Processing: {fog_count} tasks ({fog_percent:.1f}%)")
        print(f"Cloud Processing: {cloud_count} tasks ({cloud_percent:.1f}%)")
        
        if gateway.metrics['fog_times']:
            avg_fog_time = sum(gateway.metrics['fog_times']) / len(gateway.metrics['fog_times'])
            print(f"Average Fog Processing Time: {avg_fog_time:.2f} ms")
            
        if gateway.metrics['cloud_times']:
            avg_cloud_time = sum(gateway.metrics['cloud_times']) / len(gateway.metrics['cloud_times'])
            print(f"Average Cloud Processing Time: {avg_cloud_time:.2f} ms")
            
        print(f"Peak Device Usage: {gateway.max_devices_used}/{total_fog_capacity} ({gateway.max_devices_used/total_fog_capacity*100:.1f}%)")
    
    print("\nAll algorithms completed.")
    
if __name__ == "__main__":
    try:
        print("Starting script execution...")
        main()
        print("Script completed successfully.")
    except Exception as e:
        import traceback
        print(f"Error during execution: {e}")
        traceback.print_exc()
        print("Script terminated with error.") 