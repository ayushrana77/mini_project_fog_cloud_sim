#!/usr/bin/env python3
import time
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy

# Import the modules instead of specific classes since they all use the same class name
import FCFSC    # FCFS with Cooperation
import FCFSN    # FCFS without Cooperation
import RANDOMC  # Random with Cooperation
import RANDOMN  # Random without Cooperation

# Constants for fog node configuration
NUM_FOG_NODES = 2
FOG_NODES = []

# Task generation parameters
NUM_TASKS = 100000
TASK_TYPES = ['Abrupt', 'Large', 'LocationBased', 'Bulk', 'Medical', 'SmallTextual', 'Multimedia']
TASK_SIZES = {
    'small': (10, 50),
    'medium': (51, 200),
    'large': (201, 500)
}

class Task:
    """Represents a computation task with various attributes"""
    
    def __init__(self, task_id, data_type, size, arrival_time):
        self.id = task_id
        self.data_type = data_type
        self.size = size  # KB
        self.arrival_time = arrival_time
        
        # Set RAM and MIPS requirements based on size
        # Larger tasks require more resources
        self.ram = size / 10  # MB
        self.mips = size * 10  # Million Instructions Per Second
        
        # Set deadline based on size and type
        self.deadline = size * 0.1  # ms
        if data_type in ['Abrupt', 'Medical']:
            self.deadline *= 0.5  # Stricter deadline for critical tasks
        
        # Track processing information
        self.processor_node = None
        self.start_time = 0
        self.completion_time = 0
        self.processing_time = 0


def generate_tasks(num_tasks):
    """Generate a list of tasks with various characteristics"""
    tasks = []
    arrival_time = 0
    
    for i in range(num_tasks):
        # Realistic data type selection with probabilities
        data_type = random.choices(
            TASK_TYPES, 
            weights=[0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.10],
            k=1
        )[0]
        
        # Size selection based on data type
        if data_type in ['Bulk', 'Large']:
            # These types are usually larger
            size_category = random.choices(
                ['medium', 'large'],
                weights=[0.3, 0.7],
                k=1
            )[0]
        elif data_type in ['Abrupt', 'Medical']:
            # These types are usually smaller due to urgency
            size_category = random.choices(
                ['small', 'medium'],
                weights=[0.7, 0.3],
                k=1
            )[0]
        else:
            # Other types have more balanced size distribution
            size_category = random.choices(
                ['small', 'medium', 'large'],
                weights=[0.33, 0.34, 0.33],
                k=1
            )[0]
        
        # Generate actual size within the category range
        size_range = TASK_SIZES[size_category]
        size = random.randint(*size_range)
        
        # Create task with arrival time (simulating real-time arrival)
        task = Task(i, data_type, size, arrival_time)
        tasks.append(task)
        
        # Update arrival time for next task (with some randomness)
        arrival_time += random.uniform(0.1, 1.0)
    
    return tasks


class FogNode:
    """Represents a fog computing node with various capabilities"""
    
    def __init__(self, name, num_devices=100, total_storage=1000, total_ram=1000, mips_per_device=1000):
        self.name = name
        self.num_devices = num_devices
        self.total_storage = total_storage  # GB
        self.total_ram = total_ram  # GB
        self.mips_per_device = mips_per_device
        
        # Dynamic state
        self.used_storage = 0
        self.used_ram = 0
        self.queue = []
        self.max_queue_size = 100
        self.network_congestion = 0.1  # 0 to 1 scale
        self.utilization = 0.0  # 0 to 100 percentage
        
        # Metrics
        self.cumulative_processed = 0
        self.cumulative_processing_time = 0
        self.cumulative_utilization = 0.0
        self.power_consumption = 100.0  # Watts (base level)
        
        # Performance capabilities
        self.processing_efficiency = 0.95  # 0 to 1 scale
        
        # Calculate total MIPS available
        self.total_mips = self.num_devices * self.mips_per_device
        
    @property
    def available_ram(self):
        """Get available RAM in GB"""
        return self.total_ram - self.used_ram
    
    @property
    def available_mips(self):
        """Get available MIPS"""
        return self.total_mips * (1 - self.utilization/100)
    
    def process(self, task, current_time):
        """Process a task and return processing metrics"""
        # Calculate queue delay if any
        queue_delay = 0
        if len(self.queue) > 0:
            queue_delay = len(self.queue) * 0.05  # Simple model: 0.05ms per queued task
        
        # Add to queue for tracking
        self.queue.append(task.id)
        if len(self.queue) > self.max_queue_size:
            self.queue.pop(0)  # Remove oldest task if queue is full
        
        # Calculate processing time based on task size and node efficiency
        # For fog nodes, we assume very fast processing due to proximity
        processing_time = 0.05  # Base time in ms
        
        # Update node state
        self.used_storage += task.size / 1000  # Convert KB to MB
        if self.used_storage > self.total_storage:
            self.used_storage = self.total_storage
            
        self.used_ram += task.ram / 1000  # Convert MB to GB
        if self.used_ram > self.total_ram:
            self.used_ram = self.total_ram
            
        # Update utilization based on new workload
        new_utilization = (self.used_ram / self.total_ram) * 100
        self.utilization = new_utilization
        
        # Update metrics
        self.cumulative_processed += 1
        self.cumulative_processing_time += processing_time
        self.cumulative_utilization = ((self.cumulative_utilization * (self.cumulative_processed - 1)) + 
                                       self.utilization) / self.cumulative_processed
        
        # Update power consumption - increases with utilization
        self.power_consumption = 100 + (self.utilization / 10)
        
        # Update network congestion (random fluctuation)
        self.network_congestion = max(0.1, min(0.9, self.network_congestion + random.uniform(-0.05, 0.05)))
        
        # Set task metadata
        task.start_time = current_time + queue_delay
        task.completion_time = task.start_time + processing_time
        task.processing_time = processing_time
        
        # Return queue delay, processing time, and completion time
        completion_time = current_time + queue_delay + processing_time
        return queue_delay, processing_time, completion_time
    
    def can_accept_task(self, task, current_time):
        """Check if the fog node can accept a task based on available resources"""
        # Check if queue has space
        if len(self.queue) >= self.max_queue_size:
            return False
        
        # Check if enough storage is available (with some margin)
        if (self.used_storage + task.size/1000) > (self.total_storage * 0.9):
            return False
        
        # Check if enough RAM is available (with some margin)
        if (self.used_ram + task.ram/1000) > (self.total_ram * 0.9):
            return False
        
        # Check if enough MIPS is available
        if task.mips > (self.available_mips * 0.9):
            return False
        
        # All checks passed
        return True


def initialize_fog_nodes():
    """Initialize fog nodes with different capabilities"""
    global FOG_NODES
    FOG_NODES = []
    
    # Create Edge-Fog-01 (moderate capacity)
    fog1 = FogNode(
        name="Edge-Fog-01",
        num_devices=150,
        total_storage=2000,  # 2 TB
        total_ram=128,       # 128 GB
        mips_per_device=2000
    )
    FOG_NODES.append(fog1)
    
    # Create Edge-Fog-02 (higher capacity)
    fog2 = FogNode(
        name="Edge-Fog-02",
        num_devices=200,
        total_storage=4000,  # 4 TB
        total_ram=256,       # 256 GB
        mips_per_device=2500
    )
    FOG_NODES.append(fog2)


def run_policy(policy_module, tasks, fog_nodes):
    """Run a specific policy and return results"""
    # Make deep copies of tasks and fog nodes to ensure independent runs
    local_tasks = deepcopy(tasks)
    local_fog_nodes = deepcopy(fog_nodes)
    
    # Create gateway with the specified policy
    gateway = policy_module.FCFSCooperationGateway(local_fog_nodes)
    
    # Process all tasks with progress bar
    with tqdm(total=len(local_tasks), desc=f"Processing {policy_module.__name__}") as pbar:
        for i, task in enumerate(local_tasks):
            gateway.process_task(task)
            pbar.update(1)
    
    # Calculate and return results
    results = {
        'gateway': gateway,
        'tasks': local_tasks,
        'fog_nodes': local_fog_nodes,
        'metrics': gateway.metrics,
        'data_type_counts': gateway.data_type_counts
    }
    
    return results


def analyze_results(results):
    """Analyze and print comparative results"""
    print("\n=== Comparative Analysis ===\n")
    
    # Processing times
    print("=== Average Processing Times (ms) ===")
    for policy, data in results.items():
        gateway = data['gateway']
        tasks = data['tasks']
        
        # Calculate cloud and fog average processing times
        fog_times = gateway.metrics.get('fog_times', [])
        cloud_times = []
        for task in tasks:
            if task.processor_node is None or 'Cloud' in str(task.processor_node):
                cloud_times.append(task.processing_time)
        
        avg_fog = sum(fog_times) / len(fog_times) if fog_times else 0
        avg_cloud = sum(cloud_times) / len(cloud_times) if cloud_times else 0
        avg_total = (sum(fog_times) + sum(cloud_times)) / len(tasks)
        
        print(f"{policy}: Total = {avg_total:.2f}, Fog = {avg_fog:.2f}, Cloud = {avg_cloud:.2f}")
    
    # Power consumption
    print("\n=== Average Power Consumption per Node (W) ===")
    for policy, data in results.items():
        fog_nodes = data['fog_nodes']
        power_values = [f'{node.power_consumption:.2f}' for node in fog_nodes]
        print(f"{policy}: {power_values}")
    
    # Queue delays
    print("\n=== Average Queue Delays (ms) ===")
    for policy, data in results.items():
        gateway = data['gateway']
        queue_delays = gateway.metrics.get('queue_delays', [])
        avg_delay = sum(queue_delays) / len(queue_delays) if queue_delays else 0
        print(f"{policy}: {avg_delay:.2f}")
    
    # Task distribution
    print("\n=== Task Distribution ===")
    for policy, data in results.items():
        gateway = data['gateway']
        tasks = data['tasks']
        
        # Count tasks by processor type
        fog_count = 0
        cloud_count = 0
        for task in tasks:
            if task.processor_node is not None and 'Fog' in str(task.processor_node):
                fog_count += 1
            else:
                cloud_count += 1
        
        total = fog_count + cloud_count
        print(f"{policy}: Fog = {fog_count} ({fog_count/total*100:.1f}%), Cloud = {cloud_count} ({cloud_count/total*100:.1f}%)")
    
    # Selection times
    print("\n=== Average Selection Times (ms) ===")
    for policy, data in results.items():
        gateway = data['gateway']
        
        selection_times = gateway.metrics.get('node_selection_time', [])
        avg_selection = sum(selection_times) / len(selection_times) if selection_times else 0
        
        alt_selection_times = gateway.metrics.get('alt_node_selection_time', [])
        avg_alt_selection = sum(alt_selection_times) / len(alt_selection_times) if alt_selection_times else 0
        
        cloud_selection_times = gateway.metrics.get('cloud_selection_time', [1.0])  # Default to 1ms
        avg_cloud_selection = sum(cloud_selection_times) / len(cloud_selection_times)
        
        print(f"{policy}: Node = {avg_selection:.4f}, Alt Node = {avg_alt_selection:.4f}, Cloud = {avg_cloud_selection:.4f}")


def main():
    print("=== Fog-Cloud Task Allocation Simulator ===\n")
    
    # Initialize fog nodes
    initialize_fog_nodes()
    
    # Generate tasks
    print(f"Generating {NUM_TASKS} tasks...")
    tasks = generate_tasks(NUM_TASKS)
    print("Tasks generated.\n")
    
    # Available policies
    print("Available Policies:")
    print("1. FCFS Cooperation (GGFC)")
    print("2. FCFS No Cooperation (GGFNC)")
    print("3. Random Cooperation (GGRC)")
    print("4. Random No Cooperation (GGRNC)")
    print("5. Run All Policies")
    
    # Get user choice
    choice = int(input("\nSelect policy (1-5): "))
    
    # Run selected policy
    results = {}
    policies = []
    
    if choice in [1, 5]:
        policies.append(FCFSC)
    if choice in [2, 5]:
        policies.append(FCFSN)
    if choice in [3, 5]:
        policies.append(RANDOMC)
    if choice in [4, 5]:
        policies.append(RANDOMN)
    
    for policy in policies:
        policy_name = policy.__name__
        if policy_name == "FCFSC":
            policy_name = "FCFSCooperation"
        elif policy_name == "FCFSN":
            policy_name = "FCFSNoCooperation"
        elif policy_name == "RANDOMC":
            policy_name = "RandomCooperation"
        elif policy_name == "RANDOMN":
            policy_name = "RandomNoCooperation"
        
        results[policy_name] = run_policy(policy, tasks.copy(), FOG_NODES)
    
    if results:
        analyze_results(results)
    else:
        print("No valid policy selected!")


if __name__ == '__main__':
    main()
