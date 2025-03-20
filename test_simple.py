import json
import os
import numpy as np
import random
from math import radians, sin, cos, sqrt, atan2
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple

# Constants for Earth radius and simulation
EARTH_RADIUS_KM = 6371
NODE_CHECK_DELAY = 0.1  # ms
CLOUD_SELECTION_DELAY = 2.5  # ms

# Simple Task class
@dataclass
class Task:
    id: str
    size: int
    data_type: str
    mips: float
    ram: int
    location: Tuple[float, float]
    fog_candidate: bool = True

# Simple FogNode class
class FogNode:
    def __init__(self, name, location, mips, ram, storage, devices):
        self.name = name
        self.location = location
        self.mips = mips
        self.ram = ram
        self.total_storage = storage
        self.used_storage = 0
        self.available_ram = ram
        self.available_mips = mips
        self.num_devices = devices
        self.total_processed = 0
        self.utilization = 0

    def process(self, task):
        """Process task and update resources"""
        self.total_processed += 1
        
        # Update resources
        self.available_ram -= min(task.ram, self.available_ram)
        self.available_mips -= min(task.mips, self.available_mips)
        self.used_storage += min(task.size, self.total_storage - self.used_storage)
        
        # Update utilization
        self.utilization = min(100, self.utilization + (task.size / self.total_storage) * 10)
        
        return True

# Simplified Gateway
class Gateway:
    def __init__(self, fog_nodes):
        self.fog_nodes = fog_nodes
        self.device_commitments = {node.name: 0 for node in fog_nodes}
        self.sim_clock = 0.0
        
    def is_bulk_data(self, task):
        """Check if task involves bulk data"""
        if task.data_type in ['Bulk', 'Large']:
            return task.size > 3000000
        elif task.data_type in ['Video', 'HighDef']:
            return task.size > 4000000
        return False
        
    def is_fog_available(self, fog, task):
        """Check if fog node can accept task"""
        if not task.fog_candidate:
            return False
            
        # Check device commitments
        if self.device_commitments[fog.name] >= fog.num_devices * 0.9:
            return False
            
        # Check resources
        if (fog.available_ram < task.ram * 0.8 or
            fog.available_mips < task.mips * 0.8 or
            (fog.total_storage - fog.used_storage) < task.size * 0.8):
            return False
            
        return True
        
    def process_task(self, task):
        """Process a single task"""
        # Check if task is bulk data
        if self.is_bulk_data(task):
            print(f"Task {task.id} is bulk data, sending to cloud")
            return "cloud"
            
        # Try to find an available fog node
        for fog in self.fog_nodes:
            self.sim_clock += NODE_CHECK_DELAY
            
            if self.is_fog_available(fog, task):
                # Process task and commit device
                fog.process(task)
                self.device_commitments[fog.name] += 1
                
                print(f"Task {task.id} processed by fog node {fog.name}")
                return "fog"
                
        # If no fog node available, send to cloud
        print(f"Task {task.id} sent to cloud (no available fog nodes)")
        return "cloud"

# Create test data
def run_test():
    print("Running simplified allocation test...")
    
    # Create fog nodes
    fog_nodes = [
        FogNode("Fog-1", (33.72, 72.85), 200000, 327680, 800000, 3500),
        FogNode("Fog-2", (34.12, 73.25), 250000, 491520, 1000000, 3500),
        FogNode("Fog-3", (33.90, 73.05), 220000, 409600, 900000, 3500),
        FogNode("Fog-4", (33.80, 73.15), 240000, 450560, 950000, 3500)
    ]
    
    # Create gateway
    gateway = Gateway(fog_nodes)
    
    # Create test tasks - half small and half large
    tasks = []
    
    # Small tasks
    for i in range(10):
        tasks.append(Task(
            id=f"small-{i}",
            size=random.randint(10000, 500000),
            data_type=random.choice(['Small', 'Text', 'Sensor', 'IoT']),
            mips=random.randint(1000, 5000),
            ram=random.randint(128, 1024),
            location=(random.uniform(33.5, 34.5), random.uniform(72.5, 73.5))
        ))
    
    # Large tasks
    for i in range(10):
        tasks.append(Task(
            id=f"large-{i}",
            size=random.randint(1000000, 5000000),
            data_type=random.choice(['Bulk', 'Large', 'Video', 'HighDef']),
            mips=random.randint(5000, 50000),
            ram=random.randint(1024, 8192),
            location=(random.uniform(33.5, 34.5), random.uniform(72.5, 73.5))
        ))
    
    # Shuffle tasks
    random.shuffle(tasks)
    
    # Process tasks
    results = {
        'fog': 0,
        'cloud': 0
    }
    
    for i, task in enumerate(tasks):
        print(f"\nTask {i+1}/{len(tasks)}: {task.id} - Type: {task.data_type}, Size: {task.size}")
        
        # Check if task is bulk data
        is_bulk = gateway.is_bulk_data(task)
        print(f"  Is bulk data: {is_bulk}")
        
        # Process task
        result = gateway.process_task(task)
        results[result] += 1
    
    # Print summary
    print("\n=== Test Results ===")
    total = results['fog'] + results['cloud']
    fog_percent = (results['fog'] / total) * 100
    cloud_percent = (results['cloud'] / total) * 100
    
    print(f"Total Tasks: {total}")
    print(f"Fog Tasks: {results['fog']} ({fog_percent:.1f}%)")
    print(f"Cloud Tasks: {results['cloud']} ({cloud_percent:.1f}%)")
    
    # Print fog node statistics
    print("\nFog Node Statistics:")
    for fog in fog_nodes:
        print(f"{fog.name}: Processed {fog.total_processed} tasks, Utilization: {fog.utilization:.2f}%")
    
    # Print device commitments
    print("\nDevice Commitments:")
    for node_name, count in gateway.device_commitments.items():
        print(f"  {node_name}: {count} devices committed")

if __name__ == "__main__":
    run_test() 