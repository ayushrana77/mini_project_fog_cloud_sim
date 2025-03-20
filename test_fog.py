import json
import os
import numpy as np
import random
from FCFS_NO_Co import Task, FogNode, CloudService, FCFSCooperationGateway, FOG_NODES, CLOUD_SERVICES

def create_test_tasks():
    """Create a set of test tasks with various characteristics"""
    tasks = []
    
    # Task data types
    data_types = ['Small', 'Bulk', 'Video', 'Large', 'HighDef', 'Text', 'Sensor', 'IoT']
    
    # Create 20 tasks of various sizes and types
    for i in range(20):
        data_type = random.choice(data_types)
        
        # Size based on data type
        if data_type in ['Bulk', 'Large']:
            size = random.randint(500000, 4000000)
        elif data_type in ['Video', 'HighDef']:
            size = random.randint(800000, 5000000)
        else:
            size = random.randint(10000, 500000)
        
        # MIPS based on size
        mips = size / 10 + random.randint(100, 1000)
        
        # RAM based on data type
        if data_type in ['Bulk', 'Video', 'HighDef']:
            ram = random.randint(1024, 8192)
        else:
            ram = random.randint(128, 1024)
        
        task = Task(
            id=f"test-task-{i}",
            size=size,
            name=f"Test Task {i}",
            mips=mips,
            number_of_pes=4,
            ram=ram,
            bw=100,
            data_type=data_type,
            location=(random.uniform(33.5, 34.5), random.uniform(72.5, 73.5)),
            device_type="Mobile",
            arrival_time=i * 10,
            fog_candidate=True
        )
        tasks.append(task)
    
    return tasks

def test_fog_allocation():
    print("Testing fog allocation with sample tasks...")
    
    # Create fog nodes and cloud services
    fog_nodes = [FogNode(cfg) for cfg in FOG_NODES]
    cloud_services = [CloudService(cfg) for cfg in CLOUD_SERVICES]
    
    # Create gateway
    gateway = FCFSCooperationGateway(fog_nodes, cloud_services)
    
    # Create test tasks
    tasks = create_test_tasks()
    
    # Process each task
    fog_count = 0
    cloud_count = 0
    
    print("\nProcessing test tasks:")
    for i, task in enumerate(tasks):
        print(f"\nTask {i+1}/{len(tasks)}: {task.id} - Type: {task.data_type}, Size: {task.size}, MIPS: {task.mips}, RAM: {task.ram}")
        
        # Check if task is bulk data
        is_bulk = gateway.is_bulk_data(task)
        print(f"  Is bulk data: {is_bulk}")
        
        # Offload task
        result = gateway.offload_task(task)
        
        # Update counts
        if result == 0:
            fog_count += 1
        else:
            cloud_count += 1
    
    # Print summary
    total = fog_count + cloud_count
    fog_percent = (fog_count / total) * 100 if total > 0 else 0
    cloud_percent = (cloud_count / total) * 100 if total > 0 else 0
    
    print("\n=== Test Results ===")
    print(f"Total Tasks: {total}")
    print(f"Fog Tasks: {fog_count} ({fog_percent:.1f}%)")
    print(f"Cloud Tasks: {cloud_count} ({cloud_percent:.1f}%)")
    
    # Print fog node statistics
    print("\nFog Node Statistics:")
    for fog in fog_nodes:
        print(f"{fog.name}: Processed {fog.total_processed} tasks, Utilization: {fog.utilization:.2f}%")
    
    print("\nDevice Commitments:")
    for node_name, batches in gateway.device_commitments.items():
        print(f"  {node_name}: {sum(batches.values())} devices committed")

if __name__ == "__main__":
    test_fog_allocation() 