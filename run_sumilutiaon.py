import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from dataclasses import dataclass
from tqdm import tqdm

# Configuration
FOG_NODES = [
    {
        "name": "Edge-Fog-01",
        "down_bw": 300,    # Download bandwidth in Mbps
        "up_bw": 200,      # Upload bandwidth in Mbps
        "mips": 2500,      # Million Instructions Per Second
        "num_pes": 4,      # Number of Processing Elements (Cores)
        "ram": 4096,       # RAM in MB
        "size": 10000,     # Storage size in MB
        "storage": 50000,  # Total storage capacity in MB
        "num_devices": 25  # Number of supported IoT devices
    },
    {
        "name": "Edge-Fog-02",
        "down_bw": 500,
        "up_bw": 300,
        "mips": 4000,
        "num_pes": 8,
        "ram": 8192,
        "size": 20000,
        "storage": 100000,
        "num_devices": 50
    }
]

CLOUD_SERVICES = [
    # USA Data Center
    {
        "name": "USA-Service1",
        "ram": 8192,
        "mips": 9000,
        "bw": 500,
        "size": 5000,
        "cpu": 1
    },
    {
        "name": "USA-Service2",
        "ram": 16384,
        "mips": 18000,
        "bw": 500,
        "size": 10000,
        "cpu": 2
    },
    {
        "name": "USA-Service3",
        "ram": 32768,
        "mips": 27000,
        "bw": 500,
        "size": 15000,
        "cpu": 4
    },
    # Singapore Data Center
    {
        "name": "Singapore-Service1",
        "ram": 8192,
        "mips": 9000,
        "bw": 500,
        "size": 5000,
        "cpu": 1
    },
    {
        "name": "Singapore-Service2",
        "ram": 16384,
        "mips": 18000,
        "bw": 500,
        "size": 10000,
        "cpu": 2
    },
    {
        "name": "Singapore-Service3",
        "ram": 32768,
        "mips": 27000,
        "bw": 500,
        "size": 15000,
        "cpu": 4
    }
]

TRANSMISSION_LATENCY = 0.5  # ms
BATCH_SIZE = 5000  # Not used in sequential processing; kept for legacy reference

@dataclass
class Task:
    id: str
    size: int
    mips: float
    ram: int
    data_type: str
    location: tuple
    device_type: str

class FogNode:
    def __init__(self, config):
        self.name = config['name']
        self.down_bw = config['down_bw']
        self.up_bw = config['up_bw']
        self.mips = config['mips']
        self.num_pes = config['num_pes']
        self.ram = config['ram']
        self.storage_size = config['size']
        self.total_storage = config['storage']
        self.num_devices = config['num_devices']
        self.queue = []
        self.utilization = 0
        self.power_log = []
        self.used_storage = 0

    def calculate_power(self):
        return 100 + (self.utilization * 50)

    def process(self, task):
        # Calculate transmission time using download bandwidth
        transmission_time = (task.size / self.down_bw) * 1000
        processing_time = (task.mips / self.mips) * 1000
        queue_delay = len(self.queue) * 10
        
        # Update resource usage
        self.utilization = min(100, self.utilization + (processing_time / 1000))
        self.used_storage += task.size
        self.power_log.append(self.calculate_power())
        
        return transmission_time + processing_time + queue_delay

class CloudService:
    def __init__(self, config):
        self.name = config['name']
        self.ram = config['ram']
        self.mips = config['mips']
        self.bw = config['bw']
        self.size = config['size']
        self.cpu = config['cpu']

    def process(self, task):
        transmission_time = (task.size / self.bw) * 1000
        processing_time = (task.mips / self.mips) * 1000
        return transmission_time + processing_time + TRANSMISSION_LATENCY

class GlobalGateway:
    def __init__(self, fog_nodes, cloud_services):
        self.fog_nodes = fog_nodes
        self.cloud_services = cloud_services
        self.metrics = defaultdict(list)

    def select_fog(self, task, cooperation=True):
        # Check storage capacity and resource availability first
        suitable = [f for f in self.fog_nodes 
                    if f.ram >= task.ram and 
                    f.mips >= task.mips and 
                    (f.total_storage - f.used_storage) >= task.size]
        
        if suitable:
            return min(suitable, key=lambda x: x.utilization)
            
        # If cooperation is allowed, try alternate selection
        if cooperation:
            for fog in reversed(self.fog_nodes):
                if (fog.ram >= task.ram and 
                    fog.mips >= task.mips and 
                    (fog.total_storage - fog.used_storage) >= task.size):
                    return fog
        return None

    def offload_task(self, task, policy):
        # Offload to cloud service if Global Gateway is enabled and the data type is large
        if policy['gg_enabled'] and task.data_type in ['Bulk', 'Large']:
            service = next(s for s in self.cloud_services if s.ram >= task.ram and s.mips >= task.mips)
            time = service.process(task)
            self.metrics['cloud_times'].append(time)
            return time
            
        # Try to select a suitable fog node
        fog = self.select_fog(task, policy['cooperation'])
        if fog:
            fog.queue.append(task)
            processing_time = fog.process(task)
            fog.queue.remove(task)
            self.metrics['fog_times'].append(processing_time)
            self.metrics['queue_delays'].append(len(fog.queue) * 10)
            return processing_time
            
        # If no fog node is suitable, offload to cloud service
        service = next(s for s in self.cloud_services if s.ram >= task.ram and s.mips >= task.mips)
        time = service.process(task)
        self.metrics['cloud_times'].append(time)
        return time

def validate_json(filepath):
    """Validate JSON file structure."""
    try:
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
            
        if not isinstance(data, list):
            raise ValueError("Root element should be an array")
            
        required_fields = {'ID', 'Size', 'MIPS', 'RAM', 'DataType', 
                           'GeoLocation', 'DeviceType'}
        for i, item in enumerate(data):
            missing = required_fields - set(item.keys())
            if missing:
                raise KeyError(f"Item {i} missing fields: {missing}")
                
            if not isinstance(item['GeoLocation'], dict):
                raise TypeError(f"Item {i}: GeoLocation should be an object")
                
            if 'latitude' not in item['GeoLocation'] or 'longitude' not in item['GeoLocation']:
                raise KeyError(f"Item {i}: GeoLocation missing coordinates")
                
        print("JSON validation successful!")
        return True
        
    except Exception as e:
        print(f"Validation failed: {str(e)}")
        return False

def load_tasks(filepath):
    """Load tasks with comprehensive error handling."""
    if not validate_json(filepath):
        exit(1)
        
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        data = json.load(f)
        
    tasks = []
    for item in data:
        tasks.append(Task(
            id=item['ID'],
            size=item['Size'],
            mips=float(item['MIPS']),
            ram=item['RAM'],
            data_type=item['DataType'],
            location=(
                float(item['GeoLocation']['latitude']),
                float(item['GeoLocation']['longitude'])
            ),
            device_type=item['DeviceType']
        ))
    return tasks

def analyze_results(metrics, fog_nodes):
    """Print and plot the simulation results."""
    print("\n=== Simulation Results ===")
    total_tasks = len(metrics['fog_times']) + len(metrics['cloud_times'])
    print(f"Total tasks processed: {total_tasks:,}")
    print(f"Average fog time: {np.mean(metrics['fog_times']):.2f}ms")
    print(f"Average cloud time: {np.mean(metrics['cloud_times']):.2f}ms")
    print(f"Maximum queue delay: {np.max(metrics['queue_delays'])}ms")
    
    plt.figure(figsize=(15, 5))
    
    # Processing times distribution
    plt.subplot(1, 3, 1)
    plt.hist(metrics['fog_times'], bins=50, alpha=0.5, label='Fog')
    plt.hist(metrics['cloud_times'], bins=50, alpha=0.5, label='Cloud')
    plt.title('Processing Time Distribution')
    plt.xlabel('Time (ms)')
    plt.legend()
    
    # Fog nodes power consumption
    plt.subplot(1, 3, 2)
    for fog in fog_nodes:
        plt.plot(fog.power_log, label=fog.name)
    plt.title('Fog Node Power Consumption')
    plt.xlabel('Processing Steps')
    plt.ylabel('Power (W)')
    plt.legend()
    
    # Queue delay progression
    plt.subplot(1, 3, 3)
    plt.plot(metrics['queue_delays'], alpha=0.7)
    plt.title('Queue Delay Progression')
    plt.xlabel('Task Number')
    plt.ylabel('Delay (ms)')
    
    plt.tight_layout()
    plt.show()

def main():
    # Initialize infrastructure
    fog_nodes = [FogNode(cfg) for cfg in FOG_NODES]
    cloud_services = [CloudService(cfg) for cfg in CLOUD_SERVICES]
    gateway = GlobalGateway(fog_nodes, cloud_services)
    
    # Load and validate tasks from JSON file
    filepath = os.path.join(os.getcwd(), 'tuples.json')
    print(f"Loading tasks from: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        exit(1)
        
    tasks = load_tasks(filepath)
    
    # Simulation policy
    policy = {
        'gg_enabled': True,
        'cooperation': True,
        'queue_policy': 'FCFS'
    }
    
    # Process tasks sequentially (or in batches if desired)
    progress = tqdm(total=len(tasks), desc="Processing tasks", unit="task")
    for task in tasks:
        gateway.offload_task(task, policy)
        progress.update(1)
    progress.close()
    
    # Add power metrics from each fog node
    gateway.metrics['power'] = []
    for fog in fog_nodes:
        gateway.metrics['power'].extend(fog.power_log)
    
    # Analyze and display simulation results
    analyze_results(gateway.metrics, fog_nodes)

if __name__ == '__main__':
    main()
