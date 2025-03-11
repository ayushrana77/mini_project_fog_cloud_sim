import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from dataclasses import dataclass
from tqdm import tqdm
from typing import Dict, Any, List, Optional

# Infrastructure Configuration
FOG_NODES = [
    {
        "name": "Edge-Fog-01",
        "down_bw": 300,    # Mbps
        "up_bw": 200,      # Mbps
        "mips": 2500,      # Million Instructions Per Second
        "num_pes": 4,      # Processing Cores
        "ram": 4096,       # MB
        "size": 10000,     # Storage size in MB
        "storage": 50000,  # Total storage capacity in MB
        "num_devices": 25  # Supported IoT devices
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
    # Singapore Data Center
    {
        "name": "Singapore-Service1",
        "ram": 8192,
        "mips": 9000,
        "bw": 500,
        "size": 5000,
        "cpu": 1
    }
]

TRANSMISSION_LATENCY = 0.5  # ms

@dataclass
class Task:
    id: str
    size: int
    name: str
    mips: float
    number_of_pes: int
    ram: int
    bw: int
    data_type: str
    location: tuple
    device_type: str
    cloudlet_scheduler: Dict[str, Any] = None
    current_allocated_size: int = 0
    current_allocated_ram: int = 0
    current_allocated_bw: int = 0
    current_allocated_mips: Optional[List[float]] = None
    being_instantiated: bool = True
    data_percentage: int = 100
    is_reversed: bool = False
    is_server_found: bool = False
    is_cloud_served: bool = False
    is_served: bool = False
    queue_delay: float = 0.0
    internal_processing_time: float = 0.0
    tuple_times: List[float] = None
    fog_level_served: int = 0
    is_served_by_fc_cloud: bool = False
    creation_time: str = ""
    queue_wait: Optional[float] = None
    temperature: float = 0.0

class FogNode:
    def __init__(self, config):
        self.name = config['name']
        self.down_bw = config['down_bw']
        self.up_bw = config['up_bw']
        self.mips = config['mips']
        self.ram = config['ram']
        self.total_storage = config['storage']
        self.used_storage = 0
        self.queue = []
        self.utilization = 0
        self.power_log = []

    def calculate_power(self):
        return 100 + (self.utilization * 50)

    def process(self, task):
        transmission_time = (task.size / self.down_bw) * 1000
        processing_time = (task.mips / self.mips) * 1000
        queue_delay = len(self.queue) * 10
        
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

    def process(self, task):
        transmission_time = (task.size / self.bw) * 1000
        processing_time = (task.mips / self.mips) * 1000
        return transmission_time + processing_time + TRANSMISSION_LATENCY

class GlobalGateway:
    def __init__(self, fog_nodes, cloud_services):
        self.fog_nodes = fog_nodes
        self.cloud_services = cloud_services
        self.task_queue = []
        self.metrics = defaultdict(list)

    def select_fog(self, task):
        """Select first suitable fog node in configuration order"""
        for fog in self.fog_nodes:
            if (fog.ram >= task.ram and 
                fog.mips >= task.mips and 
                (fog.total_storage - fog.used_storage) >= task.size):
                return fog
        return None

    def offload_task(self, task):
        """Implements Algorithm 2 logic"""
        # Add to FCFS queue
        self.task_queue.append(task)
        
        # Process in strict FCFS order
        current_task = self.task_queue.pop(0)
        
        # Data type check first
        if current_task.data_type in ['Bulk', 'Large']:
            service = next(s for s in self.cloud_services 
                          if s.ram >= current_task.ram and s.mips >= current_task.mips)
            time = service.process(current_task)
            self.metrics['cloud_times'].append(time)
            return time
        
        # Try fog processing
        fog = self.select_fog(current_task)
        if fog:
            fog.queue.append(current_task)
            processing_time = fog.process(current_task)
            fog.queue.remove(current_task)
            self.metrics['fog_times'].append(processing_time)
            self.metrics['queue_delays'].append(len(fog.queue) * 10)
            return processing_time
        
        # Fallback to cloud
        service = next(s for s in self.cloud_services 
                      if s.ram >= current_task.ram and s.mips >= current_task.mips)
        time = service.process(current_task)
        self.metrics['cloud_times'].append(time)
        return time

def validate_json(filepath):
    """Validate JSON file structure with all required fields."""
    try:
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
            
        if not isinstance(data, list):
            raise ValueError("Root element should be an array")
            
        required_fields = {
            'ID', 'Size', 'Name', 'MIPS', 'NumberOfPes',
            'RAM', 'BW', 'DataType', 'GeoLocation', 'DeviceType'
        }
        
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
    """Load tasks with all attributes."""
    if not validate_json(filepath):
        exit(1)
        
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        data = json.load(f)
        
    tasks = []
    for item in data:
        tasks.append(Task(
            id=item['ID'],
            size=item['Size'],
            name=item['Name'],
            mips=float(item['MIPS']),
            number_of_pes=item['NumberOfPes'],
            ram=item['RAM'],
            bw=item['BW'],
            data_type=item['DataType'],
            location=(
                float(item['GeoLocation']['latitude']),
                float(item['GeoLocation']['longitude'])
            ),
            device_type=item['DeviceType'],
            cloudlet_scheduler=item.get('CloudletScheduler'),
            current_allocated_size=item.get('CurrentAllocatedSize', 0),
            current_allocated_ram=item.get('CurrentAllocatedRam', 0),
            current_allocated_bw=item.get('CurrentAllocatedBw', 0),
            current_allocated_mips=item.get('CurrentAllocatedMips'),
            being_instantiated=item.get('BeingInstantiated', True),
            data_percentage=item.get('DataPercentage', 100),
            is_reversed=item.get('IsReversed', False),
            is_server_found=item.get('IsServerFound', False),
            is_cloud_served=item.get('IsCloudServed', False),
            is_served=item.get('IsServed', False),
            queue_delay=item.get('QueueDelay', 0.0),
            internal_processing_time=item.get('InternalProcessingTime', 0.0),
            tuple_times=item.get('TupleTimes', []),
            fog_level_served=item.get('FogLevelServed', 0),
            is_served_by_fc_cloud=item.get('IsServedByFC_Cloud', False),
            creation_time=item.get('CreationTime', ""),
            queue_wait=item.get('QueueWait'),
            temperature=item.get('Temperature', 0.0)
        ))
    return tasks

def analyze_results(metrics, fog_nodes):
    print("\n=== Simulation Results ===")
    print(f"Total tasks: {len(metrics['fog_times']) + len(metrics['cloud_times']):,}")
    print(f"Avg Fog Time: {np.mean(metrics['fog_times']):.2f}ms")
    print(f"Avg Cloud Time: {np.mean(metrics['cloud_times']):.2f}ms")
    print(f"Max Queue Delay: {np.max(metrics['queue_delays'])}ms")
    
    plt.figure(figsize=(15, 5))
    
    # Processing Times
    plt.subplot(1, 3, 1)
    plt.hist(metrics['fog_times'], bins=50, alpha=0.5, label='Fog')
    plt.hist(metrics['cloud_times'], bins=50, alpha=0.5, label='Cloud')
    plt.title('Processing Time Distribution')
    plt.xlabel('Time (ms)')
    plt.legend()
    
    # Power Consumption
    plt.subplot(1, 3, 2)
    for fog in fog_nodes:
        plt.plot(fog.power_log, label=fog.name)
    plt.title('Power Consumption')
    plt.xlabel('Processing Steps')
    plt.ylabel('Power (W)')
    plt.legend()
    
    # Queue Delays
    plt.subplot(1, 3, 3)
    plt.plot(metrics['queue_delays'], alpha=0.7)
    plt.title('Queue Delay Progression')
    plt.xlabel('Task Number')
    plt.ylabel('Delay (ms)')
    
    plt.tight_layout()
    plt.show()

def main():
    fog_nodes = [FogNode(cfg) for cfg in FOG_NODES]
    cloud_services = [CloudService(cfg) for cfg in CLOUD_SERVICES]
    gateway = GlobalGateway(fog_nodes, cloud_services)
    
    filepath = os.path.join(os.getcwd(), 'tuple100k.json')
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found")
        exit(1)
        
    tasks = load_tasks(filepath)
    
    progress = tqdm(total=len(tasks), desc="Processing tasks")
    for task in tasks:
        gateway.offload_task(task)
        progress.update(1)
    progress.close()
    
    analyze_results(gateway.metrics, fog_nodes)

if __name__ == '__main__':
    main()