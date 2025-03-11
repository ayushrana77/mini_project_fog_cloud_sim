import json
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict
from dataclasses import dataclass
from tqdm import tqdm
from typing import Dict, Any, List, Optional

# Policy Configuration
POLICIES = {
    1: "FCFS Cooperation (GGFC)",
    2: "FCFS No Cooperation (GGFNC)",
    3: "Random Cooperation (GGRC)",
    4: "Random No Cooperation (GGRNC)",
    5: "Run All Policies"
}

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
        "num_devices": 2500 # Supported IoT devices
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
        "num_devices": 5000
    }
]

CLOUD_SERVICES = [
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

class BaseGateway:
    """Base class containing common gateway functionality"""
    def __init__(self, fog_nodes, cloud_services):
        self.fog_nodes = fog_nodes
        self.cloud_services = cloud_services
        self.metrics = defaultdict(list)
        
    def reset_nodes(self):
        for fog in self.fog_nodes:
            fog.used_storage = 0
            fog.queue = []
            fog.utilization = 0
            fog.power_log = []

    def select_fog(self, task):
        """Find first available fog node"""
        for fog in self.fog_nodes:
            if (fog.ram >= task.ram and 
                fog.mips >= task.mips and 
                (fog.total_storage - fog.used_storage) >= task.size):
                return fog
        return None

    def find_alternate_fog(self, task, exclude_node):
        """Cooperation policy: Find alternative fog nodes"""
        for fog in self.fog_nodes:
            if fog is not exclude_node and (fog.ram >= task.ram and 
                fog.mips >= task.mips and 
                (fog.total_storage - fog.used_storage) >= task.size):
                return fog
        return None

    def process_cloud(self, task):
        service = next(s for s in self.cloud_services 
                      if s.ram >= task.ram and s.mips >= task.mips)
        time = service.process(task)
        self.metrics['cloud_times'].append(time)
        return time

# Policy Implementations (same as previous)

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

import numpy as np
import matplotlib.pyplot as plt

def analyze_results(results):
    """Enhanced analysis with comparative metrics using results data."""
    print("\n=== Comparative Analysis ===")
    
    # Textual Metrics
    policy_names = list(results.keys())
    
    # Metric 1: Average Processing Times
    avg_times = {
        policy: [
            np.mean(metrics['fog_times']) + np.mean(metrics['cloud_times']),
            np.mean(metrics['fog_times']) if metrics['fog_times'] else 0,
            np.mean(metrics['cloud_times']) if metrics['cloud_times'] else 0
        ]
        for policy, metrics in results.items()
    }
    
    # Metric 2: Resource Utilization (Average Power per Node)
    utilizations = {
        policy: [np.mean(node_power) for node_power in metrics['power']]
        for policy, metrics in results.items()
    }
    
    # Metric 3: Queue Delays
    queue_delays = {
        policy: np.mean(metrics['queue_delays']) if metrics['queue_delays'] else 0
        for policy, metrics in results.items()
    }
    
    # Metric 4: Task Distribution
    task_dist = {
        policy: (len(metrics['fog_times']), len(metrics['cloud_times']))
        for policy, metrics in results.items()
    }
    
    # Print Textual Metrics
    print("\n=== Average Processing Times ===")
    for policy in policy_names:
        print(f"{policy}: Total = {avg_times[policy][0]:.2f} ms, Fog = {avg_times[policy][1]:.2f} ms, Cloud = {avg_times[policy][2]:.2f} ms")
    
    print("\n=== Average Power Consumption per Node ===")
    for policy in policy_names:
        print(f"{policy}: {[f'{power:.2f} W' for power in utilizations[policy]]}")
    
    print("\n=== Average Queue Delays ===")
    for policy in policy_names:
        print(f"{policy}: {queue_delays[policy]:.2f} ms")
    
    print("\n=== Task Distribution ===")
    for policy in policy_names:
        print(f"{policy}: Fog = {task_dist[policy][0]}, Cloud = {task_dist[policy][1]}")
    
    # Prepare data for visualizations
    fig = plt.figure(figsize=(18, 10))
    axs = fig.subplots(2, 2)
    
    # Plot 1: Average Processing Times
    x = np.arange(len(policy_names))
    width = 0.25
    axs[0,0].bar(x - width, [avg_times[p][0] for p in policy_names], width, label='Total')
    axs[0,0].bar(x, [avg_times[p][1] for p in policy_names], width, label='Fog')
    axs[0,0].bar(x + width, [avg_times[p][2] for p in policy_names], width, label='Cloud')
    axs[0,0].set_title('Average Processing Times')
    axs[0,0].set_ylabel('Time (ms)')
    axs[0,0].set_xticks(x)
    axs[0,0].set_xticklabels(policy_names)
    axs[0,0].legend()
    
    # Plot 2: Power Consumption (Average per Node)
    for policy, avg_powers in utilizations.items():
        axs[0,1].bar([f"{policy} Node {i+1}" for i in range(len(avg_powers))], avg_powers, label=policy)
    axs[0,1].set_title('Average Power Consumption per Node')
    axs[0,1].set_ylabel('Power (W)')
    axs[0,1].tick_params(axis='x', rotation=45)
    axs[0,1].legend()
    
    # Plot 3: Queue Delays
    axs[1,0].bar(policy_names, queue_delays.values())
    axs[1,0].set_title('Average Queue Delays')
    axs[1,0].set_ylabel('Delay (ms)')
    
    # Plot 4: Task Distribution
    for policy, (fog, cloud) in task_dist.items():
        axs[1,1].bar(policy, fog, label='Fog' if policy == policy_names[0] else "")
        axs[1,1].bar(policy, cloud, bottom=fog, label='Cloud' if policy == policy_names[0] else "")
    axs[1,1].set_title('Task Distribution')
    axs[1,1].set_ylabel('Number of Tasks')
    axs[1,1].legend()
    
    plt.tight_layout()
    plt.show()

class FCFSCooperationGateway(BaseGateway):
    """Policy 1: FCFS with Cooperation"""
    def offload_task(self, task):
        # First check for Bulk/Large data types
        if task.data_type in ['Bulk', 'Large']:
            return self.process_cloud(task)
        
        # Try primary fog node
        fog = self.select_fog(task)
        if fog:
            fog.queue.append(task)
            processing_time = fog.process(task)
            fog.queue.remove(task)
            self.metrics['fog_times'].append(processing_time)
            self.metrics['queue_delays'].append(len(fog.queue) * 10)
            return processing_time
        
        # Try cooperation
        alt_fog = self.find_alternate_fog(task, None)
        if alt_fog:
            alt_fog.queue.append(task)
            processing_time = alt_fog.process(task)
            alt_fog.queue.remove(task)
            self.metrics['fog_times'].append(processing_time)
            self.metrics['queue_delays'].append(len(alt_fog.queue) * 10)
            return processing_time
        
        # Fallback to cloud
        return self.process_cloud(task)

class FCFSNoCooperationGateway(BaseGateway):
    """Policy 2: FCFS without Cooperation"""
    def offload_task(self, task):
        if task.data_type in ['Bulk', 'Large']:
            return self.process_cloud(task)
        
        fog = self.select_fog(task)
        if fog:
            fog.queue.append(task)
            processing_time = fog.process(task)
            fog.queue.remove(task)
            self.metrics['fog_times'].append(processing_time)
            self.metrics['queue_delays'].append(len(fog.queue) * 10)
            return processing_time
        
        return self.process_cloud(task)

class RandomCooperationGateway(BaseGateway):
    """Policy 3: Random with Cooperation"""
    def offload_task(self, task):
        if task.data_type in ['Bulk', 'Large']:
            return self.process_cloud(task)
        
        suitable_nodes = [f for f in self.fog_nodes if 
                         f.ram >= task.ram and 
                         f.mips >= task.mips and 
                         (f.total_storage - f.used_storage) >= task.size]
        
        if suitable_nodes:
            fog = random.choice(suitable_nodes)
            fog.queue.append(task)
            processing_time = fog.process(task)
            fog.queue.remove(task)
            self.metrics['fog_times'].append(processing_time)
            self.metrics['queue_delays'].append(len(fog.queue) * 10)
            return processing_time
        
        return self.process_cloud(task)

class RandomNoCooperationGateway(BaseGateway):
    """Policy 4: Random without Cooperation"""
    def offload_task(self, task):
        if task.data_type in ['Bulk', 'Large']:
            return self.process_cloud(task)
        
        suitable_nodes = [f for f in self.fog_nodes if 
                         f.ram >= task.ram and 
                         f.mips >= task.mips and 
                         (f.total_storage - f.used_storage) >= task.size]
        
        if suitable_nodes:
            fog = random.choice(suitable_nodes)
            fog.queue.append(task)
            processing_time = fog.process(task)
            fog.queue.remove(task)
            self.metrics['fog_times'].append(processing_time)
            self.metrics['queue_delays'].append(len(fog.queue) * 10)
            return processing_time
        
        return self.process_cloud(task)

def run_policy(gateway_class, tasks, fog_configs):
    """Run a simulation with specified gateway policy"""
    fog_nodes = [FogNode(cfg) for cfg in fog_configs]
    cloud_services = [CloudService(cfg) for cfg in CLOUD_SERVICES]
    gateway = gateway_class(fog_nodes, cloud_services)
    
    progress = tqdm(total=len(tasks), desc="Processing tasks", unit="task")
    for task in tasks:
        gateway.offload_task(task)
        progress.update(1)
    progress.close()
    
    # Collect metrics
    return {
        'fog_times': gateway.metrics['fog_times'],
        'cloud_times': gateway.metrics['cloud_times'],
        'queue_delays': gateway.metrics['queue_delays'],
        'power': [fog.power_log for fog in fog_nodes]
    }

def main():
    # User input handling
    print("Available Policies:")
    for key, value in POLICIES.items():
        print(f"{key}. {value}")
        
    choice = int(input("\nSelect policy (1-5): "))
    
    # Load and validate tasks
    filepath = os.path.join(os.getcwd(), 'tuple100k.json')
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found")
        exit(1)
        
    tasks = load_tasks(filepath)
    
    # Run selected policies
    results = {}
    fog_node_configs = FOG_NODES  # Preserve original config
    
    if choice in [1, 5]:
        print("\nRunning FCFS Cooperation Policy...")
        results['GGFC'] = run_policy(FCFSCooperationGateway, tasks.copy(), fog_node_configs)
        
    if choice in [2, 5]:
        print("\nRunning FCFS No Cooperation Policy...")
        results['GGFNC'] = run_policy(FCFSNoCooperationGateway, tasks.copy(), fog_node_configs)
        
    if choice in [3, 5]:
        print("\nRunning Random Cooperation Policy...")
        results['GGRC'] = run_policy(RandomCooperationGateway, tasks.copy(), fog_node_configs)
        
    if choice in [4, 5]:
        print("\nRunning Random No Cooperation Policy...")
        results['GGRNC'] = run_policy(RandomNoCooperationGateway, tasks.copy(), fog_node_configs)
    
    # Analyze results
    if results:
        analyze_results(results)
    else:
        print("No valid policy selected!")

if __name__ == '__main__':
    main()