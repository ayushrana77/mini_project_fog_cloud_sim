import json
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict
from dataclasses import dataclass
from tqdm import tqdm
from typing import Dict, List, Optional, Any
from math import radians, sin, cos, sqrt, atan2

# ========== Configuration Section ==========
POLICIES = {
    1: "FCFS Cooperation (GGFC)",
    2: "FCFS No Cooperation (GGFNC)",
    3: "Random Cooperation (GGRC)",
    4: "Random No Cooperation (GGRNC)",
    5: "Run All Policies"
}

FOG_NODES = [
    {
        "name": "Edge-Fog-01",
        "location": (33.72, 72.85),
        "down_bw": 300,
        "up_bw": 200,
        "mips": 2500,
        "num_pes": 4,
        "ram": 4096,
        "storage": 50000,
        "num_devices": 2500
    },
    {
        "name": "Edge-Fog-02",
        "location": (33.65, 73.10),
        "down_bw": 500,
        "up_bw": 300,
        "mips": 4000,
        "num_pes": 8,
        "ram": 8192,
        "storage": 100000,
        "num_devices": 5000
    }
]

CLOUD_SERVICES = [
    {
        "name": "USA-Service1",
        "location": (37.09, -95.71),
        "ram": 8192,
        "mips": 9000,
        "bw": 500
    },
    {
        "name": "Singapore-Service1",
        "location": (1.35, 103.82),
        "ram": 8192,
        "mips": 9000,
        "bw": 500
    }
]

TRANSMISSION_LATENCY = 0.2  # ms
EARTH_RADIUS_KM = 6371
MAX_SIMULATION_TIME = 1000  # seconds for task arrival generation

# ========== Data Models ==========
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
    arrival_time: float = 0.0  # Added arrival time
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
        self.location = config['location']
        self.down_bw = config['down_bw']
        self.up_bw = config['up_bw']
        self.mips = config['mips']
        self.num_pes = config['num_pes']
        self.ram = config['ram']
        self.total_storage = config['storage']
        self.used_storage = 0
        self.queue = []
        self.utilization = 0
        self.power_log = []
        self.busy_until = 0.0
        self.num_devices = config['num_devices']
        self.available_ram = config['ram']  # Track available RAM
        self.available_mips = config['mips']  # Track available MIPS

    def calculate_power(self):
        return 100 + (self.utilization * 50)

    def process(self, task, arrival_time):
        distance = haversine(self.location, task.location)
        geo_latency = distance * 0.02  # 0.02ms per km
        
        transmission_time = (task.size / self.down_bw) * 1000 + geo_latency
        processing_time = (task.mips / self.mips) * 1000
        start_time = max(arrival_time, self.busy_until)
        queue_delay = start_time - arrival_time
        
        # Update resource availability during processing
        self.available_ram -= task.ram
        self.available_mips -= task.mips
        self.used_storage += task.size
        
        self.busy_until = start_time + processing_time
        self.utilization = min(100, self.utilization + (processing_time / 1000))
        self.power_log.append(self.calculate_power())
        
        # Release resources after processing
        completion_time = self.busy_until
        self.available_ram += task.ram
        self.available_mips += task.mips
        self.used_storage -= task.size
        
        return queue_delay, transmission_time + processing_time, completion_time

class CloudService:
    def __init__(self, config):
        self.name = config['name']
        self.location = config['location']
        self.ram = config['ram']
        self.mips = config['mips']
        self.bw = config['bw']

    def process(self, task):
        distance = haversine(self.location, task.location)
        geo_latency = distance * 0.1  # 0.1ms per km for cloud
        
        transmission_time = (task.size / self.bw) * 1000 + geo_latency
        processing_time = (task.mips / self.mips) * 1000
        return transmission_time + processing_time + TRANSMISSION_LATENCY

# ========== Helper Functions ==========
def haversine(loc1, loc2):
    lat1, lon1 = radians(loc1[0]), radians(loc1[1])
    lat2, lon2 = radians(loc2[0]), radians(loc2[1])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return EARTH_RADIUS_KM * c

# ========== Gateway Implementations ==========
class BaseGateway:
    def __init__(self, fog_nodes, cloud_services):
        self.fog_nodes = fog_nodes
        self.cloud_services = cloud_services
        self.metrics = defaultdict(list)
        self.sim_clock = 0.0

    def reset_nodes(self):
        for fog in self.fog_nodes:
            fog.used_storage = 0
            fog.queue = []
            fog.utilization = 0
            fog.power_log = []
            fog.busy_until = 0.0
            fog.available_ram = fog.ram
            fog.available_mips = fog.mips

    def select_fog(self, task):
        start_time = self.sim_clock
        for fog in self.fog_nodes:
            if (fog.available_ram >= task.ram and 
                fog.available_mips >= task.mips and 
                (fog.total_storage - fog.used_storage) >= task.size):
                self.metrics['node_selection_time'].append(self.sim_clock - start_time)
                return fog
        self.metrics['node_selection_time'].append(self.sim_clock - start_time)
        return None

    def find_alternate_fog(self, task, exclude_node):
        start_time = self.sim_clock
        for fog in self.fog_nodes:
            if fog is not exclude_node and (fog.available_ram >= task.ram and 
                fog.available_mips >= task.mips and 
                (fog.total_storage - fog.used_storage) >= task.size):
                self.metrics['alt_node_selection_time'].append(self.sim_clock - start_time)
                return fog
        self.metrics['alt_node_selection_time'].append(self.sim_clock - start_time)
        return None

    def process_cloud(self, task):
        start_time = self.sim_clock
        service = min(self.cloud_services, 
                     key=lambda s: haversine(s.location, task.location))
        time = service.process(task)
        self.metrics['cloud_times'].append(time)
        self.metrics['cloud_selection_time'].append(self.sim_clock - start_time)
        return time

class FCFSCooperationGateway(BaseGateway):
    def offload_task(self, task):
        if task.data_type in ['Bulk', 'Large']:
            return self.process_cloud(task)
        
        fog = self.select_fog(task)
        if fog:
            q_delay, p_time, completion_time = fog.process(task, task.arrival_time)
            self.sim_clock = max(self.sim_clock, completion_time)
            self.metrics['fog_times'].append(p_time)
            self.metrics['queue_delays'].append(q_delay)
            return p_time
        
        alt_fog = self.find_alternate_fog(task, None)
        if alt_fog:
            q_delay, p_time, completion_time = alt_fog.process(task, task.arrival_time)
            self.sim_clock = max(self.sim_clock, completion_time)
            self.metrics['fog_times'].append(p_time)
            self.metrics['queue_delays'].append(q_delay)
            return p_time
        
        return self.process_cloud(task)

class FCFSNoCooperationGateway(BaseGateway):
    def offload_task(self, task):
        if task.data_type in ['Bulk', 'Large']:
            return self.process_cloud(task)
        
        fog = self.select_fog(task)
        if fog:
            q_delay, p_time, completion_time = fog.process(task, task.arrival_time)
            self.sim_clock = max(self.sim_clock, completion_time)
            self.metrics['fog_times'].append(p_time)
            self.metrics['queue_delays'].append(q_delay)
            return p_time
        
        return self.process_cloud(task)

class RandomCooperationGateway(BaseGateway):
    def offload_task(self, task):
        if task.data_type in ['Bulk', 'Large']:
            return self.process_cloud(task)
        
        suitable = [f for f in self.fog_nodes if 
                   f.available_ram >= task.ram and 
                   f.available_mips >= task.mips and 
                   (f.total_storage - f.used_storage) >= task.size]
        
        if suitable:
            fog = random.choice(suitable)
            q_delay, p_time, completion_time = fog.process(task, task.arrival_time)
            self.sim_clock = max(self.sim_clock, completion_time)
            self.metrics['fog_times'].append(p_time)
            self.metrics['queue_delays'].append(q_delay)
            return p_time
        
        alt_suitable = [f for f in self.fog_nodes if f not in suitable]
        if alt_suitable:
            fog = random.choice(alt_suitable)
            q_delay, p_time, completion_time = fog.process(task, task.arrival_time)
            self.sim_clock = max(self.sim_clock, completion_time)
            self.metrics['fog_times'].append(p_time)
            self.metrics['queue_delays'].append(q_delay)
            return p_time
        
        return self.process_cloud(task)

class RandomNoCooperationGateway(BaseGateway):
    def offload_task(self, task):
        if task.data_type in ['Bulk', 'Large']:
            return self.process_cloud(task)
        
        suitable = [f for f in self.fog_nodes if 
                   f.available_ram >= task.ram and 
                   f.available_mips >= task.mips and 
                   (f.total_storage - f.used_storage) >= task.size]
        
        if suitable:
            fog = random.choice(suitable)
            q_delay, p_time, completion_time = fog.process(task, task.arrival_time)
            self.sim_clock = max(self.sim_clock, completion_time)
            self.metrics['fog_times'].append(p_time)
            self.metrics['queue_delays'].append(q_delay)
            return p_time
        
        return self.process_cloud(task)

# ========== Core Functions ==========
def validate_json(filepath):
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
                
        return True
        
    except Exception as e:
        print(f"Validation failed: {str(e)}")
        return False

def load_tasks(filepath):
    if not validate_json(filepath):
        exit(1)
        
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        data = json.load(f)
        
    tasks = []
    for item in data:
        arrival_time = random.uniform(0, MAX_SIMULATION_TIME)
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
            arrival_time=arrival_time
        ))
    return tasks

def analyze_results(results):
    print("\n=== Comparative Analysis ===")
    
    policy_names = list(results.keys())
    metrics = {
        'avg_total': [],
        'avg_fog': [],
        'avg_cloud': [],
        'power': [],
        'queue_delays': [],
        'task_dist_fog': [],
        'task_dist_cloud': [],
        'avg_node_select': [],
        'avg_alt_node_select': [],
        'avg_cloud_select': []
    }
    
    for policy in policy_names:
        data = results[policy]
        all_times = data['fog_times'] + data['cloud_times']
        metrics['avg_total'].append(np.mean(all_times))
        metrics['avg_fog'].append(np.mean(data['fog_times']) if data['fog_times'] else 0)
        metrics['avg_cloud'].append(np.mean(data['cloud_times']) if data['cloud_times'] else 0)
        metrics['power'].append([np.mean(node) for node in data['power']])
        metrics['queue_delays'].append(np.mean(data['queue_delays']) if data['queue_delays'] else 0)
        metrics['task_dist_fog'].append(len(data['fog_times']))
        metrics['task_dist_cloud'].append(len(data['cloud_times']))
        metrics['avg_node_select'].append(np.mean(data['node_selection_time']) if data['node_selection_time'] else 0)
        metrics['avg_alt_node_select'].append(np.mean(data['alt_node_selection_time']) if data['alt_node_selection_time'] else 0)
        metrics['avg_cloud_select'].append(np.mean(data['cloud_selection_time']) if data['cloud_selection_time'] else 0)
    
    print("\n=== Average Processing Times (ms) ===")
    for policy, total, fog, cloud in zip(policy_names, metrics['avg_total'], 
                                       metrics['avg_fog'], metrics['avg_cloud']):
        print(f"{policy}: Total = {total:.2f}, Fog = {fog:.2f}, Cloud = {cloud:.2f}")
    
    print("\n=== Average Power Consumption per Node (W) ===")
    for policy, power in zip(policy_names, metrics['power']):
        print(f"{policy}: {[f'{p:.2f}' for p in power]}")
    
    print("\n=== Average Queue Delays (ms) ===")
    for policy, delay in zip(policy_names, metrics['queue_delays']):
        print(f"{policy}: {delay:.2f}")
    
    print("\n=== Task Distribution ===")
    for policy, fog, cloud in zip(policy_names, metrics['task_dist_fog'], metrics['task_dist_cloud']):
        print(f"{policy}: Fog = {fog}, Cloud = {cloud}")
    
    print("\n=== Average Selection Times (ms) ===")
    for policy, node, alt_node, cloud in zip(policy_names, metrics['avg_node_select'], 
                                            metrics['avg_alt_node_select'], metrics['avg_cloud_select']):
        print(f"{policy}: Node = {node:.4f}, Alt Node = {alt_node:.4f}, Cloud = {cloud:.4f}")

def run_policy(gateway_class, tasks, fog_configs):
    fog_nodes = [FogNode(cfg) for cfg in fog_configs]
    cloud_services = [CloudService(cfg) for cfg in CLOUD_SERVICES]
    gateway = gateway_class(fog_nodes, cloud_services)
    
    # Sort tasks by arrival time
    sorted_tasks = sorted(tasks, key=lambda t: t.arrival_time)
    
    with tqdm(total=len(sorted_tasks), desc=f"Processing {gateway_class.__name__}") as progress:
        for task in sorted_tasks:
            # Update sim_clock to current task's arrival time
            gateway.sim_clock = max(gateway.sim_clock, task.arrival_time)
            processing_time = gateway.offload_task(task)
            progress.update(1)
    
    return {
        'fog_times': gateway.metrics['fog_times'],
        'cloud_times': gateway.metrics['cloud_times'],
        'queue_delays': gateway.metrics['queue_delays'],
        'power': [fog.power_log for fog in fog_nodes],
        'node_selection_time': gateway.metrics.get('node_selection_time', []),
        'alt_node_selection_time': gateway.metrics.get('alt_node_selection_time', []),
        'cloud_selection_time': gateway.metrics.get('cloud_selection_time', [])
    }

def main():
    print("Available Policies:")
    for key, value in POLICIES.items():
        print(f"{key}. {value}")
    
    choice = 0
    while choice < 1 or choice > 5:
        try:
            choice = int(input("\nSelect policy (1-5): "))
        except ValueError:
            print("Please enter a valid number")
    
    filepath = os.path.join(os.getcwd(), 'tuple100k.json')
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found")
        exit(1)
        
    tasks = load_tasks(filepath)
    results = {}
    
    policies = []
    if choice in [1, 5]:
        policies.append(FCFSCooperationGateway)
    if choice in [2, 5]:
        policies.append(FCFSNoCooperationGateway)
    if choice in [3, 5]:
        policies.append(RandomCooperationGateway)
    if choice in [4, 5]:
        policies.append(RandomNoCooperationGateway)
    
    for policy in policies:
        policy_name = policy.__name__.replace('Gateway', '')
        results[policy_name] = run_policy(policy, tasks.copy(), FOG_NODES)
    
    if results:
        analyze_results(results)
    else:
        print("No valid policy selected!")

if __name__ == '__main__':
    main()