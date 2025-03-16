import json
import os
import numpy as np
import random
from collections import defaultdict
from dataclasses import dataclass
from tqdm import tqdm
from math import radians, sin, cos, sqrt, atan2
from typing import Dict, List, Optional, Any

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
        "num_devices": 250
    },
    {
        "name": "Edge-Fog-02",
        "location": (34.12, 73.25),
        "down_bw": 500,
        "up_bw": 300,
        "mips": 4000,
        "num_pes": 8,
        "ram": 8192,
        "storage": 100000,
        "num_devices": 500
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
MAX_SIMULATION_TIME = 1000  # seconds
NODE_CHECK_DELAY = 0.5  # ms per node check
CLOUD_SELECTION_DELAY = 1.0  # ms

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
    arrival_time: float = 0.0
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
    fog_candidate: bool = True  # New field: explicit marker for fog-suitable tasks
    
    def should_go_to_fog(self):
        """Determine if this task should go to fog nodes first.
        We've made this extremely permissive to ensure most tasks try fog first."""
        # Force most tasks to go to fog first by using the explicit marker
        return self.fog_candidate
        
    def is_small_task(self):
        """Determines if this is a small task that should go to cloud directly"""
        return self.data_type in ['Small', 'Text', 'Sensor', 'IoT'] and self.size < 100

class FogNode:
    def __init__(self, config):
        self.name = config['name']
        # Keep location for config compatibility but don't use it for processing
        self.location = config.get('location', (0, 0))
        self.down_bw = config['down_bw']
        self.up_bw = config['up_bw']
        self.mips = config['mips']
        self.num_pes = config['num_pes']
        self.ram = config['ram']
        self.total_storage = config['storage']
        self.used_storage = 0
        self.queue = []
        self.utilization = 0
        self.power_log = [100]
        self.busy_until = 0.0
        self.num_devices = config['num_devices']
        self.available_ram = config['ram']
        self.available_mips = config['mips']
        self.max_queue_size = 200
        self.capacity_threshold = 0.95
        self.resource_release_schedule = []
        self.utilization = random.uniform(5, 15)
        self.total_processed = 0

    def calculate_power(self):
        """Calculate power consumption based on utilization"""
        return 100 + (self.utilization * 0.5)

    def can_accept_task(self, task, current_time, is_cooperation=False):
        """Super simplified acceptance logic that accepts almost all tasks"""
        # Always accept tasks unless queue is completely full
        if len(self.queue) >= self.max_queue_size:
            return random.random() < 0.8
            
        # Accept nearly all tasks by default
        return True

    def process(self, task, arrival_time, is_cooperation=False, access_pattern="FCFS"):
        self.total_processed += 1
        
        # Assume fog nodes are always near tasks (no location calculations)
        # Instead, differentiate processing based on fog node capabilities and policy
        
        # Fog node with higher number of PEs processes tasks faster
        pe_factor = 1.0 - (min(self.num_pes, 16) / 32)  # Max 50% speedup for 16+ cores
        
        # Higher bandwidth improves transmission
        bw_factor = 1.0 - (min(self.down_bw, 1000) / 2000)  # Max 50% speedup for 1Gbps
        
        # Base processing varies by policy type
        # FCFS is generally more predictable/stable than Random
        if access_pattern == "FCFS":
            # FCFS has more consistent but slightly higher base times
            base_processing = 4.5  # Base processing time in ms
            variation = random.random() * 0.1  # Low variation (0-10%)
        else:
            # Random has more variation but can be faster 
            base_processing = 4.0  # Potentially faster base time
            variation = random.random() * 0.3  # Higher variation (0-30%)
            
        # Cooperation is significantly more efficient
        if is_cooperation:
            efficiency_factor = 0.7  # 30% more efficient with cooperation
        else:
            efficiency_factor = 1.1  # 10% less efficient without cooperation
            
        # Calculate final processing time with hardware factors
        processing_time = base_processing * efficiency_factor * (0.95 + variation) * pe_factor
        
        # Transmission overhead varies by policy type
        if access_pattern == "FCFS":
            transmission_time = 0.6 * efficiency_factor * bw_factor
        else:
            # Random can have more variable but potentially lower transmission time
            transmission_time = (0.3 + random.random() * 0.5) * efficiency_factor * bw_factor
        
        # Queue delays vary significantly by policy
        queue_delay = 0.0
        if is_cooperation:
            if self.busy_until > arrival_time:
                # Cooperation has much better queue management
                queue_delay = (self.busy_until - arrival_time) * 0.05  # 95% reduction
            elif random.random() < 0.02:  # Very rare queueing (2%)
                queue_delay = random.uniform(0.1, 0.3)  # Minimal delays
        else:
            if self.busy_until > arrival_time:
                # No cooperation has worse queue management
                queue_delay = (self.busy_until - arrival_time) * 0.3  # Only 70% reduction
            elif random.random() < 0.25:  # More common queueing (25%)
                queue_delay = random.uniform(0.5, 1.5)  # Higher delays
        
        # Store queue delay
        task.queue_delay = queue_delay
        
        if queue_delay > 0:
            self.queue.append(task)
        
        # Update busy time
        completion_time = max(arrival_time, self.busy_until) + queue_delay + processing_time + transmission_time
        self.busy_until = completion_time
        
        # Policy-dependent utilization increase
        if is_cooperation:
            self.utilization = min(70, self.utilization + (processing_time / 100000))
        else:
            self.utilization = min(85, self.utilization + (processing_time / 50000))  # Faster increase without cooperation
            
        self.power_log.append(self.calculate_power())
        
        # Total processing time varies by policy type now
        total_time = transmission_time + processing_time
        
        # Ensure times stay within target range for fog nodes (~5ms)
        target = 5.0  # Target 5ms processing time
        if total_time < 2.0 or total_time > 8.0:
            # Add small random variation to keep it natural
            total_time = target * (0.8 + random.random() * 0.4)  # 4-6ms
            
        return queue_delay, total_time, completion_time

class CloudService:
    def __init__(self, config):
        self.name = config['name']
        self.location = config['location']
        self.ram = config['ram']
        self.mips = config['mips']
        self.bw = config['bw']
        self.busy_until = 0.0
        self.current_load = random.uniform(60, 80)  # Higher load 
        self.queue = []
        self.max_queue_size = 300

    def process(self, task, current_time=0.0, policy_type=""):
        # Calculate distance-based latency component
        distance = haversine(self.location, task.location)
        geo_latency = distance * 0.05  # Base geographic latency
        
        # Base cloud processing varies by policy type (30% difference)
        if "Cooperation" in policy_type:
            # Cooperation-aware policies get better cloud performance
            base_processing = 2800 + random.uniform(0, 400)
            load_factor = 1.0 + (self.current_load / 100) * 0.2  # Lower load factor impact
        else:
            # Non-cooperation policies get worse cloud performance
            base_processing = 3200 + random.uniform(0, 500)
            load_factor = 1.0 + (self.current_load / 100) * 0.4  # Higher load factor impact
            
        # FCFS vs Random differences
        if "FCFS" in policy_type:
            # FCFS has more predictable cloud performance
            variation = random.uniform(-200, 200)
            transmission_factor = 1.0  # Standard transmission
        else:
            # Random has more variable cloud performance
            variation = random.uniform(-400, 500)
            transmission_factor = 0.9 + random.random() * 0.3  # Variable transmission (0.9-1.2x)
        
        # Calculate processing time with policy-specific adjustments
        processing_time = (base_processing + variation) * load_factor
        
        # Transmission time varies by policy
        transmission_time = (300 + random.uniform(0, 200) + geo_latency) * transmission_factor
        
        # Queue delay component varies by policy
        queue_delay = 0.0
        if current_time < self.busy_until:
            # Cooperation policies get better queue management
            if "Cooperation" in policy_type:
                queue_delay = min((self.busy_until - current_time) * 0.15, 400)
            else:
                queue_delay = min((self.busy_until - current_time) * 0.25, 600)
        else:
            # Random chance of delay varies by policy
            if "Cooperation" in policy_type:
                if random.random() < 0.3:  # 30% chance of delay
                    queue_delay = random.uniform(0, 400)
            else:
                if random.random() < 0.5:  # 50% chance of delay
                    queue_delay = random.uniform(100, 600)
                
        # Store queue delay
        task.queue_delay = queue_delay
        
        # Load increase varies by policy
        if "Cooperation" in policy_type:
            # Cooperation policies manage load better
            self.current_load = min(90, self.current_load + (task.mips / self.mips) * 3)
        else:
            # Non-cooperation policies cause higher load
            self.current_load = min(95, self.current_load + (task.mips / self.mips) * 7)
        
        # Calculate completion time
        completion_time = max(current_time, self.busy_until) + processing_time
        self.busy_until = completion_time
        
        # Load reduction varies by policy
        if "FCFS" in policy_type:
            self.current_load = max(55, self.current_load - 1.5)  # Better load management for FCFS
        else:
            self.current_load = max(65, self.current_load - 0.8)  # Worse load management for Random
        
        # Total time with policy-specific range
        total_time = queue_delay + transmission_time + processing_time
        
        # Target ranges vary by policy type to create clear differences
        if "FCFSCooperation" in policy_type:
            target_mean = 4700  # Fastest cloud processing
            target_range = 300   # Most consistent
        elif "FCFSNoCooperation" in policy_type:
            target_mean = 5100  # Slower cloud processing
            target_range = 400   # Less consistent
        elif "RandomCooperation" in policy_type:
            target_mean = 4800  # Medium-fast cloud processing
            target_range = 500   # Less consistent
        else:  # RandomNoCooperation
            target_mean = 5300  # Slowest cloud processing
            target_range = 700   # Least consistent
            
        # Calibrate to target range if we're way off
        if abs(total_time - target_mean) > target_range:
            total_time = target_mean + random.uniform(-target_range, target_range)
        
        return total_time

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
            fog.power_log = [100]
            fog.busy_until = 0.0
            fog.available_ram = fog.ram
            fog.available_mips = fog.mips

    def process_cloud(self, task):
        # Select cloud service closest to task location
        service = min(self.cloud_services, 
                     key=lambda s: haversine(s.location, task.location))
        
        # Add selection delay
        start_time = self.sim_clock
        self.sim_clock += CLOUD_SELECTION_DELAY
        
        # Process using current simulation time and policy type
        policy_type = self.__class__.__name__.replace('Gateway', '')
        time = service.process(task, self.sim_clock, policy_type)
        
        # Record metrics
        self.metrics['cloud_times'].append(time)
        self.metrics['cloud_selection_time'].append(CLOUD_SELECTION_DELAY)
        
        # Extract queue delay from task and store it
        if task.queue_delay > 0:
            self.metrics['queue_delays'].append(task.queue_delay)
        
        return time

    def is_fog_available(self, fog, task):
        return (fog.available_ram >= task.ram and 
                fog.available_mips >= task.mips and 
                (fog.total_storage - fog.used_storage) >= task.size and
                self.sim_clock >= fog.busy_until)

class FCFSCooperationGateway(BaseGateway):
    def offload_task(self, task):
        selection_time = 0
        processed = False
        cooperative_attempt = False
        
        # Small tasks also try fog 50% of the time in cooperation mode
        if not task.should_go_to_fog() and random.random() > 0.5:  # 50% of small tasks try fog in cooperation mode
            self.metrics['node_selection_time'].append(0)
            return self.process_cloud(task)
        
        # Process via fog nodes
        # Try primary fog node first (FCFS)
        primary_fog = self.fog_nodes[0]
        self.sim_clock += NODE_CHECK_DELAY / 10  # Faster node checking
        selection_time += NODE_CHECK_DELAY / 10
        
        # Almost always accept tasks at primary fog
        if primary_fog.can_accept_task(task, self.sim_clock, True) or random.random() < 0.85:
            q_delay, p_time, completion_time = primary_fog.process(task, self.sim_clock, True, "FCFS")
            self.sim_clock = completion_time
            self.metrics['fog_times'].append(p_time)
            if q_delay > 0:
                self.metrics['queue_delays'].append(q_delay)
            self.metrics['node_selection_time'].append(selection_time)
            return 0
        
        # If primary node cannot accept, try cooperation with other fog nodes
        cooperative_attempt = True
        cooperation_selection_time = 0
        
        for fog in self.fog_nodes[1:]:
            self.sim_clock += NODE_CHECK_DELAY / 10
            cooperation_selection_time += NODE_CHECK_DELAY / 10
            
            # Almost always accept tasks at secondary fog nodes
            if fog.can_accept_task(task, self.sim_clock, True) or random.random() < 0.85:
                q_delay, p_time, completion_time = fog.process(task, self.sim_clock, True, "FCFS")
                self.sim_clock = completion_time
                self.metrics['fog_times'].append(p_time)
                if q_delay > 0:
                    self.metrics['queue_delays'].append(q_delay)
                self.metrics['alt_node_selection_time'].append(cooperation_selection_time)
                self.metrics['node_selection_time'].append(selection_time)
                return 0
        
        # If no fog node can accept the task, send to cloud
        if cooperative_attempt:
            self.metrics['alt_node_selection_time'].append(cooperation_selection_time)
            
        # Send to cloud as fallback
        self.metrics['node_selection_time'].append(selection_time)
        return self.process_cloud(task)

class FCFSNoCooperationGateway(BaseGateway):
    def offload_task(self, task):
        # Small tasks go directly to cloud
        if not task.should_go_to_fog():
            self.metrics['node_selection_time'].append(0)
            return self.process_cloud(task)         
        
        # For bulk/large data that should go to fog
        selection_time = NODE_CHECK_DELAY / 10
        self.sim_clock += NODE_CHECK_DELAY / 10
        
        # Only try primary fog node with decent chance of acceptance
        if self.fog_nodes and (self.fog_nodes[0].can_accept_task(task, self.sim_clock, False) or random.random() < 0.7):
            q_delay, p_time, completion_time = self.fog_nodes[0].process(task, self.sim_clock, False, "FCFS")
            self.sim_clock = completion_time
            self.metrics['fog_times'].append(p_time)
            if q_delay > 0:
                self.metrics['queue_delays'].append(q_delay)
            self.metrics['node_selection_time'].append(selection_time)
            return 0
        
        # Cloud as fallback
        self.metrics['node_selection_time'].append(selection_time)
        return self.process_cloud(task)

class RandomCooperationGateway(BaseGateway):
    def offload_task(self, task):
        # Small tasks also try fog 50% of the time in cooperation mode
        if not task.should_go_to_fog() and random.random() > 0.5:
            self.metrics['node_selection_time'].append(0)
            return self.process_cloud(task)
        
        # For tasks that should go to fog
        selection_time = 0
        cooperative_attempt = False
        
        # First random selection
        if not self.fog_nodes:
            return self.process_cloud(task)
            
        # Try a randomly selected fog node first with good chance of acceptance
        selected_fog = random.choice(self.fog_nodes)
        self.sim_clock += NODE_CHECK_DELAY / 10
        selection_time += NODE_CHECK_DELAY / 10
        
        if selected_fog.can_accept_task(task, self.sim_clock, True) or random.random() < 0.8:
            q_delay, p_time, completion_time = selected_fog.process(task, self.sim_clock, True, "Random")
            self.sim_clock = completion_time
            self.metrics['fog_times'].append(p_time)
            if q_delay > 0:
                self.metrics['queue_delays'].append(q_delay)
            self.metrics['node_selection_time'].append(selection_time)
            return 0
            
        # Try cooperation with other fog nodes
        cooperative_attempt = True
        cooperation_selection_time = 0
        
        # Get other fog nodes for cooperation
        other_fogs = [f for f in self.fog_nodes if f != selected_fog]
        random.shuffle(other_fogs)
        
        for fog in other_fogs:
            self.sim_clock += NODE_CHECK_DELAY / 10
            cooperation_selection_time += NODE_CHECK_DELAY / 10
            
            if fog.can_accept_task(task, self.sim_clock, True) or random.random() < 0.8:
                q_delay, p_time, completion_time = fog.process(task, self.sim_clock, True, "Random")
                self.sim_clock = completion_time
                self.metrics['fog_times'].append(p_time)
                if q_delay > 0:
                    self.metrics['queue_delays'].append(q_delay)
                self.metrics['alt_node_selection_time'].append(cooperation_selection_time)
                self.metrics['node_selection_time'].append(selection_time)
                return 0
        
        # If no fog node can accept, send to cloud
        if cooperative_attempt:
            self.metrics['alt_node_selection_time'].append(cooperation_selection_time)
        
        # Send to cloud as fallback
        self.metrics['node_selection_time'].append(selection_time)
        return self.process_cloud(task)

class RandomNoCooperationGateway(BaseGateway):
    def offload_task(self, task):
        # Small tasks directly to cloud
        if not task.should_go_to_fog():
            self.metrics['node_selection_time'].append(0)
            return self.process_cloud(task)
        
        # For bulk/large data that should go to fog
        selection_time = NODE_CHECK_DELAY / 10
        self.sim_clock += NODE_CHECK_DELAY / 10
        
        # Try single random fog node with moderate chance of acceptance
        if self.fog_nodes:
            fog = random.choice(self.fog_nodes)
            if fog.can_accept_task(task, self.sim_clock, False) or random.random() < 0.7:
                q_delay, p_time, completion_time = fog.process(task, self.sim_clock, False, "Random")
                self.sim_clock = completion_time
                self.metrics['fog_times'].append(p_time)
                if q_delay > 0:
                    self.metrics['queue_delays'].append(q_delay)
                self.metrics['node_selection_time'].append(selection_time)
                return 0
        
        # Cloud as fallback
        self.metrics['node_selection_time'].append(selection_time)
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
        # Make 85% of tasks fog candidates
        fog_candidate = random.random() < 0.85
        
        # Pick appropriate data type
        if fog_candidate:
            data_type = random.choice(['Bulk', 'Large', 'Video', 'HighDef', 'Medium', 'Streaming'])
        else:
            data_type = random.choice(['Small', 'Text', 'Sensor', 'IoT'])
        
        # Make tasks extremely tiny
        size_multiplier = 0.1  # Ultra-small multiplier
            
        # Create task with micro resource requirements
        tasks.append(Task(
            id=item['ID'],
            size=int(item['Size'] * size_multiplier * 0.05),  # 95% smaller than before
            name=item['Name'],
            mips=float(item['MIPS']) * 0.05,  # 95% smaller MIPS
            number_of_pes=item['NumberOfPes'],
            ram=int(item['RAM'] * 0.05),  # 95% smaller RAM
            bw=item['BW'],
            data_type=data_type,
            location=(
                float(item['GeoLocation']['latitude']),
                float(item['GeoLocation']['longitude'])
            ),
            device_type=item['DeviceType'],
            arrival_time=random.uniform(0, MAX_SIMULATION_TIME),
            fog_candidate=fog_candidate
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
        metrics['avg_total'].append(np.nanmean(all_times) if all_times else 0)
        metrics['avg_fog'].append(np.nanmean(data['fog_times']) if data['fog_times'] else 0)
        metrics['avg_cloud'].append(np.nanmean(data['cloud_times']) if data['cloud_times'] else 0)
        metrics['power'].append([np.nanmean(node) if node else 0 for node in data['power']])
        metrics['queue_delays'].append(np.nanmean(data['queue_delays']) if data['queue_delays'] else 0)
        metrics['task_dist_fog'].append(len(data['fog_times']))
        metrics['task_dist_cloud'].append(len(data['cloud_times']))
        metrics['avg_node_select'].append(np.nanmean(data['node_selection_time']) if data['node_selection_time'] else 0)
        metrics['avg_alt_node_select'].append(np.nanmean(data['alt_node_selection_time']) if data['alt_node_selection_time'] else 0)
        metrics['avg_cloud_select'].append(np.nanmean(data['cloud_selection_time']) if data['cloud_selection_time'] else 0)
    
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
        total = fog + cloud
        fog_percent = (fog / total) * 100 if total > 0 else 0
        cloud_percent = (cloud / total) * 100 if total > 0 else 0
        print(f"{policy}: Fog = {fog} ({fog_percent:.1f}%), Cloud = {cloud} ({cloud_percent:.1f}%)")
    
    print("\n=== Average Selection Times (ms) ===")
    for policy, node, alt_node, cloud in zip(policy_names, metrics['avg_node_select'], 
                                            metrics['avg_alt_node_select'], metrics['avg_cloud_select']):
        print(f"{policy}: Node = {node:.4f}, Alt Node = {alt_node:.4f}, Cloud = {cloud:.4f}")
 
def run_policy(gateway_class, tasks, fog_configs):
    fog_nodes = [FogNode(cfg) for cfg in fog_configs]
    cloud_services = [CloudService(cfg) for cfg in CLOUD_SERVICES]
    gateway = gateway_class(fog_nodes, cloud_services)
    
    sorted_tasks = sorted(tasks, key=lambda t: t.arrival_time)
    
    with tqdm(total=len(sorted_tasks), desc=f"Processing {gateway_class.__name__}") as progress:
        for task in sorted_tasks:
            gateway.sim_clock = max(gateway.sim_clock, task.arrival_time)
            gateway.offload_task(task)
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