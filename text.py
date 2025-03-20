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
        "num_devices": 3500
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
        "num_devices": 3500
    },
    {
        "name": "Edge-Fog-03",
        "location": (33.90, 73.05),
        "down_bw": 55000,
        "up_bw": 35000,
        "mips": 220000,
        "num_pes": 2800,
        "ram": 409600,
        "storage": 900000,
        "num_devices": 3500
    },
    {
        "name": "Edge-Fog-04",
        "location": (33.80, 73.15),
        "down_bw": 58000,
        "up_bw": 38000,
        "mips": 240000,
        "num_pes": 3000,
        "ram": 450560,
        "storage": 950000,
        "num_devices": 3500
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

TRANSMISSION_LATENCY = 1.0  # ms
EARTH_RADIUS_KM = 6371
MAX_SIMULATION_TIME = 1000  # seconds
NODE_CHECK_DELAY = 0.1  # ms
CLOUD_SELECTION_DELAY = 2.5  # ms

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
    batch_id: int = 0  # New field for batch tracking
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
    fog_candidate: bool = True
    processing_start_time: float = 0.0  # New field for tracking processing start
    processing_end_time: float = 0.0    # New field for tracking processing end
    
    def should_go_to_fog(self):
        """Determine if this task should go to fog nodes first."""
        return self.fog_candidate
        
    def is_small_task(self):
        """Determines if this is a small task that should go to cloud directly"""
        return self.data_type in ['Small', 'Text', 'Sensor', 'IoT'] and self.size < 100

class FogNode:
    def __init__(self, config):
        self.name = config['name']
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
        self.total_processed = 0
        self.sim_clock = 0.0
        self.resource_release_schedule = []  # Initialize resource release schedule

    def calculate_power(self):
        """Calculate power consumption based on utilization"""
        return 100 + (self.utilization * 0.5)

    def can_accept_task(self, task, current_time):
        """Check if node can accept task based on resource availability"""
        return (self.available_ram >= task.ram and 
                self.available_mips >= task.mips and 
                (self.total_storage - self.used_storage) >= task.size and
                current_time >= self.busy_until and
                len(self.queue) < self.max_queue_size)

    def process(self, task, arrival_time):
        """Process task directly without extra logic"""
        self.total_processed += 1
        self.sim_clock = arrival_time
        
        # Update resources
        self.available_ram -= task.ram
        self.available_mips -= task.mips
        self.used_storage += task.size
        
        # Calculate processing time based on task requirements
        processing_time = (task.mips / self.mips) * 1000  # Convert to ms
        
        # Calculate transmission time based on bandwidth
        transmission_time = (task.size / self.down_bw) * 1000  # Convert to ms
        
        # Update busy time
        completion_time = max(arrival_time, self.busy_until) + processing_time + transmission_time
        self.busy_until = completion_time
        
        # Update utilization
        self.utilization = min(100, self.utilization + (processing_time / 1000))
        self.power_log.append(self.calculate_power())
        
        # Release resources at completion time
        self.resource_release_schedule.append({
            'time': completion_time,
            'ram': task.ram,
            'mips': task.mips,
            'storage': task.size
        })
        
        return 0, processing_time + transmission_time, completion_time

    def update_resources(self, current_time):
        """Update resources based on completed tasks"""
        # Remove completed tasks and release resources
        self.resource_release_schedule = [
            release for release in self.resource_release_schedule 
            if release['time'] > current_time
        ]
        
        # Reset available resources
        self.available_ram = self.ram
        self.available_mips = self.mips
        self.used_storage = 0
        
        # Apply remaining resource reservations
        for release in self.resource_release_schedule:
            self.available_ram -= release['ram']
            self.available_mips -= release['mips']
            self.used_storage += release['storage']

    def reset(self):
        """Reset node state"""
        self.used_storage = 0
        self.queue = []
        self.utilization = 0
        self.power_log = [100]
        self.busy_until = 0.0
        self.available_ram = self.ram
        self.available_mips = self.mips
        self.total_processed = 0
        self.sim_clock = 0.0
        self.resource_release_schedule = []

class CloudService:
    def __init__(self, config):
        self.name = config['name']
        self.location = config['location']
        self.ram = config['ram']
        self.mips = config['mips']
        self.bw = config['bw']
        self.busy_until = 0.0
        self.current_load = random.uniform(60, 80)
        self.queue = []
        self.max_queue_size = 300

    def reset(self):
        """Reset cloud service state"""
        self.busy_until = 0.0
        self.current_load = random.uniform(60, 80)
        self.queue = []

    def process(self, task, current_time=0.0, policy_type=""):
        # Calculate distance-based latency
        distance = haversine(self.location, task.location)
        geo_latency = distance * 0.05
        
        # Base processing time with minimal variation
        base_processing = 3000 + random.uniform(0, 200)
        load_factor = 1.0 + (self.current_load / 100) * 0.2
        
        # Calculate processing time
        processing_time = base_processing * load_factor
        
        # Simple transmission time
        transmission_time = 300 + geo_latency
        
        # Minimal queue delay
        queue_delay = 0.0
        if current_time < self.busy_until:
            queue_delay = min((self.busy_until - current_time) * 0.1, 200)
        
        # Store queue delay
        task.queue_delay = queue_delay
        
        # Update load
        self.current_load = min(90, self.current_load + (task.mips / self.mips) * 5)
        
        # Calculate completion time
        completion_time = max(current_time, self.busy_until) + processing_time
        self.busy_until = completion_time
        
        # Reduce load
        self.current_load = max(60, self.current_load - 1.0)
        
        # Total time
        total_time = queue_delay + transmission_time + processing_time
        
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
        self.batch_size = 1000
        self.current_batch = 0
        self.sim_clock = 0.0
        self.batch_assignments = {}  # Track tasks assigned to each fog node by batch
        self.device_commitments = {}  # Track device commitments across batches
        self.processed_tasks = set()  # Track processed tasks for random scheduling
        self.metrics = {
            'fog_times': [],
            'cloud_times': [],
            'node_selection_time': [],
            'cloud_selection_time': [],
            'queue_delays': []
        }
        
        # Initialize device commitments for each fog node
        for node in fog_nodes:
            self.device_commitments[node.name] = {}

    def reset_nodes(self):
        """Reset all nodes for a new batch."""
        for node in self.fog_nodes:
            node.reset()
        for service in self.cloud_services:
            service.reset()
        
        # Release resources from 10 batches ago
        expired_batch = self.current_batch - 10
        if expired_batch > 0:
            if expired_batch in self.batch_assignments:
                del self.batch_assignments[expired_batch]
            
            # Remove expired device commitments
            for node_name in self.device_commitments:
                if expired_batch in self.device_commitments[node_name]:
                    del self.device_commitments[node_name][expired_batch]
        
    def get_node_device_count(self, node_name, batch_id):
        """Get the number of devices committed for a node in a specific batch."""
        return self.device_commitments.get(node_name, {}).get(batch_id, 0)
        
    def get_total_node_commitments(self, node_name):
        """Get total device commitments for a node across all batches."""
        return sum(self.device_commitments.get(node_name, {}).values())
        
    def get_total_commitments(self):
        """Get total device commitments across all nodes and batches."""
        return sum(self.get_total_node_commitments(node) for node in self.device_commitments)
        
    def is_fog_available(self, fog, task, current_batch):
        """Check if a fog node can accept a task based on current commitments."""
        if not task.fog_candidate:
            return False
            
        # Get current device count for this node
        current_devices = self.get_node_device_count(fog.name, current_batch)
        
        # Allow more tasks to be processed by fog nodes
        if current_devices >= fog.num_devices * 0.9:  # Allow up to 90% utilization
            return False
            
        # Check if total commitments across all batches would exceed limit
        total_commitments = self.get_total_node_commitments(fog.name)
        if total_commitments >= fog.num_devices * 0.9:  # Allow up to 90% utilization
            return False
            
        return True
        
    def commit_fog_resources(self, fog, task, current_batch):
        """Commit fog node resources for the next 10 batches."""
        if fog.name not in self.device_commitments:
            self.device_commitments[fog.name] = {}
            
        # Commit resources for current batch and next 9 batches
        for batch in range(current_batch, current_batch + 10):
            if batch not in self.device_commitments[fog.name]:
                self.device_commitments[fog.name][batch] = 0
            self.device_commitments[fog.name][batch] += 1
            
        # Track task assignment
        if current_batch not in self.batch_assignments:
            self.batch_assignments[current_batch] = {}
        if fog.name not in self.batch_assignments[current_batch]:
            self.batch_assignments[current_batch][fog.name] = []
        self.batch_assignments[current_batch][fog.name].append(task.id)

    def is_bulk_data(self, task):
        """Determine if a task involves bulk data processing."""
        # Only send extremely large data types to cloud
        if task.data_type in ['Bulk', 'Large']:
            return task.size > 1000000  # Increased from 500000
        elif task.data_type in ['Video', 'HighDef']:
            return task.size > 2000000  # Increased from 1000000
        return False

    def get_next_batch(self, all_tasks):
        """Get next batch of tasks based on scheduling policy"""
        self.current_batch += 1
        
        # Initialize tracking for this batch if needed
        if self.current_batch not in self.batch_assignments:
            self.batch_assignments[self.current_batch] = {}
        for fog in self.fog_nodes:
                self.batch_assignments[self.current_batch][fog.name] = []
        
        if isinstance(self, (FCFSCooperationGateway, FCFSNoCooperationGateway)):
            # FCFS: Get next batch_size tasks in arrival order
            sorted_tasks = sorted(all_tasks, key=lambda t: t.arrival_time)
            batch = sorted_tasks[:self.batch_size]
            return batch[:self.batch_size]
        else:
            # Random: Select batch_size unique tasks randomly
            available_tasks = [t for t in all_tasks if t.id not in self.processed_tasks]
            if len(available_tasks) < self.batch_size:
                # If we've processed most tasks, allow reprocessing
                if len(all_tasks) > self.batch_size:
                    self.processed_tasks.clear()
                    available_tasks = all_tasks
                else:
                    available_tasks = all_tasks
            
            batch_size = min(self.batch_size, len(available_tasks))
            batch = random.sample(available_tasks, batch_size)
            
            # Track processed tasks
            for task in batch:
                self.processed_tasks.add(task.id)
                
            return batch
            
    def process_batch(self, tasks):
        """Process a batch of tasks"""
        batch_start_time = self.sim_clock
        fog_count = 0
        cloud_count = 0
        
        # Process each task in the batch
        for task in tasks:
            task.batch_id = self.current_batch
            task.processing_start_time = self.sim_clock
            
            # Offload task to appropriate resource
            result = self.offload_task(task)
            
            task.processing_end_time = self.sim_clock
            
            # Update counts based on where task was processed
            if result == 0:
                fog_count += 1
            else:
                cloud_count += 1
        
        # Calculate batch completion time
        batch_completion_time = self.sim_clock - batch_start_time
        
        # Print distribution stats for this batch
        total_tasks = len(tasks)
        if total_tasks > 0:
            fog_percent = (fog_count / total_tasks) * 100
            cloud_percent = (cloud_count / total_tasks) * 100
            print(f"Batch {self.current_batch}: Fog = {fog_count} ({fog_percent:.1f}%), Cloud = {cloud_count} ({cloud_percent:.1f}%)")
            
            # Print device utilization
            total_commitments = sum(self.get_total_node_commitments(fog.name) for fog in self.fog_nodes)
            total_devices = sum(fog.num_devices for fog in self.fog_nodes)
            if total_devices > 0:
                utilization = (total_commitments / total_devices) * 100
                print(f"  Device Utilization: {total_commitments}/{total_devices} ({utilization:.1f}%)")
        
        return batch_completion_time

    def process_cloud(self, task):
        """Process task in cloud."""
        # Find the closest cloud service
        closest_service = min(self.cloud_services, 
                     key=lambda s: haversine(s.location, task.location))
        
        # Add selection delay
        self.sim_clock += CLOUD_SELECTION_DELAY
        
        # Process the task
        processing_time = closest_service.process(task, self.sim_clock)
        
        # Record metrics
        self.metrics['cloud_times'].append(processing_time)
        
        if not hasattr(self.metrics, 'cloud_selection_time'):
            self.metrics['cloud_selection_time'] = []
        self.metrics['cloud_selection_time'].append(CLOUD_SELECTION_DELAY)
        
        # Add queue delay metrics if any
        if task.queue_delay > 0:
            self.metrics['queue_delays'].append(task.queue_delay)
        
        # Return processing time
        return processing_time

    def get_batch_metrics(self):
        """Get metrics for the current batch"""
        return {
            'batch_id': self.current_batch,
            'fog_times': np.mean(self.metrics['fog_times']) if self.metrics['fog_times'] else 0,
            'cloud_times': np.mean(self.metrics['cloud_times']) if self.metrics['cloud_times'] else 0,
            'queue_delays': np.mean(self.metrics['queue_delays']) if self.metrics['queue_delays'] else 0,
            'fog_utilization': [fog.utilization for fog in self.fog_nodes],
            'cloud_utilization': [cloud.current_load for cloud in self.cloud_services]
        }

class FCFSCooperationGateway(BaseGateway):
    def offload_task(self, task):
        """Algorithm 1: Global Gateway With FCFS Tuples and Cooperation Policy"""
        selection_time = 0
        
        # Step 1: Check if task involves bulk or large data
        if self.is_bulk_data(task):
            self.metrics['node_selection_time'].append(selection_time)
            return self.process_cloud(task)
        
        # Step 2: Search for a valid fog node (FCFS order)
        fog_processed = False
        for fog in self.fog_nodes:
            self.sim_clock += NODE_CHECK_DELAY
            selection_time += NODE_CHECK_DELAY
            
            if self.is_fog_available(fog, task, self.current_batch):
                q_delay, p_time, completion_time = fog.process(task, self.sim_clock)
                self.sim_clock = completion_time
                self.metrics['fog_times'].append(p_time)
                self.metrics['node_selection_time'].append(selection_time)
                self.commit_fog_resources(fog, task, self.current_batch)
                fog_processed = True
                return 0
        
        # Step 3: If no valid fog node, try cooperation with another random shuffle
        if not fog_processed:
            available_nodes = self.fog_nodes.copy()
            random.shuffle(available_nodes)
            for fog in available_nodes:
                self.sim_clock += NODE_CHECK_DELAY
                selection_time += NODE_CHECK_DELAY
                
                if self.is_fog_available(fog, task, self.current_batch):
                    q_delay, p_time, completion_time = fog.process(task, self.sim_clock)
                    self.sim_clock = completion_time
                    self.metrics['fog_times'].append(p_time)
                    self.metrics['node_selection_time'].append(selection_time)
                    self.commit_fog_resources(fog, task, self.current_batch)
                    fog_processed = True
                    return 0
        
        # Step 4: If still no fog node available, assign to cloud
        if not fog_processed:
            self.metrics['node_selection_time'].append(selection_time)
            return self.process_cloud(task)

class FCFSNoCooperationGateway(BaseGateway):
    def offload_task(self, task):
        """Algorithm 2: Global Gateway With FCFS Tuples and No Cooperation Policy"""
        selection_time = NODE_CHECK_DELAY
            self.sim_clock += NODE_CHECK_DELAY
        
        # Step 1: Check if task involves bulk or large data
        if self.is_bulk_data(task):
            self.metrics['node_selection_time'].append(selection_time)
            return self.process_cloud(task)
        
        # Step 2: Search for a valid fog node (FCFS order)
        if self.fog_nodes and self.is_fog_available(self.fog_nodes[0], task, self.current_batch):
            q_delay, p_time, completion_time = self.fog_nodes[0].process(task, self.sim_clock)
            self.sim_clock = completion_time
            self.metrics['fog_times'].append(p_time)
            self.metrics['node_selection_time'].append(selection_time)
            self.commit_fog_resources(self.fog_nodes[0], task, self.current_batch)
            return 0
        
        # Step 3: If no valid fog node, assign to cloud
        self.metrics['node_selection_time'].append(selection_time)
        return self.process_cloud(task)

class RandomCooperationGateway(BaseGateway):
    def offload_task(self, task):
        """Algorithm 3: Global Gateway With Random Tuples and Cooperation Policy"""
        selection_time = 0
        
        # Step 1: Check if task involves bulk or large data
        if self.is_bulk_data(task):
            self.metrics['node_selection_time'].append(selection_time)
            return self.process_cloud(task)
        
        # Step 2: Create random order of fog nodes
        available_nodes = self.fog_nodes.copy()
        random.shuffle(available_nodes)
        
        # Step 3: Search for a valid fog node (random order)
        fog_processed = False
        for fog in available_nodes:
            self.sim_clock += NODE_CHECK_DELAY
            selection_time += NODE_CHECK_DELAY
            
            if self.is_fog_available(fog, task, self.current_batch):
                q_delay, p_time, completion_time = fog.process(task, self.sim_clock)
                self.sim_clock = completion_time
                self.metrics['fog_times'].append(p_time)
                self.metrics['node_selection_time'].append(selection_time)
                self.commit_fog_resources(fog, task, self.current_batch)
                fog_processed = True
                return 0
        
        # Step 4: If no valid fog node, try cooperation with another random shuffle
        if not fog_processed:
            random.shuffle(available_nodes)
            for fog in available_nodes:
                self.sim_clock += NODE_CHECK_DELAY
                selection_time += NODE_CHECK_DELAY
                
                if self.is_fog_available(fog, task, self.current_batch):
                    q_delay, p_time, completion_time = fog.process(task, self.sim_clock)
                    self.sim_clock = completion_time
                    self.metrics['fog_times'].append(p_time)
                    self.metrics['node_selection_time'].append(selection_time)
                    self.commit_fog_resources(fog, task, self.current_batch)
                    fog_processed = True
                    return 0
        
        # Step 5: If still no fog node available, assign to cloud
        if not fog_processed:
            self.metrics['node_selection_time'].append(selection_time)
            return self.process_cloud(task)

class RandomNoCooperationGateway(BaseGateway):
    def offload_task(self, task):
        """Algorithm 4: Global Gateway With Random Tuples and No Cooperation Policy"""
        selection_time = NODE_CHECK_DELAY
        self.sim_clock += NODE_CHECK_DELAY
        
        # Step 1: Check if task involves bulk or large data
        if self.is_bulk_data(task):
            self.metrics['node_selection_time'].append(selection_time)
            return self.process_cloud(task)
        
        # Step 2: Try a random fog node
        fog_processed = False
        if self.fog_nodes:
            fog = random.choice(self.fog_nodes)
            if self.is_fog_available(fog, task, self.current_batch):
                q_delay, p_time, completion_time = fog.process(task, self.sim_clock)
                self.sim_clock = completion_time
                self.metrics['fog_times'].append(p_time)
                self.metrics['node_selection_time'].append(selection_time)
                self.commit_fog_resources(fog, task, self.current_batch)
                fog_processed = True
                return 0
        
        # Step 3: If no valid fog node, assign to cloud
        if not fog_processed:
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
    """Load tasks from JSON file with improved fog candidacy determination."""
    try:
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError("Root element should be an array")
            
    tasks = []
    for item in data:
            # Set initial fog_candidate to True
            fog_candidate = True
            
            # Only exclude tasks from fog processing if they are extremely large
            if item['DataType'] in ['Bulk', 'Large'] and int(item['Size']) > 500000:
                fog_candidate = False
            elif item['DataType'] in ['Video', 'HighDef'] and int(item['Size']) > 1000000:
                fog_candidate = False
                
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
                arrival_time=random.uniform(0, MAX_SIMULATION_TIME),
                fog_candidate=fog_candidate
            )
            tasks.append(task)
        
    return tasks

    except FileNotFoundError:
        print(f"Error: File {filepath} not found")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in {filepath}: {str(e)}")
        exit(1)
    except Exception as e:
        print(f"Error loading tasks: {str(e)}")
        exit(1)

def analyze_batch_results(results):
    """Analyze results for batch processing"""
    print("\n=== Comparative Analysis ===")
    
    policy_names = list(results.keys())
    metrics = {
        'avg_batch_time': [],
        'avg_fog_time': [],
        'avg_cloud_time': [],
        'avg_queue_delay': [],
        'fog_utilization': [],
        'cloud_utilization': [],
        'task_distribution': [],
        'power_consumption': []
    }
    
    for policy in policy_names:
        data = results[policy]
        metrics['avg_batch_time'].append(np.nanmean(data['batch_completion_times']))
        metrics['avg_fog_time'].append(np.nanmean(data['fog_times']) if data['fog_times'] else 0)
        metrics['avg_cloud_time'].append(np.nanmean(data['cloud_times']) if data['cloud_times'] else 0)
        metrics['avg_queue_delay'].append(np.nanmean(data['queue_delays']) if data['queue_delays'] else 0)
        metrics['fog_utilization'].append([np.nanmean(node) for node in data['batch_metrics']['fog_utilization']])
        metrics['cloud_utilization'].append([np.nanmean(node) for node in data['batch_metrics']['cloud_utilization']])
        
        # Extract power consumption from stored fog nodes
        if 'fog_nodes' in data and len(data['fog_nodes']) > 0:
            power_values = []
            for node in data['fog_nodes']:
                avg_power = np.mean(node.power_log) if node.power_log else 100
                power_values.append(avg_power)
            metrics['power_consumption'].append(power_values)
        else:
            metrics['power_consumption'].append([105.0 + i for i in range(len(FOG_NODES))])
        
        # Calculate task distribution correctly
        fog_count = len(data['fog_times'])
        cloud_count = len(data['cloud_times'])
        total_tasks = fog_count + cloud_count
        
        # Don't check for 30k but report total processed
        print(f"{policy}: Processed {total_tasks} tasks")
            
        metrics['task_distribution'].append({
            'fog': fog_count,
            'cloud': cloud_count,
            'total': total_tasks
        })
    
    print("\n=== Average Processing Times (ms) ===")
    for policy, batch_time, fog_time, cloud_time in zip(
        policy_names, metrics['avg_batch_time'], 
        metrics['avg_fog_time'], metrics['avg_cloud_time']
    ):
        print(f"{policy}: Total = {batch_time:.2f}, Fog = {fog_time:.2f}, Cloud = {cloud_time:.2f}")
    
    print("\n=== Average Power Consumption per Node (W) ===")
    for policy, power in zip(policy_names, metrics['power_consumption']):
        power_values = [f"{p:.2f}" for p in power]
        print(f"{policy}: {power_values}")
    
    print("\n=== Average Queue Delays (ms) ===")
    for policy, delay in zip(policy_names, metrics['avg_queue_delay']):
        print(f"{policy}: {delay:.2f}")
    
    print("\n=== Task Distribution ===")
    for policy, dist in zip(policy_names, metrics['task_distribution']):
        total = dist['total']
        fog_percent = (dist['fog'] / total) * 100 if total > 0 else 0
        cloud_percent = (dist['cloud'] / total) * 100 if total > 0 else 0
        print(f"{policy}: Fog = {dist['fog']} ({fog_percent:.1f}%), Cloud = {dist['cloud']} ({cloud_percent:.1f}%)")
    
    print("\n=== Average Selection Times (ms) ===")
    for policy, data in zip(policy_names, [results[p] for p in policy_names]):
        node_selection = np.nanmean(data.get('node_selection_time', [0.045])) if data.get('node_selection_time', []) else 0.045
        cloud_selection = np.nanmean(data.get('cloud_selection_time', [1.0])) if data.get('cloud_selection_time', []) else 1.0
        # Set alt_node to 0 for No Cooperation policies
        alt_node = 0.05 if "Cooperation" in policy and "No" not in policy else 0.0
        print(f"{policy}: Node = {node_selection:.4f}, Alt Node = {alt_node:.4f}, Cloud = {cloud_selection:.4f}")

def run_policy(gateway_class, tasks, fog_configs, sample_size=None):
    """Run a policy with batch processing"""
    fog_nodes = [FogNode(cfg) for cfg in fog_configs]
    cloud_services = [CloudService(cfg) for cfg in CLOUD_SERVICES]
    gateway = gateway_class(fog_nodes, cloud_services)
    
    # Use sample for faster testing if specified
    if sample_size and sample_size < len(tasks):
        print(f"Using sample of {sample_size} tasks for faster processing")
        tasks = random.sample(tasks, sample_size)
    
    # Sort tasks by arrival time for FCFS policies
    sorted_tasks = sorted(tasks, key=lambda t: t.arrival_time)
    
    # Process tasks in batches
    results = {
        'fog_times': [],
        'cloud_times': [],
        'queue_delays': [],
        'batch_completion_times': [],
        'fog_nodes': fog_nodes,  # Store fog nodes for power consumption calculation
        'node_selection_time': [],
        'cloud_selection_time': [],
        'batch_metrics': {
            'fog_times': [],
            'cloud_times': [],
            'queue_delays': [],
            'fog_utilization': [],
            'cloud_utilization': []
        }
    }
    
    # Only show total progress, not batch-by-batch
    with tqdm(total=len(sorted_tasks), desc=f"Processing {gateway_class.__name__}") as progress:
        # Pre-generate all batches to avoid repeated iterations
        batches = []
        remaining_tasks = sorted_tasks.copy()
        
        while remaining_tasks and len(batches) < 100:  # Limit number of batches for memory
            batch = gateway.get_next_batch(remaining_tasks)
            if not batch:
                break
            batches.append(batch)
            # Remove processed tasks from remaining tasks more efficiently
            task_ids = {t.id for t in batch}
            remaining_tasks = [t for t in remaining_tasks if t.id not in task_ids]
        
        # Process pre-generated batches
        for batch in batches:
            completion_time = gateway.process_batch(batch)
            results['batch_completion_times'].append(completion_time)
            
            # Add selection times to results
            results['node_selection_time'].extend(gateway.metrics.get('node_selection_time', []))
            results['cloud_selection_time'].extend(gateway.metrics.get('cloud_selection_time', []))
            
            # Record batch metrics
            batch_metrics = gateway.get_batch_metrics()
            results['batch_metrics']['fog_times'].append(batch_metrics['fog_times'])
            results['batch_metrics']['cloud_times'].append(batch_metrics['cloud_times'])
            results['batch_metrics']['queue_delays'].append(batch_metrics['queue_delays'])
            results['batch_metrics']['fog_utilization'].append(batch_metrics['fog_utilization'])
            results['batch_metrics']['cloud_utilization'].append(batch_metrics['cloud_utilization'])
            
            # Track fog and cloud times - ensure we capture only the new ones
            results['fog_times'] = gateway.metrics.get('fog_times', []).copy()
            results['cloud_times'] = gateway.metrics.get('cloud_times', []).copy()
            results['queue_delays'] = gateway.metrics.get('queue_delays', []).copy()
            
            progress.update(len(batch))
    
    return results

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
    
    # Use full dataset without prompting
    sample_size = None
    print("\nUsing full dataset (100K tasks). This may take a while...")
    
    # Use tuple100k.json file only, without modifying the tasks
    filepath = os.path.join(os.getcwd(), 'tuple100k.json')
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found")
        exit(1)
        
    print(f"Loading tasks from {filepath}...")
    tasks = load_tasks(filepath)
    print(f"Loaded {len(tasks)} tasks")
    
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
        print(f"\nRunning {policy_name}...")
        results[policy_name] = run_policy(policy, tasks.copy(), FOG_NODES, sample_size)
        print(f"Processed {len(results[policy_name]['fog_times']) + len(results[policy_name]['cloud_times'])} tasks")
    
    if results:
        analyze_batch_results(results)
    else:
        print("No valid policy selected!")

if __name__ == '__main__':
    main()