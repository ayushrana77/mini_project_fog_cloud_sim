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
        self.max_queue_size = 500  # Increased from 200 to 500
        self.total_processed = 0
        self.sim_clock = 0.0
        self.resource_release_schedule = []
        # Add cumulative tracking variables
        self.cumulative_processed = 0
        self.cumulative_utilization = 0
        self.peak_utilization = 0  # Track peak utilization
        self.total_ram_hours = 0   # Track total RAM usage over time
        self.total_mips_hours = 0  # Track total MIPS usage over time
        self.total_storage_hours = 0  # Track total storage usage over time

    def calculate_power(self):
        """Calculate power consumption based on utilization"""
        return 100 + (self.utilization * 0.5)

    def can_accept_task(self, task, current_time):
        """Check if node can accept task based on resource availability"""
        # Allow tasks to be accepted if resources are available, even if busy
        return (self.available_ram >= task.ram and 
                self.available_mips >= task.mips and 
                (self.total_storage - self.used_storage) >= task.size and
                len(self.queue) < self.max_queue_size)

    def process(self, task, arrival_time):
        """Process task with flexible resource allocation."""
        self.total_processed += 1
        self.cumulative_processed += 1  # Update cumulative count
        self.sim_clock = arrival_time
        
        # Calculate actual resource allocation (may be relaxed)
        allocated_ram = min(task.ram, self.available_ram)
        allocated_mips = min(task.mips, self.available_mips)
        allocated_storage = min(task.size, self.total_storage - self.used_storage)
        
        # Update resources
        self.available_ram -= allocated_ram
        self.available_mips -= allocated_mips
        self.used_storage += allocated_storage
        
        # Calculate processing time based on task requirements and actual allocation
        efficiency = min(allocated_mips / task.mips, allocated_ram / task.ram)
        processing_time = (task.mips / self.mips) * (1000 / efficiency)  # Adjust for resource efficiency
        
        # Calculate transmission time based on bandwidth
        transmission_time = (task.size / self.down_bw) * 1000  # Convert to ms
        
        # Update busy time
        completion_time = max(arrival_time, self.busy_until) + processing_time + transmission_time
        self.busy_until = completion_time
        
        # Update utilization
        utilization_increase = min(20, (processing_time / 1000) * (allocated_mips / self.mips) * 100)
        self.utilization = min(100, self.utilization + utilization_increase)
        
        # Calculate current resource utilization percentages
        ram_util = (self.ram - self.available_ram) / self.ram * 100
        mips_util = (self.mips - self.available_mips) / self.mips * 100
        storage_util = self.used_storage / self.total_storage * 100
        
        # Update total resource-hours (converted from ms to hours)
        process_hours = processing_time / (1000 * 60 * 60)  # Convert ms to hours
        self.total_ram_hours += allocated_ram * process_hours
        self.total_mips_hours += allocated_mips * process_hours
        self.total_storage_hours += allocated_storage * process_hours
        
        # Calculate weighted resource utilization (prioritize constrained resources)
        resource_utilization = (ram_util * 0.35 + mips_util * 0.35 + storage_util * 0.3)
        
        # Apply exponential moving average with higher alpha for more recent data impact
        alpha = 0.5  # Increased from 0.3 to give more weight to recent utilization
        self.cumulative_utilization = alpha * resource_utilization + (1 - alpha) * self.cumulative_utilization
        
        # Track peak utilization
        self.peak_utilization = max(self.peak_utilization, resource_utilization)
        
        self.power_log.append(self.calculate_power())
        
        # Release resources at completion time
        self.resource_release_schedule.append({
            'time': completion_time,
            'ram': allocated_ram,
            'mips': allocated_mips,
            'storage': allocated_storage
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
        """Reset node state while preserving cumulative statistics"""
        self.used_storage = 0
        self.queue = []
        self.utilization = 0  # Reset batch utilization
        self.power_log = [100]
        self.busy_until = 0.0
        self.available_ram = self.ram
        self.available_mips = self.mips
        self.total_processed = 0  # Reset batch processed count
        self.sim_clock = 0.0
        self.resource_release_schedule = []
        # Don't reset cumulative statistics

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

# ========== Gateway Implementation ==========
class BaseGateway:
    def __init__(self, fog_nodes, cloud_services):
        self.fog_nodes = fog_nodes
        self.cloud_services = cloud_services
        self.batch_size = 1000
        self.current_batch = 0
        self.sim_clock = 0.0
        self.batch_assignments = {}  # Track tasks assigned to each fog node by batch
        self.device_commitments = {}  # Track device commitments across batches
        self.processed_tasks = set()  # Track processed tasks
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
        
        # Adjust allocation strategy based on data type
        utilization_factor = 0.97  # Default high utilization
        
        # Large and Bulk data types get lower utilization thresholds
        if task.data_type in ['Large', 'Bulk']:
            utilization_factor = 0.90
            
        # Get current device count for this node
        current_devices = self.get_node_device_count(fog.name, current_batch)
        
        # Check device commitments for current batch
        if current_devices >= fog.num_devices * utilization_factor:
            return False
            
        # Check if total commitments across all batches would exceed limit
        total_commitments = self.get_total_node_commitments(fog.name)
        if total_commitments >= fog.num_devices * utilization_factor:
            return False
        
        # Allocate resources based on data type
        ram_factor = 0.9
        mips_factor = 0.9
        storage_factor = 0.9
        
        if task.data_type in ['Large', 'Bulk']:
            # More stringent resource requirements for large data
            ram_factor = 1.0
            mips_factor = 1.0
            storage_factor = 1.0
        
        # Check if node has enough resources
        if (fog.available_ram < task.ram * ram_factor or 
            fog.available_mips < task.mips * mips_factor or 
            (fog.total_storage - fog.used_storage) < task.size * storage_factor):
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
        # Send all Large and Bulk tasks directly to cloud
        if task.data_type in ['Large', 'Bulk']:
            return True  # All Large and Bulk tasks go to cloud
        elif task.data_type in ['Abrupt', 'LocationBased', 'Medical', 'SmallTextual', 'Multimedia']:
            return task.size > 230  # Higher than avg (194)
        return False

    def get_next_batch(self, all_tasks):
        """Get next batch of tasks based on FCFS scheduling policy"""
        self.current_batch += 1
        
        # Initialize tracking for this batch if needed
        if self.current_batch not in self.batch_assignments:
            self.batch_assignments[self.current_batch] = {}
            for fog in self.fog_nodes:
                self.batch_assignments[self.current_batch][fog.name] = []
        
            # FCFS: Get next batch_size tasks in arrival order
            sorted_tasks = sorted(all_tasks, key=lambda t: t.arrival_time)
        return sorted_tasks[:self.batch_size]
            
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
            print(f"\nBatch {self.current_batch}: Fog = {fog_count} ({fog_percent:.1f}%), Cloud = {cloud_count} ({cloud_percent:.1f}%)")
            
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
    def __init__(self, fog_nodes, cloud_services):
        super().__init__(fog_nodes, cloud_services)
        # Track allocation by data type
        self.data_type_counts = {
            'Abrupt': {'fog': 0, 'cloud': 0},
            'Large': {'fog': 0, 'cloud': 0},
            'LocationBased': {'fog': 0, 'cloud': 0},
            'Bulk': {'fog': 0, 'cloud': 0},
            'Medical': {'fog': 0, 'cloud': 0},
            'SmallTextual': {'fog': 0, 'cloud': 0},
            'Multimedia': {'fog': 0, 'cloud': 0}
        }
        self.verbose_output = False  # Control detailed task-level output

    def offload_task(self, task):
        """Algorithm 1: Global Gateway With FCFS Tuples and Cooperation Policy"""
        selection_time = 0
        rejected_by_fog = False
        reassigned = False
        task_type = task.data_type
        allocation = "Unknown"
        
        # Step 1: Check if task involves bulk data based on thresholds
        if self.is_bulk_data(task):
            self.metrics['node_selection_time'].append(selection_time)
            allocation = "Cloud"
            processing_time = self.process_cloud(task)
            # Track cloud allocation for this data type
            if task_type in self.data_type_counts:
                self.data_type_counts[task_type]['cloud'] += 1
            # Print task information only if verbose mode is on
            if self.verbose_output:
                print(f"Task {task.id}: {task_type}, Allocated to {allocation}, Size: {task.size}, Lifetime: {processing_time:.2f}ms")
            return processing_time
        
        # Step 2: Search for a valid fog node with load balancing
        fog_processed = False
        
        # Sort fog nodes by workload (prioritize less-loaded nodes)
        # Calculate workload as a combination of:
        # 1. Cumulative processed tasks
        # 2. Current utilization (available resources)
        # 3. Queue length
        sorted_nodes = sorted(self.fog_nodes, 
                             key=lambda f: (
                                 f.cumulative_processed, 
                                 (f.ram - f.available_ram) / f.ram,
                                 (f.mips - f.available_mips) / f.mips,
                                 f.used_storage / f.total_storage
                             ))
        
        # Try each node in workload-balanced order
        for fog in sorted_nodes:
                self.sim_clock += NODE_CHECK_DELAY
                selection_time += NODE_CHECK_DELAY
                
                if self.is_fog_available(fog, task, self.current_batch):
                    q_delay, p_time, completion_time = fog.process(task, self.sim_clock)
                    self.sim_clock = completion_time
                    self.metrics['fog_times'].append(p_time)
                    self.metrics['node_selection_time'].append(selection_time)
                    self.commit_fog_resources(fog, task, self.current_batch)
                    fog_processed = True
                allocation = f"Fog ({fog.name})"
                # Track fog allocation for this data type
                if task_type in self.data_type_counts:
                    self.data_type_counts[task_type]['fog'] += 1
                # Print task information only if verbose mode is on
                if self.verbose_output:
                    print(f"Task {task.id}: {task_type}, Allocated to {allocation}, Size: {task.size}, Lifetime: {p_time:.2f}ms")
                    # Print fog resource status
                    self.print_fog_status()
                    return 0
        
        # Step 3: If no valid fog node, try cooperation with another random shuffle
        if not fog_processed:
            rejected_by_fog = True
        available_nodes = self.fog_nodes.copy()
        random.shuffle(available_nodes)
        
            # Try each node in random order
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
                    allocation = f"Fog ({fog.name})"
                    reassigned = True
                    # Track fog allocation for this data type
                    if task_type in self.data_type_counts:
                        self.data_type_counts[task_type]['fog'] += 1
                    # Print task information only if verbose mode is on
                    if self.verbose_output:
                        print(f"Task {task.id}: {task_type}, Allocated to {allocation}, Size: {task.size}, Reassigned: Yes, Lifetime: {p_time:.2f}ms")
                        # Print fog resource status
                        self.print_fog_status()
                return 0
        
        # Step 4: If still no fog node available, try one more time with relaxed constraints
        if not fog_processed:
            # Adjust relaxation factor based on data type
            relaxation_factor = 0.8
            if task.data_type in ['Abrupt', 'SmallTextual', 'Medical']:
                relaxation_factor = 0.7  # More relaxed for smaller data types
            
            for fog in self.fog_nodes:
                self.sim_clock += NODE_CHECK_DELAY
                selection_time += NODE_CHECK_DELAY
                
                # Check if node has basic resources available with relaxed constraints
                if (fog.available_ram >= task.ram * relaxation_factor and
                    fog.available_mips >= task.mips * relaxation_factor and
                    (fog.total_storage - fog.used_storage) >= task.size * relaxation_factor and
                    len(fog.queue) < fog.max_queue_size):
                    
                    q_delay, p_time, completion_time = fog.process(task, self.sim_clock)
                    self.sim_clock = completion_time
                    self.metrics['fog_times'].append(p_time)
                    self.metrics['node_selection_time'].append(selection_time)
                    self.commit_fog_resources(fog, task, self.current_batch)
                    fog_processed = True
                    allocation = f"Fog ({fog.name}) [relaxed]"
                    reassigned = True
                    # Track fog allocation for this data type
                    if task_type in self.data_type_counts:
                        self.data_type_counts[task_type]['fog'] += 1
                    # Print task information only if verbose mode is on
                    if self.verbose_output:
                        print(f"Task {task.id}: {task_type}, Allocated to {allocation}, Size: {task.size}, Reassigned: Yes, Lifetime: {p_time:.2f}ms")
                        # Print fog resource status
                        self.print_fog_status()
                    return 0
        
        # Step 5: If still no fog node available, assign to cloud
        if not fog_processed:
            self.metrics['node_selection_time'].append(selection_time)
            allocation = "Cloud"
            processing_time = self.process_cloud(task)
            # Track cloud allocation for this data type
            if task_type in self.data_type_counts:
                self.data_type_counts[task_type]['cloud'] += 1
            # Print task information only if verbose mode is on
            if self.verbose_output:
                print(f"Task {task.id}: {task_type}, Allocated to {allocation}, Size: {task.size}, Rejected by Fog: Yes, Lifetime: {processing_time:.2f}ms")
            return processing_time
    
    def print_fog_status(self):
        """Print the current status of all fog resources"""
        active_nodes = sum(1 for fog in self.fog_nodes if fog.cumulative_processed > 0)
        total_devices = sum(fog.num_devices for fog in self.fog_nodes)
        committed_devices = sum(self.get_total_node_commitments(fog.name) for fog in self.fog_nodes)
        available_devices = total_devices - committed_devices
        total_storage = sum(fog.total_storage for fog in self.fog_nodes)
        used_storage = sum(fog.used_storage for fog in self.fog_nodes)
        
        print(f"Fog Status: Active Nodes: {active_nodes}/{len(self.fog_nodes)}, "
              f"Available Devices: {available_devices}/{total_devices} ({available_devices/total_devices*100:.1f}%), "
              f"Storage: {used_storage}/{total_storage} ({used_storage/total_storage*100:.1f}%)")
        
        # Print individual node status
        for fog in self.fog_nodes:
            node_commitments = self.get_total_node_commitments(fog.name)
            node_availability = fog.num_devices - node_commitments
            print(f"  - {fog.name}: Processed: {fog.cumulative_processed}, "
                  f"Available Devices: {node_availability}/{fog.num_devices}, "
                  f"Storage: {fog.used_storage}/{fog.total_storage}, "
                  f"Avg Util: {fog.cumulative_utilization:.1f}%, "
                  f"Peak Util: {fog.peak_utilization:.1f}%")
        
        # Print data type distribution metrics for batches
        if hasattr(self, 'data_type_counts'):
            print("\nData Type Distribution:")
            for data_type, counts in self.data_type_counts.items():
                fog_count = counts.get('fog', 0)
                cloud_count = counts.get('cloud', 0)
                total = fog_count + cloud_count
                if total > 0:
                    fog_pct = (fog_count / total) * 100
                    print(f"  {data_type:<15}: Fog: {fog_count} ({fog_pct:.1f}%), Cloud: {cloud_count} ({100-fog_pct:.1f}%)")
        
        print("")

# ========== Core Functions ==========
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
            
            # Determine if task should be a fog candidate based on data type and size
            if item['DataType'] in ['Large', 'Bulk'] and int(item['Size']) > 250:
                fog_candidate = False
            elif item['DataType'] in ['Abrupt', 'LocationBased', 'Medical', 'SmallTextual', 'Multimedia'] and int(item['Size']) > 230:
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

def main():
    try:
        print("Running FCFS Cooperation Policy for full dataset...")
        
        # Use tuple100k.json file only, without modifying the tasks
        filepath = os.path.join(os.getcwd(), 'tuple100k.json')
        if not os.path.exists(filepath):
            print(f"Error: File {filepath} not found")
            exit(1)
        
        print(f"Loading tasks from {filepath}...")
        all_tasks = load_tasks(filepath)
        print(f"Loaded {len(all_tasks)} tasks")
        
        # Sort tasks by arrival time and use all tasks (full dataset)
        sorted_tasks = sorted(all_tasks, key=lambda t: t.arrival_time)
        
        # Allow running with a subset for quicker results (default to full dataset)
        task_limit = len(sorted_tasks)  # Process all tasks
        tasks = sorted_tasks[:task_limit]
        print(f"Processing {len(tasks)} tasks")
        
        results = {}
        
        # Only use FCFSCooperationGateway
        policy_name = "FCFSCooperation"
        print(f"\nRunning {policy_name}...")
        
        # Create gateway directly to have more control
        fog_nodes = [FogNode(cfg) for cfg in FOG_NODES]
    cloud_services = [CloudService(cfg) for cfg in CLOUD_SERVICES]
        gateway = FCFSCooperationGateway(fog_nodes, cloud_services)
        
        # Turn off verbose output for large datasets
        gateway.verbose_output = (len(tasks) <= 1000)
        
        # Initialize result tracking
        results[policy_name] = {
        'fog_times': [],
        'cloud_times': [],
        'queue_delays': [],
        'batch_completion_times': [],
            'fog_nodes': fog_nodes,
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
    
        # Process tasks in batches
        remaining_tasks = tasks.copy()
        batch_counter = 0
        total_fog_count = 0
        total_cloud_count = 0
        
        # Track overall progress
        total_tasks = len(tasks)
        processed_tasks = 0
        start_time = import_time = __import__('time').time()
        
        print(f"Starting processing of {total_tasks} tasks at {__import__('time').strftime('%H:%M:%S')}")
        
        while remaining_tasks:
            batch_counter += 1
            gateway.current_batch = batch_counter
            
            # Get next batch of tasks
            batch_size = min(BATCH_SIZE, len(remaining_tasks))
            batch = remaining_tasks[:batch_size]
            remaining_tasks = remaining_tasks[batch_size:]
            
            print(f"\nProcessing Batch {batch_counter} with {len(batch)} tasks ({processed_tasks}/{total_tasks} processed)")
            
            # Process batch
            batch_start_time = gateway.sim_clock
            fog_count = 0
            cloud_count = 0
            
            for task in tqdm(batch, desc=f"Batch {batch_counter}"):
                task.batch_id = batch_counter
                task.processing_start_time = gateway.sim_clock
                
                # Offload task to appropriate resource
                result = gateway.offload_task(task)
                
                # Track where the task was processed
                if result == 0:
                    fog_count += 1
                else:
                    cloud_count += 1
                
                task.processing_end_time = gateway.sim_clock
                processed_tasks += 1
            
            # Update total counts
            total_fog_count += fog_count
            total_cloud_count += cloud_count
            
            # Calculate batch completion time
            batch_completion_time = gateway.sim_clock - batch_start_time
            results[policy_name]['batch_completion_times'].append(batch_completion_time)
            
            # Print batch summary
            if len(batch) > 0:
                fog_percent = (fog_count / len(batch)) * 100
                cloud_percent = (cloud_count / len(batch)) * 100
                print(f"\nBatch {batch_counter} Summary:")
                print(f"Tasks: Fog = {fog_count} ({fog_percent:.1f}%), Cloud = {cloud_count} ({cloud_percent:.1f}%)")
                print(f"Batch Completion Time: {batch_completion_time:.2f}ms")
                
                # Print overall progress
                elapsed = __import__('time').time() - start_time
                remaining = (elapsed / processed_tasks) * (total_tasks - processed_tasks) if processed_tasks > 0 else 0
                print(f"Overall Progress: {processed_tasks}/{total_tasks} ({processed_tasks/total_tasks*100:.1f}%)")
                print(f"Elapsed Time: {elapsed:.1f}s, Estimated Remaining: {remaining:.1f}s")
                
                # Print data type distribution
                print("\nData Type Distribution (So Far):")
                for data_type, counts in gateway.data_type_counts.items():
                    fog_type_count = counts.get('fog', 0)
                    cloud_type_count = counts.get('cloud', 0)
                    type_total = fog_type_count + cloud_type_count
                    if type_total > 0:
                        fog_type_pct = (fog_type_count / type_total) * 100
                        print(f"  {data_type:<15}: Fog: {fog_type_count} ({fog_type_pct:.1f}%), Cloud: {cloud_type_count} ({100-fog_type_pct:.1f}%)")
                
                # Reset nodes for next batch if needed
                gateway.reset_nodes()
        
        # Print final summary
        print("\n=== Final Processing Summary ===")
        total_processed = total_fog_count + total_cloud_count
        
        print(f"Total Tasks Processed: {total_processed}")
        if total_processed > 0:
            print(f"Fog Tasks: {total_fog_count} ({total_fog_count/total_processed*100:.1f}%)")
            print(f"Cloud Tasks: {total_cloud_count} ({total_cloud_count/total_processed*100:.1f}%)")
        
        # Calculate average processing times
        avg_fog_time = np.mean(gateway.metrics['fog_times']) if gateway.metrics['fog_times'] else 0
        avg_cloud_time = np.mean(gateway.metrics['cloud_times']) if gateway.metrics['cloud_times'] else 0
        
        print(f"\nAverage Processing Times:")
        print(f"Fog: {avg_fog_time:.2f}ms")
        print(f"Cloud: {avg_cloud_time:.2f}ms")
        
        # Print fog node statistics
        print("\nFog Node Processing Statistics:")
        total_fog_processed = sum(fog.cumulative_processed for fog in fog_nodes)
        for fog in fog_nodes:
            # Calculate percentage of fog workload handled by this node
            if total_fog_processed > 0:
                workload_percentage = (fog.cumulative_processed / total_fog_processed) * 100
    else:
                workload_percentage = 0
            
            print(f"{fog.name}: Processed {fog.cumulative_processed} tasks ({workload_percentage:.1f}% of fog workload), "
                  f"Avg Utilization: {fog.cumulative_utilization:.2f}%, "
                  f"Peak Utilization: {fog.peak_utilization:.2f}%")
            
            # Additional utilization details
            if fog.cumulative_processed > 0:
                avg_ram_usage = fog.total_ram_hours / (fog.cumulative_processed * 1e-6)  # Adjusted for scale
                avg_mips_usage = fog.total_mips_hours / (fog.cumulative_processed * 1e-6)
                avg_storage_usage = fog.total_storage_hours / (fog.cumulative_processed * 1e-6)
                
                print(f"  Resource usage per task: RAM: {avg_ram_usage:.2f} units, "
                      f"MIPS: {avg_mips_usage:.2f} units, Storage: {avg_storage_usage:.2f} units")
        
        # Print overall performance metrics
        total_time = __import__('time').time() - start_time
        print(f"\nTotal processing time: {total_time:.2f}s ({total_processed/total_time:.2f} tasks/second)")
        
        # Print data type distribution
        print("\nFinal Data Type Distribution:")
        for data_type, counts in gateway.data_type_counts.items():
            fog_count = counts.get('fog', 0)
            cloud_count = counts.get('cloud', 0)
            total = fog_count + cloud_count
            if total > 0:
                fog_pct = (fog_count / total) * 100
                print(f"  {data_type:<15}: Fog: {fog_count} ({fog_pct:.1f}%), Cloud: {cloud_count} ({100-fog_pct:.1f}%)")
    
    except FileNotFoundError as e:
        print(f"File not found error: {e}")
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
    except KeyError as e:
        print(f"Key error: {e}")
    except Exception as e:
        import traceback
        print(f"An error occurred: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    main()