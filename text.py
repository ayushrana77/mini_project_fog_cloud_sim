import json
import os
import numpy as np
import random
from collections import defaultdict
from dataclasses import dataclass
from tqdm import tqdm
from math import radians, sin, cos, sqrt, atan2
from typing import Dict, List, Optional, Any
import time
from copy import deepcopy

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
        "num_devices": 250
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
        "num_devices": 250
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
        
        # Add variable resource holding time based on task type and size
        # This simulates tasks that hold resources for different durations
        holding_time_factor = 1.0  # Default factor
        
        # Large and Bulk data types hold resources longer
        if task.data_type in ['Large', 'Bulk']:
            holding_time_factor = 2.5
        # Medium-sized data types hold resources moderately longer
        elif task.data_type in ['Multimedia', 'LocationBased'] and task.size > 150:
            holding_time_factor = 1.8
        # Medical data gets priority with shorter holding time
        elif task.data_type == 'Medical':
            holding_time_factor = 0.7
        # Small data types release resources quickly
        elif task.data_type in ['SmallTextual', 'Abrupt'] and task.size < 100:
            holding_time_factor = 0.5
        
        # Adjust processing time with the holding factor
        adjusted_processing_time = processing_time * holding_time_factor
        
        # Update busy time
        completion_time = max(arrival_time, self.busy_until) + adjusted_processing_time + transmission_time
        self.busy_until = completion_time
        
        # Update utilization
        utilization_increase = min(20, (adjusted_processing_time / 1000) * (allocated_mips / self.mips) * 100)
        self.utilization = min(100, self.utilization + utilization_increase)
        
        # Calculate a more meaningful cumulative utilization based on percentage of total capacity used
        # Use weighted average of RAM, MIPS, and storage utilization
        ram_util = (self.ram - self.available_ram) / self.ram * 100
        mips_util = (self.mips - self.available_mips) / self.mips * 100
        storage_util = self.used_storage / self.total_storage * 100
        resource_utilization = (ram_util + mips_util + storage_util) / 3
        
        # Apply exponential moving average to smooth utilization over time
        alpha = 0.3  # Smoothing factor (higher = more weight to new values)
        self.cumulative_utilization = alpha * resource_utilization + (1 - alpha) * self.cumulative_utilization
        
        self.power_log.append(self.calculate_power())
        
        # Schedule resource release with the adjusted time
        self.resource_release_schedule.append({
            'time': completion_time,
            'ram': allocated_ram,
            'mips': allocated_mips,
            'storage': allocated_storage,
            'holding_time_factor': holding_time_factor,  # Store for metrics
            'data_type': task.data_type  # Store for metrics
        })
        
        return 0, adjusted_processing_time + transmission_time, completion_time

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
        # Track device usage statistics
        self.total_device_capacity = sum(fog.num_devices for fog in fog_nodes)
        self.total_devices_used = 0  # Track how many devices are currently being used
        self.max_devices_used = 0  # Track peak device usage
        self.metrics = {
            'fog_times': [],
            'cloud_times': [],
            'node_selection_time': [],
            'cloud_selection_time': [],
            'queue_delays': [],
            'device_usage': []  # Track device usage over time
        }
        
        # Initialize device commitments for each fog node
        for node in fog_nodes:
            self.device_commitments[node.name] = {}
        
    def reset_nodes(self):
        """Reset all nodes for a new batch."""
        # Store current device usage for metrics
        self.metrics['device_usage'].append(self.total_devices_used)
        
        for node in self.fog_nodes:
            node.reset()
        for service in self.cloud_services:
            service.reset()
        
        # Release resources from 10 batches ago
        expired_batch = self.current_batch - 10
        if expired_batch > 0:
            if expired_batch in self.batch_assignments:
                # Calculate devices to release
                released_devices = 0
                for node_name, task_ids in self.batch_assignments[expired_batch].items():
                    released_devices += len(task_ids)
                
                # Update device count
                self.total_devices_used -= released_devices
                if self.total_devices_used < 0:
                    self.total_devices_used = 0
                
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
    
    def get_remaining_device_capacity(self):
        """Get the number of devices still available."""
        return self.total_device_capacity - self.total_devices_used
        
    def is_fog_available(self, fog, task, current_batch):
        """Check if a fog node can accept a task based on current commitments."""
        if not task.fog_candidate:
            return False
        
        # Check if we've reached global device capacity
        if self.total_devices_used >= self.total_device_capacity:
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
        
        # Update global device usage
        self.total_devices_used += 1
        # Track peak device usage
        self.max_devices_used = max(self.max_devices_used, self.total_devices_used)

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
            
            # Print current device usage
            print(f"  Current Device Usage: {self.total_devices_used}/{self.total_device_capacity} ({self.total_devices_used/self.total_device_capacity*100:.1f}%)")
            print(f"  Peak Device Usage: {self.max_devices_used}/{self.total_device_capacity} ({self.max_devices_used/self.total_device_capacity*100:.1f}%)")
        
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
        
        # Step 2: Search for a valid fog node (FCFS order)
        fog_processed = False
        
        # For Medium-size tasks, prioritize nodes with more availability
        if (task.data_type in ['Large', 'Bulk'] and task.size > 200) or \
           (task.data_type in ['Abrupt', 'LocationBased', 'Medical', 'SmallTextual', 'Multimedia'] and task.size > 190):
            # Sort fog nodes by available resources (most available first)
            sorted_nodes = sorted(self.fog_nodes, 
                                 key=lambda f: (f.available_ram / f.ram + 
                                              f.available_mips / f.mips + 
                                              (f.total_storage - f.used_storage) / f.total_storage) / 3,
                                 reverse=True)
        else:
            # For smaller tasks, use default order
            sorted_nodes = self.fog_nodes
        
        # Try each node in the determined order
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
                  f"Utilization: {fog.cumulative_utilization:.1f}%")
        
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

class FCFSGateway(BaseGateway):
    def __init__(self, fog_nodes, cloud_services):
        super().__init__(fog_nodes, cloud_services)
        self.data_type_counts = {
            'Abrupt': {'fog': 0, 'cloud': 0},
            'Large': {'fog': 0, 'cloud': 0},
            'LocationBased': {'fog': 0, 'cloud': 0},
            'Bulk': {'fog': 0, 'cloud': 0},
            'Medical': {'fog': 0, 'cloud': 0},
            'SmallTextual': {'fog': 0, 'cloud': 0},
            'Multimedia': {'fog': 0, 'cloud': 0}
        }
        self.verbose_output = False

    def offload_task(self, task):
        """Algorithm 2: Global Gateway With FCFS Tuples and No Cooperation Policy"""
        selection_time = 0
        task_type = task.data_type
        allocation = "Unknown"
        
        # Step 1: Check if task involves bulk data
        if self.is_bulk_data(task):
            self.metrics['node_selection_time'].append(selection_time)
            allocation = "Cloud"
            processing_time = self.process_cloud(task)
            if task_type in self.data_type_counts:
                self.data_type_counts[task_type]['cloud'] += 1
            if self.verbose_output:
                print(f"Task {task.id}: {task_type}, Allocated to {allocation}, Size: {task.size}, Lifetime: {processing_time:.2f}ms")
            return processing_time
        
        # Step 2: Search for a valid fog node (FCFS order)
        fog_processed = False
        
        # For Medium-size tasks, prioritize nodes with more availability
        if (task.data_type in ['Large', 'Bulk'] and task.size > 200) or \
           (task.data_type in ['Abrupt', 'LocationBased', 'Medical', 'SmallTextual', 'Multimedia'] and task.size > 190):
            # Sort fog nodes by available resources (most available first)
            sorted_nodes = sorted(self.fog_nodes, 
                                 key=lambda f: (f.available_ram / f.ram + 
                                              f.available_mips / f.mips + 
                                              (f.total_storage - f.used_storage) / f.total_storage) / 3,
                                 reverse=True)
        else:
            # For smaller tasks, use default FCFS order
            sorted_nodes = self.fog_nodes
        
        # Try each node in the determined order
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
                if task_type in self.data_type_counts:
                    self.data_type_counts[task_type]['fog'] += 1
                if self.verbose_output:
                    print(f"Task {task.id}: {task_type}, Allocated to {allocation}, Size: {task.size}, Lifetime: {p_time:.2f}ms")
                    # Print fog resource status
                    self.print_fog_status()
                return 0
        
        # Step 3: If no valid fog node, try one more time with relaxed constraints
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
                    if task_type in self.data_type_counts:
                        self.data_type_counts[task_type]['fog'] += 1
                    if self.verbose_output:
                        print(f"Task {task.id}: {task_type}, Allocated to {allocation}, Size: {task.size}, Lifetime: {p_time:.2f}ms")
                        # Print fog resource status
                        self.print_fog_status()
                    return 0
        
        # Step 4: If still no fog node available, assign to cloud
        if not fog_processed:
            self.metrics['node_selection_time'].append(selection_time)
            allocation = "Cloud"
            processing_time = self.process_cloud(task)
            if task_type in self.data_type_counts:
                self.data_type_counts[task_type]['cloud'] += 1
            if self.verbose_output:
                print(f"Task {task.id}: {task_type}, Allocated to {allocation}, Size: {task.size}, Lifetime: {processing_time:.2f}ms")
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
                  f"Utilization: {fog.cumulative_utilization:.1f}%")
        
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

class RandomGateway(BaseGateway):
    def __init__(self, fog_nodes, cloud_services):
        super().__init__(fog_nodes, cloud_services)
        self.data_type_counts = {
            'Abrupt': {'fog': 0, 'cloud': 0},
            'Large': {'fog': 0, 'cloud': 0},
            'LocationBased': {'fog': 0, 'cloud': 0},
            'Bulk': {'fog': 0, 'cloud': 0},
            'Medical': {'fog': 0, 'cloud': 0},
            'SmallTextual': {'fog': 0, 'cloud': 0},
            'Multimedia': {'fog': 0, 'cloud': 0}
        }
        self.verbose_output = False

    def offload_task(self, task):
        """Algorithm 4: Global Gateway With Random Tuples and No Cooperation Policy"""
        selection_time = 0
        task_type = task.data_type
        allocation = "Unknown"
        
        # Step 1: Check if task involves bulk data
        if self.is_bulk_data(task):
            self.metrics['node_selection_time'].append(selection_time)
            allocation = "Cloud"
            processing_time = self.process_cloud(task)
            if task_type in self.data_type_counts:
                self.data_type_counts[task_type]['cloud'] += 1
            if self.verbose_output:
                print(f"Task {task.id}: {task_type}, Allocated to {allocation}, Size: {task.size}, Lifetime: {processing_time:.2f}ms")
            return processing_time
        
        # Step 2: Search for a valid fog node (Random order)
        fog_processed = False
        
        # For Medium-size tasks, prioritize nodes with more availability
        if (task.data_type in ['Large', 'Bulk'] and task.size > 200) or \
           (task.data_type in ['Abrupt', 'LocationBased', 'Medical', 'SmallTextual', 'Multimedia'] and task.size > 190):
            # Sort fog nodes by available resources (most available first), then randomize
            available_nodes = sorted(self.fog_nodes, 
                                 key=lambda f: (f.available_ram / f.ram + 
                                              f.available_mips / f.mips + 
                                              (f.total_storage - f.used_storage) / f.total_storage) / 3,
                                 reverse=True)
            # Add some randomization while still prioritizing resources
            if len(available_nodes) > 1:
                # Keep the first half sorted by resources, randomize the second half
                half_point = max(1, len(available_nodes) // 2)
                available_nodes = available_nodes[:half_point] + random.sample(available_nodes[half_point:], len(available_nodes) - half_point)
        else:
            # For smaller tasks, use random order
            available_nodes = self.fog_nodes.copy()
            random.shuffle(available_nodes)
        
        # Try each node in the determined order
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
                if task_type in self.data_type_counts:
                    self.data_type_counts[task_type]['fog'] += 1
                if self.verbose_output:
                    print(f"Task {task.id}: {task_type}, Allocated to {allocation}, Size: {task.size}, Lifetime: {p_time:.2f}ms")
                    # Print fog resource status
                    self.print_fog_status()
                return 0
        
        # Step 3: If no valid fog node, try one more time with relaxed constraints
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
                    if task_type in self.data_type_counts:
                        self.data_type_counts[task_type]['fog'] += 1
                    if self.verbose_output:
                        print(f"Task {task.id}: {task_type}, Allocated to {allocation}, Size: {task.size}, Lifetime: {p_time:.2f}ms")
                        # Print fog resource status
                        self.print_fog_status()
                    return 0
        
        # Step 4: If still no fog node available, assign to cloud
        if not fog_processed:
            self.metrics['node_selection_time'].append(selection_time)
            allocation = "Cloud"
            processing_time = self.process_cloud(task)
            if task_type in self.data_type_counts:
                self.data_type_counts[task_type]['cloud'] += 1
            if self.verbose_output:
                print(f"Task {task.id}: {task_type}, Allocated to {allocation}, Size: {task.size}, Lifetime: {processing_time:.2f}ms")
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
                  f"Utilization: {fog.cumulative_utilization:.1f}%")
        
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

class RandomCooperationGateway(RandomGateway):
    def __init__(self, fog_nodes, cloud_services):
        super().__init__(fog_nodes, cloud_services)
        self.verbose_output = False  # Control detailed task-level output

    def offload_task(self, task):
        """Algorithm 3: Global Gateway With Random Tuples and Cooperation Policy"""
        selection_time = 0
        rejected_by_fog = False
        reassigned = False
        task_type = task.data_type
        allocation = "Unknown"
        
        # Step 1: Check if task involves bulk data
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
        
        # Step 2: Search for a valid fog node (Random order)
        fog_processed = False
        
        # For Medium-size tasks, prioritize nodes with more availability
        if (task.data_type in ['Large', 'Bulk'] and task.size > 200) or \
           (task.data_type in ['Abrupt', 'LocationBased', 'Medical', 'SmallTextual', 'Multimedia'] and task.size > 190):
            # Sort fog nodes by available resources (most available first), then randomize
            available_nodes = sorted(self.fog_nodes, 
                                 key=lambda f: (f.available_ram / f.ram + 
                                              f.available_mips / f.mips + 
                                              (f.total_storage - f.used_storage) / f.total_storage) / 3,
                                 reverse=True)
            # Add some randomization while still prioritizing resources
            if len(available_nodes) > 1:
                # Keep the first half sorted by resources, randomize the second half
                half_point = max(1, len(available_nodes) // 2)
                available_nodes = available_nodes[:half_point] + random.sample(available_nodes[half_point:], len(available_nodes) - half_point)
        else:
            # For smaller tasks, use random order
            available_nodes = self.fog_nodes.copy()
            random.shuffle(available_nodes)
        
        # Try each node in the determined order
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
        
        # Complete random order for cooperation phase
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
                allocation = f"Fog ({fog.name}) [cooperation]"
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
                  f"Utilization: {fog.cumulative_utilization:.1f}%")
        
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

def run_algorithm(gateway, tasks, algorithm_name):
    """Run a specific algorithm and return its results"""
    print(f"\nRunning {algorithm_name}...")
    
    # Initialize result tracking
    results = {
        'fog_times': [],
        'cloud_times': [],
        'queue_delays': [],
        'batch_completion_times': [],
        'node_selection_time': [],
        'cloud_selection_time': [],
        'batch_metrics': {
            'fog_times': [],
            'cloud_times': [],
            'queue_delays': [],
            'fog_utilization': [],
            'cloud_utilization': []
        },
        'holding_time_metrics': {
            'Large': 0,
            'Bulk': 0,
            'Multimedia': 0,
            'LocationBased': 0,
            'Medical': 0,
            'SmallTextual': 0,
            'Abrupt': 0
        },
        'resource_holding_fog_vs_cloud': {
            'short_holding_fog': 0,
            'medium_holding_fog': 0,
            'long_holding_fog': 0,
            'short_holding_cloud': 0,
            'medium_holding_cloud': 0,
            'long_holding_cloud': 0
        }
    }
    
    # Process tasks in batches
    remaining_tasks = tasks.copy()
    batch_counter = 0
    total_fog_count = 0
    total_cloud_count = 0
    processed_task_ids = set()  # Track processed task IDs
    
    # Track overall progress
    total_tasks = len(tasks)
    processed_tasks = 0
    start_time = __import__('time').time()
    
    print(f"Starting processing of {total_tasks} tasks at {__import__('time').strftime('%H:%M:%S')}")
    
    while remaining_tasks:
        batch_counter += 1
        gateway.current_batch = batch_counter
        
        # For random algorithms, shuffle remaining tasks before selecting batch
        if "Random" in algorithm_name:
            random.shuffle(remaining_tasks)
        
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
            # Skip if task was already processed
            if task.id in processed_task_ids:
                continue
            
            task.batch_id = batch_counter
            task.processing_start_time = gateway.sim_clock
            
            # Determine holding category for tracking
            holding_category = 'medium'
            if task.data_type in ['Large', 'Bulk'] or (task.data_type in ['Multimedia', 'LocationBased'] and task.size > 150):
                holding_category = 'long'
            elif task.data_type in ['SmallTextual', 'Abrupt'] and task.size < 100 or task.data_type == 'Medical':
                holding_category = 'short'
            
            # Offload task to appropriate resource
            result = gateway.offload_task(task)
            
            # Track where the task was processed
            if result == 0:
                fog_count += 1
                # Update resource holding metrics for fog
                results['resource_holding_fog_vs_cloud'][f'{holding_category}_holding_fog'] += 1
            else:
                cloud_count += 1
                # Update resource holding metrics for cloud
                results['resource_holding_fog_vs_cloud'][f'{holding_category}_holding_cloud'] += 1
            
            task.processing_end_time = gateway.sim_clock
            processed_tasks += 1
            processed_task_ids.add(task.id)  # Mark task as processed
        
        # Update total counts
        total_fog_count += fog_count
        total_cloud_count += cloud_count
        
        # Calculate batch completion time
        batch_completion_time = gateway.sim_clock - batch_start_time
        results['batch_completion_times'].append(batch_completion_time)
        
        # Print batch summary
        if len(batch) > 0:
            fog_percent = (fog_count / len(batch)) * 100
            cloud_percent = (cloud_count / len(batch)) * 100
            print(f"\nBatch {batch_counter} Summary:")
            print(f"Tasks: Fog = {fog_count} ({fog_percent:.1f}%), Cloud = {cloud_count} ({cloud_percent:.1f}%)")
            print(f"Batch Completion Time: {batch_completion_time:.2f}ms")
            
            # Print holding time distribution for this batch
            print(f"\nResource Holding Distribution (Batch {batch_counter}):")
            print(f"  Short-holding tasks: Fog={results['resource_holding_fog_vs_cloud']['short_holding_fog']}, " 
                  f"Cloud={results['resource_holding_fog_vs_cloud']['short_holding_cloud']}")
            print(f"  Medium-holding tasks: Fog={results['resource_holding_fog_vs_cloud']['medium_holding_fog']}, "
                  f"Cloud={results['resource_holding_fog_vs_cloud']['medium_holding_cloud']}")
            print(f"  Long-holding tasks: Fog={results['resource_holding_fog_vs_cloud']['long_holding_fog']}, "
                  f"Cloud={results['resource_holding_fog_vs_cloud']['long_holding_cloud']}")
        
        # Track fog resource saturation
        fog_nodes_full = sum(1 for fog in gateway.fog_nodes if fog.available_ram < fog.ram * 0.2)
        fog_nodes_heavy = sum(1 for fog in gateway.fog_nodes if fog.available_ram < fog.ram * 0.5 and fog.available_ram >= fog.ram * 0.2)
        
        print(f"\nFog Resource Saturation:")
        print(f"  Full nodes: {fog_nodes_full}/{len(gateway.fog_nodes)}")
        print(f"  Heavily loaded nodes: {fog_nodes_heavy}/{len(gateway.fog_nodes)}")
        
        # Add detailed resource utilization report
        print("\nDetailed Resource Utilization:")
        print(f"{'Fog Node':<15} {'RAM':<25} {'MIPS':<25} {'Storage':<25}")
        print(f"{'':<15} {'Available':<12} {'Used':<12} {'Available':<12} {'Used':<12} {'Available':<12} {'Used':<12}")
        print("-" * 90)
        
        total_ram = 0
        total_mips = 0
        total_storage = 0
        available_ram = 0
        available_mips = 0
        available_storage = 0
        
        for fog in gateway.fog_nodes:
            # Calculate used values
            used_ram = fog.ram - fog.available_ram
            used_mips = fog.mips - fog.available_mips
            available_storage = fog.total_storage - fog.used_storage
            
            # Add to totals
            total_ram += fog.ram
            total_mips += fog.mips
            total_storage += fog.total_storage
            available_ram += fog.available_ram
            available_mips += fog.available_mips
            available_storage += available_storage
            
            # Format values for better readability
            ram_available = f"{fog.available_ram/1024:.1f}KB ({fog.available_ram/fog.ram*100:.1f}%)"
            ram_used = f"{used_ram/1024:.1f}KB ({used_ram/fog.ram*100:.1f}%)"
            
            mips_available = f"{fog.available_mips/1000:.1f}K ({fog.available_mips/fog.mips*100:.1f}%)"
            mips_used = f"{used_mips/1000:.1f}K ({used_mips/fog.mips*100:.1f}%)"
            
            storage_available = f"{available_storage/1024:.1f}KB ({available_storage/fog.total_storage*100:.1f}%)"
            storage_used = f"{fog.used_storage/1024:.1f}KB ({fog.used_storage/fog.total_storage*100:.1f}%)"
            
            print(f"{fog.name:<15} {ram_available:<12} {ram_used:<12} {mips_available:<12} {mips_used:<12} {storage_available:<12} {storage_used:<12}")
        
        # Add total row
        print("-" * 90)
        total_used_ram = total_ram - available_ram
        total_used_mips = total_mips - available_mips
        total_used_storage = total_storage - available_storage
        
        ram_available_pct = (available_ram / total_ram * 100) if total_ram > 0 else 0
        ram_used_pct = (total_used_ram / total_ram * 100) if total_ram > 0 else 0
        
        mips_available_pct = (available_mips / total_mips * 100) if total_mips > 0 else 0
        mips_used_pct = (total_used_mips / total_mips * 100) if total_mips > 0 else 0
        
        storage_available_pct = (available_storage / total_storage * 100) if total_storage > 0 else 0
        storage_used_pct = (total_used_storage / total_storage * 100) if total_storage > 0 else 0
        
        print(f"{'TOTAL':<15} {available_ram/1024:.1f}KB ({ram_available_pct:.1f}%) {total_used_ram/1024:.1f}KB ({ram_used_pct:.1f}%) {available_mips/1000:.1f}K ({mips_available_pct:.1f}%) {total_used_mips/1000:.1f}K ({mips_used_pct:.1f}%) {available_storage/1024:.1f}KB ({storage_available_pct:.1f}%) {total_used_storage/1024:.1f}KB ({storage_used_pct:.1f}%)")
        
        # Print cloud service utilization
        print("\nCloud Service Load:")
        for cloud in gateway.cloud_services:
            print(f"  {cloud.name}: {cloud.current_load:.1f}% load, Busy until: {cloud.busy_until:.2f}ms")
        
        # Print overall progress
        elapsed = __import__('time').time() - start_time
        remaining = (elapsed / processed_tasks) * (total_tasks - processed_tasks) if processed_tasks > 0 else 0
        print(f"\nOverall Progress: {processed_tasks}/{total_tasks} ({processed_tasks/total_tasks*100:.1f}%)")
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
    
    # Calculate final metrics
    results['total_fog_count'] = total_fog_count
    results['total_cloud_count'] = total_cloud_count
    results['total_time'] = __import__('time').time() - start_time
    results['avg_fog_time'] = np.mean(gateway.metrics['fog_times']) if gateway.metrics['fog_times'] else 0
    results['avg_cloud_time'] = np.mean(gateway.metrics['cloud_times']) if gateway.metrics['cloud_times'] else 0
    
    # Calculate additional metrics for resource holding analysis
    total_fog = sum(results['resource_holding_fog_vs_cloud'][f'{h}_holding_fog'] for h in ['short', 'medium', 'long'])
    total_cloud = sum(results['resource_holding_fog_vs_cloud'][f'{h}_holding_cloud'] for h in ['short', 'medium', 'long'])
    
    for holding_type in ['short', 'medium', 'long']:
        fog_count = results['resource_holding_fog_vs_cloud'][f'{holding_type}_holding_fog']
        cloud_count = results['resource_holding_fog_vs_cloud'][f'{holding_type}_holding_cloud']
        total_holding = fog_count + cloud_count
        
        results[f'{holding_type}_holding_fog_pct'] = (fog_count / total_holding * 100) if total_holding > 0 else 0
        results[f'{holding_type}_holding_cloud_pct'] = (cloud_count / total_holding * 100) if total_holding > 0 else 0
    
    return results

def main():
    """Run the experiment"""
    print("\n=== Running all algorithms with limited task dataset ===")
    print("\nLoading tasks from Tuple10.json (smaller test dataset)...")
    
    # Load sorted tasks
    with open("tuples.json", "r") as f:
        data = json.load(f)
    
    # Create Task objects and sort by arrival time
    tasks = [Task(**task) for task in data]
    sorted_tasks = sorted(tasks, key=lambda x: x.arrival_time)
    
    # Limit to 5000 tasks for faster processing, but in this case we use all from Tuple10
    task_limit = len(sorted_tasks)
    tasks = sorted_tasks[:task_limit]
    print(f"Loaded {len(sorted_tasks)} tasks for processing")
    
    # Initialize fog nodes to get num_devices
    fog_nodes = [FogNode(config) for config in FOG_NODES]
    total_fog_capacity = sum(node.num_devices for node in fog_nodes)
    print(f"Total fog device capacity: {total_fog_capacity} devices")
    print(f"Average tasks per device: {task_limit/total_fog_capacity:.2f} tasks/device")
    
    # Define batch size based on node capacity
    batch_size = min(BATCH_SIZE, total_fog_capacity // 2)  # Use at most half capacity per batch
    print(f"Using batch size: {batch_size} (based on fog capacity)")
    
    # Set verbose output for the smaller dataset
    verbose = True
    print("\nProcessing tasks with various algorithms...")
    print(f"Fog Nodes: {len(FOG_NODES)}, Cloud Services: {len(CLOUD_SERVICES)}")
    
    # Define algorithms to run
    algorithms = ["FCFSCooperation", "FCFSNoCooperation", "RandomCooperation", "RandomNoCooperation"]
    
    # Run all algorithms
    for algorithm_name in algorithms:
        print(f"\n=== Running {algorithm_name} algorithm ===")
        start_time = time.time()
        
        # Initialize algorithm-specific gateway
        if algorithm_name == "FCFSCooperation":
            gateway = FCFSCooperationGateway([FogNode(config) for config in FOG_NODES], 
                                          [CloudService(config) for config in CLOUD_SERVICES])
        elif algorithm_name == "FCFSNoCooperation":
            gateway = FCFSGateway([FogNode(config) for config in FOG_NODES], 
                               [CloudService(config) for config in CLOUD_SERVICES])
        elif algorithm_name == "RandomCooperation":
            gateway = RandomCooperationGateway([FogNode(config) for config in FOG_NODES], 
                                            [CloudService(config) for config in CLOUD_SERVICES])
        elif algorithm_name == "RandomNoCooperation":
            gateway = RandomGateway([FogNode(config) for config in FOG_NODES], 
                                 [CloudService(config) for config in CLOUD_SERVICES])
        else:
            print(f"Unknown algorithm: {algorithm_name}")
            continue
            
        # Set batch size and verbose mode
        gateway.batch_size = batch_size
        gateway.verbose_output = verbose
        
        # Process all tasks
        fog_count = 0
        cloud_count = 0
        remaining_tasks = tasks.copy()
        
        # Process tasks in batches
        while remaining_tasks:
            # Get next batch
            batch_size = min(gateway.batch_size, len(remaining_tasks))
            current_batch = remaining_tasks[:batch_size]
            remaining_tasks = remaining_tasks[batch_size:]
            
            # Process the batch
            completion_time = gateway.process_batch(current_batch)
            
            # Count tasks by processing location
            for task in current_batch:
                if task.id in gateway.processed_tasks:
                    continue
                gateway.processed_tasks.add(task.id)
                
                if hasattr(task, 'is_cloud_served') and task.is_cloud_served:
                    cloud_count += 1
                else:
                    fog_count += 1
            
            # Reset nodes for the next batch
            gateway.reset_nodes()
        
        # Record metrics
        runtime = time.time() - start_time
        
        # Display algorithm summary
        total_tasks = fog_count + cloud_count
        fog_percent = (fog_count / total_tasks * 100) if total_tasks > 0 else 0
        cloud_percent = (cloud_count / total_tasks * 100) if total_tasks > 0 else 0
        
        print(f"\nAlgorithm: {algorithm_name}")
        print(f"Total execution time: {runtime:.2f} seconds")
        print(f"Total tasks processed: {total_tasks}")
        print(f"Fog Processing: {fog_count} tasks ({fog_percent:.1f}%)")
        print(f"Cloud Processing: {cloud_count} tasks ({cloud_percent:.1f}%)")
        
        if gateway.metrics['fog_times']:
            avg_fog_time = sum(gateway.metrics['fog_times']) / len(gateway.metrics['fog_times'])
            print(f"Average Fog Processing Time: {avg_fog_time:.2f} ms")
            
        if gateway.metrics['cloud_times']:
            avg_cloud_time = sum(gateway.metrics['cloud_times']) / len(gateway.metrics['cloud_times'])
            print(f"Average Cloud Processing Time: {avg_cloud_time:.2f} ms")
            
        print(f"Peak Device Usage: {gateway.max_devices_used}/{total_fog_capacity} ({gateway.max_devices_used/total_fog_capacity*100:.1f}%)")
    
    print("\nAll algorithms completed.")
    
if __name__ == "__main__":
    try:
        print("Starting script execution...")
        main()
        print("Script completed successfully.")
    except Exception as e:
        import traceback
        print(f"Error during execution: {e}")
        traceback.print_exc()
        print("Script terminated with error.")