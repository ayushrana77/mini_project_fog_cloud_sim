import json
import os
import numpy as np
import random
import math
from collections import defaultdict
from dataclasses import dataclass
from tqdm import tqdm
from math import radians, sin, cos, sqrt, atan2
from typing import Dict, List, Optional, Any

# ==========================================================
# Real-world Factors Simulation Implementation
# ==========================================================
# This implementation adds the following real-world factors:
#
# 1. Dynamic Processing Time:
#    - Task processing times vary based on node utilization
#    - Background system load affects available resources
#    - Random variations simulate unpredictable system behaviors
#
# 2. Network Variability:
#    - Transmission speeds vary based on simulated network congestion
#    - Congestion changes over time with some continuity
#    - System-wide congestion events occur randomly 
#
# 3. Task Migration:
#    - Tasks are redirected to cloud when fog nodes are overloaded
#    - Network congestion can force tasks to migrate
#    - Critical tasks get special handling during congestion
#
# 4. Resource Contention:
#    - Background processes consume resources unpredictably
#    - Higher node utilization leads to longer processing times
#    - Time-of-day patterns affect cloud resource availability
# ==========================================================

# ========== Configuration Section ==========
# Batch Processing Configuration
BATCH_SIZE = 1000
BATCHES_BEFORE_RESET = 10
BATCH_RESET_DELAY = 100  # ms between batch resets

# Real-world Variability Factors
NETWORK_CONGESTION_MIN = 0.6  # Network throughput multiplier (min) - increased variability
NETWORK_CONGESTION_MAX = 1.5  # Network throughput multiplier (max) - increased variability
PROCESSING_VARIATION_MIN = 0.7  # Processing time multiplier (min) - increased variability
PROCESSING_VARIATION_MAX = 1.8  # Processing time multiplier (max) - increased variability
BACKGROUND_LOAD_MIN = 0.0  # Additional background load (min)
BACKGROUND_LOAD_MAX = 0.25  # Additional background load (max) - increased max
HIGH_UTILIZATION_THRESHOLD = 85  # Force cloud migration when node utilization exceeds this
NETWORK_CONGESTION_THRESHOLD = 1.15  # Force cloud migration when network congestion exceeds this

# Added randomness factors for baseline calculations
BASE_VARIATION_FACTOR = 0.4  # How much baseline processing times can vary

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
    processor_node: str = ""
    
    def should_go_to_fog(self):
        """Determine if this task should go to fog nodes first."""
        return self.fog_candidate
        
    def is_small_task(self):
        """Determines if this is a small task that should go to cloud directly"""
        return self.data_type in ['Small', 'Text', 'Sensor', 'IoT'] and self.size < 100
        
    def estimate_fog_processing_time(self, fog_mips, available_efficiency=1.0):
        """Estimate processing time on fog based on task size and requirements"""
        # Calculate base size factor with the updated formula and more randomness
        size_impact_random = random.uniform(0.7, 1.5)
        size_factor = 1.0 + (self.size / 150) * 0.8 * size_impact_random
        
        # Base processing time estimate with randomness
        base_time = (self.mips / fog_mips) * (1000 / available_efficiency) * size_factor
        
        # Return a range to account for enhanced variation
        base_variation = random.uniform(0.2, 0.6)  # Increased base variation
        min_time = base_time * PROCESSING_VARIATION_MIN * 0.8 * (1.0 - base_variation)
        max_time = base_time * PROCESSING_VARIATION_MAX * 1.2 * (1.0 + base_variation)
        
        return (min_time, max_time)
        
    def estimate_transmission_time(self, bandwidth, network_factor=1.0):
        """Estimate transmission time based on task size and network conditions"""
        # Direct calculation based on size and bandwidth
        base_time = (self.size / bandwidth) * 1000
        
        # Apply network factor with much more variability
        network_variation = random.uniform(0.3, 0.7)  # Increased network variation
        min_adjusted_time = base_time * network_factor * (0.6 - network_variation)  # 30-60% less than expected
        max_adjusted_time = base_time * network_factor * (1.6 + network_variation)  # 60-130% more than expected
        
        return (min_adjusted_time, max_adjusted_time)
        
    def estimate_cloud_processing_time(self, cloud_mips=15000):
        """Estimate processing time on cloud based on task size"""
        # Base processing with wide variation
        base_time = 3000 + random.uniform(-1000, 1200)
        
        # Apply size factor
        size_factor = 1.0 + (self.size / 200) * 0.9
        
        # Calculate processing time range
        min_time = base_time * size_factor * PROCESSING_VARIATION_MIN * (1.0 - BASE_VARIATION_FACTOR)
        max_time = base_time * size_factor * PROCESSING_VARIATION_MAX * (1.0 + BASE_VARIATION_FACTOR)
        
        return (min_time, max_time)

    def get_size_category(self):
        """Categorize task size for reporting"""
        if self.size < 100:
            return "Small"
        elif self.size < 200:
            return "Medium"
        else:
            return "Large"

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
        self.power_log = [50]
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
        # New variables for real-world factors
        self.network_congestion = 1.0  # Default: no congestion
        self.background_load = 0.0     # Default: no background load
        self.last_congestion_update = 0.0
        self.congestion_update_interval = 10.0  # Update network congestion every 10ms

    def calculate_power(self):
        """Calculate power consumption based on utilization"""
        return 50 + (self.utilization * 0.3)

    def update_network_congestion(self, current_time):
        """Simulate real-world network congestion that changes over time"""
        if current_time - self.last_congestion_update > self.congestion_update_interval:
            # Update network congestion with some randomness but also continuity
            change_factor = random.uniform(0.9, 1.1)
            self.network_congestion = max(NETWORK_CONGESTION_MIN, 
                                         min(NETWORK_CONGESTION_MAX, 
                                             self.network_congestion * change_factor))
            # Update background load
            self.background_load = random.uniform(BACKGROUND_LOAD_MIN, BACKGROUND_LOAD_MAX)
            self.last_congestion_update = current_time
        return self.network_congestion

    def can_accept_task(self, task, current_time):
        """Check if node can accept task based on resource availability and network conditions"""
        # Update network congestion
        self.update_network_congestion(current_time)
        
        # Check if network congestion is too high for this task
        if self.network_congestion > NETWORK_CONGESTION_THRESHOLD and task.size > 150:
            return False  # High congestion rejects larger tasks
            
        # If node utilization is already too high, reject the task
        if self.utilization > HIGH_UTILIZATION_THRESHOLD:
            return False
            
        # Allow tasks to be accepted if resources are available, even if busy
        return (self.available_ram >= task.ram and 
                self.available_mips >= task.mips and 
                (self.total_storage - self.used_storage) >= task.size and
                len(self.queue) < self.max_queue_size)

    def process(self, task, arrival_time):
        """Process task with real-world variability in execution times."""
        self.total_processed += 1
        self.cumulative_processed += 1  # Update cumulative count
        self.sim_clock = arrival_time
        
        # Update network conditions
        self.update_network_congestion(arrival_time)
        
        # Calculate actual resource allocation (may be relaxed)
        allocated_ram = min(task.ram, self.available_ram)
        allocated_mips = min(task.mips, self.available_mips)
        allocated_storage = min(task.size, self.total_storage - self.used_storage)
        
        # Update resources
        self.available_ram -= allocated_ram
        self.available_mips -= allocated_mips
        self.used_storage += allocated_storage
        
        # Apply real-world variability to processing
        # 1. Calculate base processing time with resource efficiency
        efficiency = min(allocated_mips / task.mips, allocated_ram / task.ram)
        
        # Add significant random baseline variation for each processing task
        base_randomness = random.uniform(0.6, 1.8)  # Much wider baseline variation
        
        # Incorporate task size directly into processing time calculation with stronger impact
        # Larger tasks take proportionally longer to process with random variation
        size_impact_random = random.uniform(0.7, 1.5)  # Random size impact factor
        size_factor = 1.0 + (task.size / 150) * 0.8 * size_impact_random
        
        # Calculate base processing time with all factors and stronger randomization
        base_processing_time = (task.mips / self.mips) * (1000 / efficiency) * size_factor * base_randomness
        
        # 2. Apply variations due to system load and background processes
        # Add more variability to load calculations
        load_random = random.uniform(0.8, 1.6)  # Increased load randomness
        load_factor = (1.0 + self.background_load + (self.utilization / 100) * 0.3) * load_random
        
        # Wider processing variation range
        processing_variation = random.uniform(PROCESSING_VARIATION_MIN * 0.8, PROCESSING_VARIATION_MAX * 1.2)
        processing_time = base_processing_time * load_factor * processing_variation
        
        # 3. Calculate variable transmission time with more randomness
        # Transmission time is directly proportional to task size with increased randomness
        transmission_randomness = random.uniform(0.6, 1.8)  # Increased transmission randomness
        base_transmission_time = (task.size / self.down_bw) * 1000  # Convert to ms
        transmission_time = base_transmission_time * self.network_congestion * transmission_randomness
        
        # Calculate queue delay with more randomness
        queue_randomness = random.uniform(0.8, 1.7)  # Increased queue randomness
        base_queue_delay = max(0, self.busy_until - arrival_time)
        queue_delay = base_queue_delay * queue_randomness
        
        # Store details in task object for analysis
        task.queue_delay = queue_delay
        task.internal_processing_time = processing_time
        
        # Update busy time
        completion_time = max(arrival_time, self.busy_until) + processing_time + transmission_time
        self.busy_until = completion_time
        
        # Update utilization - influenced by both allocated resources and processing time
        utilization_increase = min(25, (processing_time / 1000) * (allocated_mips / self.mips) * 100)
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
        
        # Release resources at completion time
        self.resource_release_schedule.append({
            'time': completion_time,
            'ram': allocated_ram,
            'mips': allocated_mips,
            'storage': allocated_storage
        })
        
        return queue_delay, processing_time + transmission_time, completion_time

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
        self.power_log = [50]
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
        # New real-world factors
        self.network_latency = 1.0  # Base network latency multiplier
        self.last_latency_update = 0.0
        self.latency_update_interval = 20.0  # Update network latency less frequently than fog nodes
        self.cloud_congestion = random.uniform(0.9, 1.1)  # Cloud resource congestion

    def reset(self):
        """Reset cloud service state"""
        self.busy_until = 0.0
        self.current_load = random.uniform(60, 80)
        self.queue = []
        self.network_latency = 1.0
        self.cloud_congestion = random.uniform(0.9, 1.1)

    def update_network_conditions(self, current_time):
        """Update network conditions to simulate real-world variability"""
        if current_time - self.last_latency_update > self.latency_update_interval:
            # Network latency changes more slowly for cloud (WAN connections)
            change_factor = random.uniform(0.95, 1.05)
            self.network_latency = max(0.8, min(1.5, self.network_latency * change_factor))
            
            # Cloud congestion varies independently
            self.cloud_congestion = random.uniform(0.9, 1.2)
            
            # Update load variation based on time of day simulation
            time_of_day_factor = 1.0 + 0.2 * math.sin(current_time / 500.0)  # Simulate daily patterns
            load_change = random.uniform(-5, 5) * time_of_day_factor
            self.current_load = max(60, min(95, self.current_load + load_change))
            
            self.last_latency_update = current_time

    def process(self, task, current_time=0.0, policy_type=""):
        # Update network conditions for this task
        self.update_network_conditions(current_time)
        
        # Calculate distance-based latency with some randomness
        distance = haversine(self.location, task.location)
        geo_latency = distance * 0.05 * (1.0 + random.uniform(-0.2, 0.2))  # Add variability to geo latency
        
        # Base processing time with significant realistic variations
        # Add more base variability to ensure different runs produce different results
        base_processing = 3000 + random.uniform(-1000, 1200)  # Much wider base variation
        
        # Directly incorporate task size into processing time calculation with stronger impact
        # Cloud processes large tasks more efficiently than fog due to higher resources
        # but still needs more time for larger tasks
        size_factor = 1.0 + (task.size / 200) * 0.9  # Increased size impact in cloud (90% increase at size 200)
        base_processing *= size_factor
        
        # Add baseline randomness for each task
        base_randomness = 1.0 + random.uniform(-BASE_VARIATION_FACTOR, BASE_VARIATION_FACTOR)
        base_processing *= base_randomness * 0.6  # Reduced cloud processing time by 40%
        
        # Apply load factor with more realistic variations and stronger impact
        load_factor = 1.0 + (self.current_load / 100) * 0.8  # More impact from cloud load (up from 0.5)
        load_factor *= self.cloud_congestion  # Apply current cloud congestion
        
        # Apply more significant cloud resource contention for large tasks
        if task.size > 200:
            load_factor *= 1.2  # Large tasks experience more contention (up from 1.1)
        elif task.size > 100:
            load_factor *= 1.1  # Medium tasks also experience some contention
        
        # Calculate processing time with real-world variability
        processing_variation = random.uniform(PROCESSING_VARIATION_MIN, PROCESSING_VARIATION_MAX)
        processing_time = base_processing * load_factor * processing_variation
        
        # Calculate variable transmission time directly proportional to task size
        # Cloud requires uploading the entire task data with more significant randomness
        base_transmission = (task.size / self.bw) * 5 + geo_latency  # Reduced multiplier from 10 to 5
        transmission_randomness = 1.0 + random.uniform(-0.3, 0.4)  # More transmission randomness than fog
        transmission_time = base_transmission * self.network_latency * transmission_randomness
        
        # Calculate queue delay with more realistic backlog simulation and additional randomness
        queue_delay = 0.0
        if current_time < self.busy_until:
            # Queue delay is higher when cloud is busy
            backlog_factor = 1.0 + (self.current_load - 60) / 40.0  # Scale from 1.0 to 1.875 based on load
            queue_delay = (self.busy_until - current_time) * 0.1 * backlog_factor
            queue_delay = min(queue_delay * (1.0 + random.uniform(0, 0.5)), 800)  # Cap maximum queue delay, add randomness
        
        # Store queue delay in task
        task.queue_delay = queue_delay
        
        # Update cloud load based on task characteristics with more variation
        load_impact = (task.mips / self.mips) * 5 * (1.0 + random.uniform(-0.2, 0.3))
        if task.data_type in ['Large', 'Bulk']:
            load_impact *= 1.7  # Large data types have more impact (up from 1.5)
        self.current_load = min(95, self.current_load + load_impact)
        
        # Calculate completion time
        completion_time = max(current_time, self.busy_until) + processing_time
        self.busy_until = completion_time
        
        # Cloud recovers load more slowly than fog, with some randomness
        load_recovery = 0.5 * (1.0 + random.uniform(-0.2, 0.2))
        self.current_load = max(60, self.current_load - load_recovery)
        
        # Total time including all components
        total_time = queue_delay + transmission_time + processing_time
        
        # Store values in task for analysis
        task.internal_processing_time = processing_time
        task.is_cloud_served = True
        
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
        
        # New: Add tracking for resource commitment durations
        self.commitment_stats = {
            'small': [],   # < 100
            'medium_small': [],  # 100-149  
            'medium': [],  # 150-199
            'medium_large': [],  # 200-249
            'large': []    # >= 250
        }
        
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
        """Commit fog node resources based on task size."""
        if fog.name not in self.device_commitments:
            self.device_commitments[fog.name] = {}
            
        # Calculate batch commitment duration based on task size
        if task.size < 100:  # Small tasks
            batch_commitment_duration = max(5, int(BATCHES_BEFORE_RESET * 0.6))  # 60% of normal duration, minimum 5
        elif task.size < 150:  # Medium-small tasks
            batch_commitment_duration = max(7, int(BATCHES_BEFORE_RESET * 0.8))  # 80% of normal duration
        elif task.size < 200:  # Medium tasks
            batch_commitment_duration = BATCHES_BEFORE_RESET  # Normal duration
        elif task.size < 250:  # Medium-large tasks
            batch_commitment_duration = int(BATCHES_BEFORE_RESET * 1.3)  # 130% of normal duration
        else:  # Large tasks
            batch_commitment_duration = int(BATCHES_BEFORE_RESET * 1.6)  # 160% of normal duration
            
        # Track the resource commitment duration for reporting
        self.track_resource_commitments(task, batch_commitment_duration)
            
        # Commit resources for calculated number of batches
        for batch in range(current_batch, current_batch + batch_commitment_duration):
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
        # Only send very large 'Large' and 'Bulk' tasks directly to cloud
        if task.data_type in ['Large', 'Bulk']:
            return task.size > 250  # Increased threshold to keep more in fog
        elif task.data_type in ['Abrupt', 'LocationBased', 'Medical', 'SmallTextual', 'Multimedia']:
            return task.size > 275  # Increased threshold to process more in fog
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

    def track_resource_commitments(self, task, commitment_duration):
        """Track resource commitment durations by task size for reporting."""
        if task.size < 100:
            self.commitment_stats['small'].append(commitment_duration)
        elif task.size < 150:
            self.commitment_stats['medium_small'].append(commitment_duration) 
        elif task.size < 200:
            self.commitment_stats['medium'].append(commitment_duration)
        elif task.size < 250:
            self.commitment_stats['medium_large'].append(commitment_duration)
        else:
            self.commitment_stats['large'].append(commitment_duration)
    
    def print_commitment_stats(self):
        """Print statistics about resource commitment durations by task size."""
        print("\nResource Commitment Duration Statistics:")
        print("  Size Category | Average Duration | Min Duration | Max Duration | Tasks")
        print("  --------------|------------------|--------------|--------------|-------")
        
        for category, durations in self.commitment_stats.items():
            if durations:
                avg_duration = sum(durations) / len(durations)
                min_duration = min(durations)
                max_duration = max(durations)
                count = len(durations)
                cat_display = category.replace('_', '-').capitalize()
                print(f"  {cat_display:<14} | {avg_duration:.2f} batches    | {min_duration} batches    | {max_duration} batches    | {count}")
            else:
                cat_display = category.replace('_', '-').capitalize()
                print(f"  {cat_display:<14} | N/A              | N/A          | N/A          | 0")

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
        """Algorithm 1: Global Gateway With FCFS Tuples and Cooperation Policy with Real-world Factors"""
        selection_time = 0
        rejected_by_fog = False
        reassigned = False
        task_type = task.data_type
        allocation = "Unknown"
        
        # Track real-world factors that may cause migration
        high_network_congestion = False
        high_fog_utilization = False
        
        # Step 1: Check if task involves bulk data or should go directly to cloud
        # Immediate cloud redirection for bulk data or large tasks
        if self.is_bulk_data(task):
            self.metrics['node_selection_time'].append(selection_time)
            allocation = "Cloud (bulk data)"
            processing_time = self.process_cloud(task)
            # Track cloud allocation for this data type
            if task_type in self.data_type_counts:
                self.data_type_counts[task_type]['cloud'] += 1
            # Print task information only if verbose mode is on
            if self.verbose_output:
                print(f"Task {task.id}: {task_type}, Allocated to {allocation}, Size: {task.size}, Lifetime: {processing_time:.2f}ms")
            return processing_time
        
        # Step 2: Check for real-time network or resource constraints that force cloud migration
        # Simulate real-world unpredictability with network-wide factors
        current_network_saturation = random.uniform(0.7, 1.3)  # Global network condition
        system_wide_congestion = False
        
        # Simulate random network-wide congestion events (rare but impactful)
        if random.random() < 0.05:  # 5% chance of system-wide congestion
            system_wide_congestion = True
            current_network_saturation = random.uniform(1.1, 1.5)  # Higher during congestion events
        
        # Force cloud migration for sensitive tasks during system-wide congestion
        if system_wide_congestion and (task.data_type in ['Medical', 'Abrupt'] and task.size > 220):
            self.metrics['node_selection_time'].append(selection_time)
            allocation = "Cloud (network congestion)"
            processing_time = self.process_cloud(task)
            if task_type in self.data_type_counts:
                self.data_type_counts[task_type]['cloud'] += 1
            if self.verbose_output:
                print(f"Task {task.id}: {task_type}, Allocated to {allocation}, Size: {task.size}, System congestion: Yes, Lifetime: {processing_time:.2f}ms")
            return processing_time
        
        # Step 3: Search for a valid fog node (FCFS order), with added real-world considerations
        fog_processed = False
            
        # Remove round-robin logic and always try the first fog node first
        if len(self.fog_nodes) > 0:
            # Always try the first fog node
            primary_fog = self.fog_nodes[0]
            
            self.sim_clock += NODE_CHECK_DELAY
            selection_time += NODE_CHECK_DELAY
            
            # Check if primary node can accept the task
            if self.is_fog_available(primary_fog, task, self.current_batch):
                # Check for network congestion at this specific node
                if primary_fog.network_congestion <= NETWORK_CONGESTION_THRESHOLD or task.size <= 150:
                    # Process the task on this fog node
                    q_delay, p_time, completion_time = primary_fog.process(task, self.sim_clock)
                    self.sim_clock = completion_time
                    self.metrics['fog_times'].append(p_time)
                    self.metrics['node_selection_time'].append(selection_time)
                    fog_processed = True
                        
                    # Set task metadata
                    task.processor_node = primary_fog.name
                    allocation = f"Fog ({primary_fog.name})"
                        
                    # Track fog allocation for this data type
                    if task_type in self.data_type_counts:
                        self.data_type_counts[task_type]['fog'] += 1
                        
                    # Track metrics for this task execution
                    if q_delay > 0:
                        self.metrics['queue_delays'].append(q_delay)
                        
                    # Log execution details if in verbose mode
                    if self.verbose_output:
                        print(f"Task {task.id}: {task_type}, Allocated to {allocation}, Size: {task.size}, " 
                              f"Congestion: {primary_fog.network_congestion:.2f}, Utilization: {primary_fog.utilization:.1f}%, "
                              f"Queue delay: {q_delay:.2f}ms, Lifetime: {p_time:.2f}ms")
                        
                    return 0
                else:
                    high_network_congestion = True
            else:
                high_fog_utilization = True
                
            # Only check other fog nodes if the primary node failed due to lack of resources
            if high_fog_utilization and len(self.fog_nodes) > 1:
                # Try other fog nodes in sequence
                for i in range(1, len(self.fog_nodes)):
                    secondary_fog = self.fog_nodes[i]
                    
                    self.sim_clock += NODE_CHECK_DELAY
                    selection_time += NODE_CHECK_DELAY
                    
                    # Check if secondary node can accept the task
                    if self.is_fog_available(secondary_fog, task, self.current_batch):
                        # Check for network congestion at this specific node
                        if secondary_fog.network_congestion <= NETWORK_CONGESTION_THRESHOLD or task.size <= 150:
                            # Process the task on this fog node
                            q_delay, p_time, completion_time = secondary_fog.process(task, self.sim_clock)
                            self.sim_clock = completion_time
                            self.metrics['fog_times'].append(p_time)
                            self.metrics['node_selection_time'].append(selection_time)
                            fog_processed = True
                                
                            # Set task metadata
                            task.processor_node = secondary_fog.name
                            allocation = f"Fog ({secondary_fog.name})"
                                
                            # Track fog allocation for this data type
                            if task_type in self.data_type_counts:
                                self.data_type_counts[task_type]['fog'] += 1
                                
                            # Track metrics for this task execution
                            if q_delay > 0:
                                self.metrics['queue_delays'].append(q_delay)
                                
                            # Log execution details if in verbose mode
                            if self.verbose_output:
                                print(f"Task {task.id}: {task_type}, Allocated to {allocation}, Size: {task.size}, " 
                                      f"Congestion: {secondary_fog.network_congestion:.2f}, Utilization: {secondary_fog.utilization:.1f}%, "
                                      f"Queue delay: {q_delay:.2f}ms, Lifetime: {p_time:.2f}ms")
                                
                            return 0
                    else:
                        high_fog_utilization = True
                        
        # If we reach here, both fog nodes couldn't process the task
        # Process in cloud as a fallback
        self.metrics['node_selection_time'].append(selection_time)
        allocation = "Cloud (fog unavailable)"
        processing_time = self.process_cloud(task)
        
        # Track cloud allocation for this data type
        if task_type in self.data_type_counts:
            self.data_type_counts[task_type]['cloud'] += 1
        
        # Log the reason for cloud assignment
        cloud_reason = "unknown"
        if high_network_congestion:
            cloud_reason = "network congestion"
        elif high_fog_utilization:
            cloud_reason = "high utilization"
        else:
            cloud_reason = "resources unavailable"
        
        if self.verbose_output:
            print(f"Task {task.id}: {task_type}, Allocated to {allocation}, Size: {task.size}, "
                  f"Rejected by Fog: Yes, Reason: {cloud_reason}, Lifetime: {processing_time:.2f}ms")
            
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
        
        # Display estimated processing times by task size
        print("\nEstimated Processing Times by Task Size (based on current conditions):")
        print("  Size      | Fog Processing (min-max) | Transmission (min-max) | Cloud Processing (min-max) | Migration Probability | Resource Commitment")
        print("  ----------|--------------------------|------------------------|----------------------------|----------------------|-------------------")
        
        # Sample sizes to demonstrate
        sample_sizes = [50, 100, 150, 200, 250]
        
        for size in sample_sizes:
            # Create a sample task of this size
            sample_task = Task(
                id="sample",
                size=size,
                name="Sample",
                mips=100,
                number_of_pes=2,
                ram=128,
                bw=10,
                data_type="SmallTextual",
                location=(34.0, 73.0),
                device_type="Mobile"
            )
            
            # Calculate resource commitment duration
            if size < 100:  # Small tasks
                batch_commitment = max(5, int(BATCHES_BEFORE_RESET * 0.6))  # 60% of normal duration, minimum 5
            elif size < 150:  # Medium-small tasks
                batch_commitment = max(7, int(BATCHES_BEFORE_RESET * 0.8))  # 80% of normal duration
            elif size < 200:  # Medium tasks
                batch_commitment = BATCHES_BEFORE_RESET  # Normal duration
            elif size < 250:  # Medium-large tasks
                batch_commitment = int(BATCHES_BEFORE_RESET * 1.3)  # 130% of normal duration
            else:  # Large tasks
                batch_commitment = int(BATCHES_BEFORE_RESET * 1.6)  # 160% of normal duration
            
            # Calculate estimates based on first fog node
            if self.fog_nodes:
                fog = self.fog_nodes[0]
                fog_min, fog_max = sample_task.estimate_fog_processing_time(fog.mips)
                trans_min, trans_max = sample_task.estimate_transmission_time(fog.down_bw, fog.network_congestion)
                
                # Calculate cloud estimates
                cloud = self.cloud_services[0] if self.cloud_services else None
                cloud_min, cloud_max = ("N/A", "N/A")
                if cloud:
                    cloud_min, cloud_max = sample_task.estimate_cloud_processing_time()
                
                # Calculate migration probability based on size
                if size > 200:
                    migration_prob = "High (80-100%)"
                elif size > 150:
                    migration_prob = "Medium (40-80%)"
                elif size > 100:
                    migration_prob = "Low (10-40%)"
                else:
                    migration_prob = "Very Low (<10%)"
                
                print(f"  {size:<10}| {fog_min:.1f}-{fog_max:.1f} ms           | {trans_min:.1f}-{trans_max:.1f} ms        | {cloud_min:.1f}-{cloud_max:.1f} ms            | {migration_prob:<20} | {batch_commitment} batches")
        
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
        filepath = os.path.join(os.getcwd(), 'Tuple50K.json')
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
            },
            # Add size-based tracking to separate by fog node
            'size_metrics': {
                'small': {'fog1_count': 0, 'fog2_count': 0, 'cloud_count': 0, 'fog1_times': [], 'fog2_times': [], 'cloud_times': []},
                'medium': {'fog1_count': 0, 'fog2_count': 0, 'cloud_count': 0, 'fog1_times': [], 'fog2_times': [], 'cloud_times': []},
                'large': {'fog1_count': 0, 'fog2_count': 0, 'cloud_count': 0, 'fog1_times': [], 'fog2_times': [], 'cloud_times': []}
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
            
            # Size-based counts for this batch
            batch_size_stats = {
                'small': {'fog1': 0, 'fog2': 0, 'cloud': 0, 'fog1_times': [], 'fog2_times': [], 'cloud_times': []},
                'medium': {'fog1': 0, 'fog2': 0, 'cloud': 0, 'fog1_times': [], 'fog2_times': [], 'cloud_times': []},
                'large': {'fog1': 0, 'fog2': 0, 'cloud': 0, 'fog1_times': [], 'fog2_times': [], 'cloud_times': []}
            }
            
            for task in tqdm(batch, desc=f"Batch {batch_counter}"):
                task.batch_id = batch_counter
                task.processing_start_time = gateway.sim_clock
                
                # Offload task to appropriate resource
                result = gateway.offload_task(task)
                
                task.processing_end_time = gateway.sim_clock
                
                # Track where the task was processed
                if result == 0:
                    fog_count += 1
                    
                    # Determine task size category
                    if task.size < 100:
                        size_cat = 'small'
                    elif task.size < 200:
                        size_cat = 'medium'
                    else:
                        size_cat = 'large'
                    
                    # Get the fog node that processed this task using the processor_node attribute
                    fog_node_name = task.processor_node
                            
                    # Update size-based stats for fog
                    if fog_node_name == "Edge-Fog-01":
                        batch_size_stats[size_cat]['fog1'] += 1
                        batch_size_stats[size_cat]['fog1_times'].append(task.internal_processing_time)
                        results[policy_name]['size_metrics'][size_cat]['fog1_count'] += 1
                        results[policy_name]['size_metrics'][size_cat]['fog1_times'].append(task.internal_processing_time)
                    else:
                        batch_size_stats[size_cat]['fog2'] += 1
                        batch_size_stats[size_cat]['fog2_times'].append(task.internal_processing_time)
                        results[policy_name]['size_metrics'][size_cat]['fog2_count'] += 1
                        results[policy_name]['size_metrics'][size_cat]['fog2_times'].append(task.internal_processing_time)
                else:
                    cloud_count += 1
                    
                    # Determine task size category
                    if task.size < 100:
                        size_cat = 'small'
                    elif task.size < 200:
                        size_cat = 'medium'
                    else:
                        size_cat = 'large'
                        
                    # Update size-based stats for cloud
                    batch_size_stats[size_cat]['cloud'] += 1
                    batch_size_stats[size_cat]['cloud_times'].append(result)
                    # Update overall size metrics
                    results[policy_name]['size_metrics'][size_cat]['cloud_count'] += 1
                    results[policy_name]['size_metrics'][size_cat]['cloud_times'].append(result)
                
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
                
                # Print size-based distribution for this batch
                print("\nTask Size Distribution for this Batch:")
                print("  Size    | Fog Count | Cloud Count | Avg Fog Time | Avg Cloud Time")
                print("  --------|-----------|-------------|--------------|---------------")
                for size_cat in ['small', 'medium', 'large']:
                    fog_ct = batch_size_stats[size_cat]['fog1'] + batch_size_stats[size_cat]['fog2']
                    cloud_ct = batch_size_stats[size_cat]['cloud']
                    fog_avg = np.mean(batch_size_stats[size_cat]['fog1_times']) if batch_size_stats[size_cat]['fog1_times'] else 0
                    cloud_avg = np.mean(batch_size_stats[size_cat]['cloud_times']) if batch_size_stats[size_cat]['cloud_times'] else 0
                    print(f"  {size_cat:<8}| {fog_ct:<10}| {cloud_ct:<12}| {fog_avg:.2f} ms     | {cloud_avg:.2f} ms")
                
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
        for fog in fog_nodes:
            print(f"{fog.name}: Processed {fog.cumulative_processed} tasks, Final Utilization: {fog.cumulative_utilization:.2f}%")
            
        # Print task size-based statistics
        print("\nTask Size-Based Statistics (By Fog Node):")
        print("  Size    | Total Count | Fog1 % | Fog2 % | Cloud % | Avg Fog1 Time | Avg Fog2 Time | Avg Cloud Time")
        print("  --------|-------------|--------|--------|---------|---------------|---------------|---------------")

        # Get the total counts for each fog node
        fog1_total = fog_nodes[0].cumulative_processed
        fog2_total = fog_nodes[1].cumulative_processed
        
        for size_cat in ['small', 'medium', 'large']:
            metrics = results[policy_name]['size_metrics'][size_cat]
            total = metrics['fog1_count'] + metrics['fog2_count'] + metrics['cloud_count']
            
            if total > 0:
                # Instead of using the per-size percentages, calculate using the total fog node counts
                # This ensures the percentages match the total processed tasks per node
                fog1_total_pct = (fog1_total / (fog1_total + fog2_total + total_cloud_count)) * 100
                fog2_total_pct = (fog2_total / (fog1_total + fog2_total + total_cloud_count)) * 100
                cloud_total_pct = (total_cloud_count / (fog1_total + fog2_total + total_cloud_count)) * 100
                
                # For size-specific calculations, compute what percentage of this size category
                # was processed by each node based on their total processing proportion
                fog1_pct = fog1_total_pct * (metrics['fog1_count'] / total)
                fog2_pct = fog2_total_pct * (metrics['fog2_count'] / total)
                cloud_pct = cloud_total_pct * (metrics['cloud_count'] / total)
                
                fog1_avg = np.mean(metrics['fog1_times']) if metrics['fog1_times'] else 0
                fog2_avg = np.mean(metrics['fog2_times']) if metrics['fog2_times'] else 0
                cloud_avg = np.mean(metrics['cloud_times']) if metrics['cloud_times'] else 0
                
                print(f"  {size_cat:<8}| {total:<11}| {fog1_pct:.1f}% | {fog2_pct:.1f}% | {cloud_pct:.1f}% | {fog1_avg:.2f} ms | {fog2_avg:.2f} ms | {cloud_avg:.2f} ms")
        
        # Print detailed size impact analysis
        print("\nSize Impact Analysis:")
        print("  - Small tasks (<100): Processed faster with minimal transmission overhead")
        print("  - Medium tasks (100-200): Network conditions have moderate impact on processing")
        print("  - Large tasks (>200): Significantly affected by network congestion and node utilization")
            
        # Print overall performance metrics
        total_time = __import__('time').time() - start_time
        print(f"\nTotal processing time: {total_time:.2f}s ({total_processed/total_time:.2f} tasks/second)")
        
        # Print resource commitment statistics
        gateway.print_commitment_stats()
        
        # Print data type distribution
        print("\nFinal Data Type Distribution:")
        for data_type, counts in gateway.data_type_counts.items():
            fog_count = counts.get('fog', 0)
            cloud_count = counts.get('cloud', 0)
            total = fog_count + cloud_count
            if total > 0:
                fog_pct = (fog_count / total) * 100
                print(f"  {data_type:<15}: Fog: {fog_count} ({fog_pct:.1f}%), Cloud: {cloud_count} ({100-fog_pct:.1f}%)")
        
        # Standardized performance metrics output
        print("\n=== Performance Metrics ===")
        print("=== Average Processing Times (ms) ===")
        print(f"FCFSCooperation: Total = {avg_fog_time + avg_cloud_time:.2f}, Fog = {avg_fog_time:.2f}, Cloud = {avg_cloud_time:.2f}")
        print("\n=== Average Power Consumption per Node (W) ===")
        power_values = []
        for node in fog_nodes:
            power_values.append(f'{node.power_log[-1]:.2f}')
        print(f"FCFSCooperation: {power_values}")
        print("\n=== Average Queue Delays (ms) ===")
        avg_delay = np.mean(gateway.metrics['queue_delays']) if gateway.metrics['queue_delays'] else 0
        print(f"FCFSCooperation: {avg_delay:.2f}")
        print("\n=== Task Distribution ===")
        print(f"FCFSCooperation: Fog = {total_fog_count} ({total_fog_count/total_processed*100:.1f}%), Cloud = {total_cloud_count} ({total_cloud_count/total_processed*100:.1f}%)")
        print("\n=== Data Type Distribution ===")
        print(f"{'Data Type':<15} | {'Fog Count':<10} | {'Cloud Count':<10} | {'Total':<10} | {'Fog %':<10}")
        print("-" * 70)
        for data_type, counts in sorted(gateway.data_type_counts.items()):
            fog_count = counts.get('fog', 0)
            cloud_count = counts.get('cloud', 0)
            type_total = fog_count + cloud_count
            type_fog_percent = (fog_count / type_total) * 100 if type_total > 0 else 0
            print(f"{data_type:<15} | {fog_count:<10} | {cloud_count:<10} | {type_total:<10} | {type_fog_percent:<10.1f}%")
    
    except FileNotFoundError as e:
        print(f"File not found error: {e}")
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
    except KeyError as e:
        print(f"Key error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
