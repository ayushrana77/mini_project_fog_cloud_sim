import random
import time
from dataclasses import dataclass
from datetime import datetime
import json
from enum import Enum
from typing import Optional

class Priority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class NetworkState(Enum):
    GOOD = 1
    AVERAGE = 2
    POOR = 3

class Season(Enum):
    SPRING = (3, 5)
    SUMMER = (6, 8)
    AUTUMN = (9, 11)
    WINTER = (12, 2)

@dataclass
class IoTTask:
    task_id: int
    timestamp: str
    data_type: str
    size: float
    location: tuple
    cpu_requirements: int
    ram_requirements: int
    bandwidth: int
    network_quality: dict
    priority: Priority
    sensor_data: Optional[dict] = None  # Allow None for non-sensor tasks
    encryption_required: bool = False
    device_id: str = ""
    battery_level: float = 0.0
    correlation_id: Optional[int] = None

class NetworkSimulator:
    def __init__(self):
        self.state = NetworkState.GOOD
        self.transition_matrix = {
            NetworkState.GOOD: {NetworkState.GOOD: 0.7, NetworkState.AVERAGE: 0.25, NetworkState.POOR: 0.05},
            NetworkState.AVERAGE: {NetworkState.AVERAGE: 0.6, NetworkState.GOOD: 0.3, NetworkState.POOR: 0.1},
            NetworkState.POOR: {NetworkState.POOR: 0.5, NetworkState.AVERAGE: 0.4, NetworkState.GOOD: 0.1}
        }
        
    def get_network_quality(self):
        self.state = random.choices(
            list(self.transition_matrix[self.state].keys()),
            weights=list(self.transition_matrix[self.state].values())
        )[0]
        
        quality_params = {
            NetworkState.GOOD: {'latency': (10, 50), 'packet_loss': (0.1, 0.5)},
            NetworkState.AVERAGE: {'latency': (50, 150), 'packet_loss': (0.5, 2)},
            NetworkState.POOR: {'latency': (150, 500), 'packet_loss': (2, 5)}
        }
        return {
            'state': self.state.name,
            'latency_ms': random.randint(*quality_params[self.state]['latency']),
            'packet_loss_%': round(random.uniform(*quality_params[self.state]['packet_loss']), 1)
        }

class TaskGenerator:
    def __init__(self, num_devices=10, location_range=(33.6844, 73.0479)):
        self.task_id = 0
        self.devices = [f"DEV_{i:03d}" for i in range(num_devices)]
        self.device_batteries = {dev: 100.0 for dev in self.devices}
        self.location_range = location_range
        self.network_sim = NetworkSimulator()
        self.last_task_type = None
        self.correlation_counter = 0
        
        current_month = datetime.now().month
        self.season = next((s for s in Season 
                          if (s.value[0] <= current_month <= s.value[1]) or 
                          (s.value[0] > s.value[1] and (current_month >= s.value[0] or current_month <= s.value[1]))), Season.SPRING)

    def _get_data_type_weights(self):
        base_weights = {
            'sensor': 0.35, 
            'small_text': 0.25, 
            'medical': 0.15,
            'location': 0.12, 
            'multimedia': 0.08, 
            'abrupt': 0.03,
            'large': 0.015, 
            'bulk': 0.005
        }
        
        seasonal_adj = self._get_seasonal_adjustment()
        for k, v in seasonal_adj.items():
            base_weights[k] = max(0.01, base_weights[k] + v)
        
        time_adj = self._get_time_adjustment()
        for k, v in time_adj.items():
            base_weights[k] = max(0.01, base_weights[k] + v)
        
        total = sum(base_weights.values())
        return {k: v/total for k, v in base_weights.items()}
    
    def _generate_location(self):
        base_lat, base_lon = self.location_range
        return (
            round(base_lat + random.uniform(-0.05, 0.05), 6),
            round(base_lon + random.uniform(-0.05, 0.05), 6)
        )

    def _get_size_distribution(self, data_type):
        size_ranges = {
            'sensor': (0.001, 0.1),
            'small_text': (0.001, 1),
            'medical': (1, 10),
            'location': (0.1, 2),
            'multimedia': (5, 50),
            'abrupt': (0.5, 5),
            'large': (10, 100),
            'bulk': (100, 1000)
        }
        min_size, max_size = size_ranges[data_type]
        return round(random.uniform(min_size, max_size), 2)

    def _get_cpu_ram(self, data_type):
        requirements = {
            'sensor': (100, 128),
            'small_text': (200, 256),
            'medical': (1000, 1024),
            'location': (300, 512),
            'multimedia': (2000, 2048),
            'abrupt': (800, 1024),
            'large': (3000, 4096),
            'bulk': (5000, 8192)
        }
        return requirements[data_type]

    def _get_bandwidth(self, data_type):
        bandwidth_ranges = {
            'sensor': (100, 300),
            'small_text': (200, 500),
            'medical': (300, 800),
            'location': (200, 400),
            'multimedia': (500, 1000),
            'abrupt': (300, 600),
            'large': (800, 1200),
            'bulk': (1000, 2000)
        }
        return random.randint(*bandwidth_ranges[data_type])

    def _get_seasonal_adjustment(self):
        adjustments = {
            Season.SUMMER: {'sensor': 0.1, 'multimedia': 0.05},
            Season.WINTER: {'sensor': -0.05, 'medical': 0.07},
            Season.SPRING: {'location': 0.08},
            Season.AUTUMN: {'abrupt': 0.03}
        }
        return adjustments.get(self.season, {})

    def _get_time_adjustment(self):
        current_hour = datetime.now().hour
        return {'multimedia': 0.07, 'location': 0.05} if 6 <= current_hour < 18 else {'sensor': 0.04, 'medical': -0.03}

    def _update_battery(self, device_id, task_size):
        base_drain = task_size * 0.005
        self.device_batteries[device_id] = max(0, min(100, 
            self.device_batteries[device_id] - base_drain + random.uniform(-0.1, 0.1)))
        if random.random() < 0.05:
            self.device_batteries[device_id] = min(100, self.device_batteries[device_id] + random.uniform(10, 30))

    def _generate_sensor_data(self, data_type):
        if data_type != 'sensor':
            return None
        sensor_data = {
            'temperature': round(random.uniform(-20, 50), 1),
            'humidity': round(random.uniform(0, 100)),
            'pressure': round(random.uniform(900, 1100), 1),
            'motion': random.choice([True, False]),
            'status': random.choices(['normal', 'warning', 'critical'], [0.85, 0.1, 0.05])[0]
        }
        if self.season == Season.SUMMER:
            sensor_data['temperature'] = round(random.uniform(25, 45), 1)
        elif self.season == Season.WINTER:
            sensor_data['temperature'] = round(random.uniform(-20, 5), 1)
        return sensor_data

    def _get_priority(self, data_type):
        priorities = {
            'sensor': Priority.LOW,
            'small_text': Priority.MEDIUM,
            'medical': Priority.HIGH,
            'location': Priority.MEDIUM,
            'multimedia': Priority.MEDIUM,
            'abrupt': Priority.CRITICAL,
            'large': Priority.HIGH,
            'bulk': Priority.HIGH
        }
        return priorities[data_type]

    def _get_correlation_id(self, current_type):
        if self.last_task_type == 'location' and current_type == 'multimedia':
            self.correlation_counter += 1
            return self.correlation_counter
        if current_type == 'multimedia' and random.random() < 0.3:
            return self.correlation_counter
        return None

    def generate_task(self):
        self.task_id += 1
        weights = self._get_data_type_weights()
        data_type = random.choices(list(weights.keys()), weights=list(weights.values()))[0]
        
        correlation_id = self._get_correlation_id(data_type)
        self.last_task_type = data_type if correlation_id else None

        device_id = random.choice(self.devices)
        size = self._get_size_distribution(data_type)
        self._update_battery(device_id, size)

        return IoTTask(
            task_id=self.task_id,
            timestamp=datetime.now().isoformat(),
            data_type=data_type,
            size=size,
            location=self._generate_location(),
            cpu_requirements=self._get_cpu_ram(data_type)[0],
            ram_requirements=self._get_cpu_ram(data_type)[1],
            bandwidth=self._get_bandwidth(data_type),
            network_quality=self.network_sim.get_network_quality(),
            priority=self._get_priority(data_type),
            sensor_data=self._generate_sensor_data(data_type),
            encryption_required=random.choices([True, False], [0.3 if data_type in ['medical', 'location'] else 0.05, 0.7])[0],
            device_id=device_id,
            battery_level=round(self.device_batteries[device_id], 2),
            correlation_id=correlation_id
        )

def json_default(obj):
    if isinstance(obj, Enum):
        return obj.name
    raise TypeError(f"Type {type(obj)} not serializable")

def main():
    generator = TaskGenerator(num_devices=5)
    try:
        while True:
            if random.random() < 0.7:
                task = generator.generate_task()
                print(f"Generated task:\n{json.dumps(task.__dict__, indent=2, default=json_default)}")
                time.sleep(random.expovariate(1/2))
            else:
                time.sleep(random.uniform(5, 10))
                task = generator.generate_task()
                print(f"Generated LARGE task:\n{json.dumps(task.__dict__, indent=2, default=json_default)}")
    except KeyboardInterrupt:
        print("\nTask generation stopped.")

if __name__ == "__main__":
    main()