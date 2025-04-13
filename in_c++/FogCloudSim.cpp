#include "FogCloudSim.h"
// Removing dependency on external JSON library and implementing a simplified approach
// #include <nlohmann/json.hpp> // JSON for Modern C++
#include <ctime>
#include <cmath>
#include <random>
#include <fstream>
#include <iostream>
#include <chrono>
#include <algorithm>

// using json = nlohmann::json;
using namespace std;

// Define the FOG_NODES_CONFIG and CLOUD_SERVICES_CONFIG
vector<map<string, any>> FOG_NODES_CONFIG = {
    {
        {"name", string("Edge-Fog-01")},
        {"location", make_pair(33.72, 72.85)},
        {"down_bw", 50000.0},
        {"up_bw", 30000.0},
        {"mips", 200000.0},
        {"num_pes", 2400},
        {"ram", 327680},
        {"storage", 800000},
        {"num_devices", 3500}
    },
    {
        {"name", string("Edge-Fog-02")},
        {"location", make_pair(34.12, 73.25)},
        {"down_bw", 60000.0},
        {"up_bw", 40000.0},
        {"mips", 250000.0},
        {"num_pes", 3200},
        {"ram", 491520},
        {"storage", 1000000},
        {"num_devices", 3500}
    }
};

vector<map<string, any>> CLOUD_SERVICES_CONFIG = {
    {
        {"name", string("USA-Service1")},
        {"location", make_pair(37.09, -95.71)},
        {"ram", 16384},
        {"mips", 15000.0},
        {"bw", 800.0}
    },
    {
        {"name", string("Singapore-Service1")},
        {"location", make_pair(1.35, 103.82)},
        {"ram", 16384},
        {"mips", 15000.0},
        {"bw", 800.0}
    }
};

// Task implementation
Task::Task(const string& id, int size, const string& name, double mips, int number_of_pes, 
           int ram, int bw, const string& data_type, const Location& location, const string& device_type)
    : id(id), size(size), name(name), mips(mips), number_of_pes(number_of_pes), ram(ram), 
      bw(bw), data_type(data_type), location(location), device_type(device_type) {
    
    arrival_time = 0.0; // Default arrival time
    internal_processing_time = 0.0; // Will be calculated during processing
    queue_wait = 0.0; // Will be calculated during processing
    processing_start_time = 0.0;
    processing_end_time = 0.0;
    fog_candidate = true; // Default to fog candidate
}

// FogNode implementation
FogNode::FogNode(const map<string, any>& config) 
    : name("DefaultFogNode"), location({0.0, 0.0}), down_bw(100.0), up_bw(50.0),
    mips(10000.0), num_pes(4), ram(8192), total_storage(100000), used_storage(0),
    utilization(0.0), busy_until(0.0), num_devices(0), available_ram(8192),
    available_mips(10000.0), max_queue_size(100), total_processed(0), sim_clock(0.0),
    cumulative_processed(0), cumulative_utilization(0.0), network_congestion(1.0),
    background_load(0.0), last_congestion_update(0.0), congestion_update_interval(10.0) {
    
    // Parse from configuration
    try {
        if (config.count("name") > 0) {
            name = any_cast<string>(config.at("name"));
        }
        
        if (config.count("location") > 0) {
            auto loc = any_cast<vector<double>>(config.at("location"));
            if (loc.size() >= 2) {
                location = {loc[0], loc[1]};
            }
        }
        
        if (config.count("down_bw") > 0) {
            down_bw = any_cast<double>(config.at("down_bw"));
        }
        
        if (config.count("up_bw") > 0) {
            up_bw = any_cast<double>(config.at("up_bw"));
        }
        
        if (config.count("mips") > 0) {
            mips = any_cast<double>(config.at("mips"));
            available_mips = mips;
        }
        
        if (config.count("num_pes") > 0) {
            num_pes = any_cast<int>(config.at("num_pes"));
        }
        
        if (config.count("ram") > 0) {
            ram = any_cast<int>(config.at("ram"));
            available_ram = ram;
        }
        
        if (config.count("storage") > 0) {
            total_storage = any_cast<int>(config.at("storage"));
        }
        
        if (config.count("max_queue_size") > 0) {
            max_queue_size = any_cast<int>(config.at("max_queue_size"));
        }
    } catch (const bad_any_cast& e) {
        cerr << "Error parsing FogNode configuration: " << e.what() << endl;
    }
}

double FogNode::calculate_power() {
    // Simple power calculation based on utilization
    double base_power = 100.0; // Base power consumption in Watts
    double max_power = 300.0;  // Maximum power consumption at full utilization
    
    double power = base_power + (max_power - base_power) * (utilization / 100.0);
    power_log.push_back(power);
    
    return power;
}

void FogNode::update_network_congestion(double current_time) {
    // Skip if not time to update yet
    if (current_time - last_congestion_update < congestion_update_interval) {
        return;
    }
    
    // Random congestion update
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> congestion_dist(0.5, 1.5);
    
    network_congestion = congestion_dist(gen);
    last_congestion_update = current_time;
    
    // Update background load
    uniform_real_distribution<> load_dist(0.0, 0.1);
    background_load = load_dist(gen);
}

bool FogNode::can_accept_task(const Task& task, double current_time) {
    // Update resources first
    update_resources(current_time);
    
    // Check if queue is full
    if (queue.size() >= max_queue_size) {
        return false;
    }
    
    // Check if we have enough resources
    if (available_ram < task.ram || available_mips < task.mips) {
        return false;
    }
    
    // Check if utilization is too high
    if (utilization > 90.0) {
        return false;
    }
    
    return true;
}

double FogNode::process(Task& task, double arrival_time) {
    // Update resources first
    update_resources(arrival_time);
    
    // Calculate processing time based on task requirements and node capabilities
    double base_processing_time = (task.mips * task.size) / available_mips;
    
    // Apply variation factors
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> proc_var_dist(0.9, 1.1);
    double proc_variation = proc_var_dist(gen);
    
    // Account for background load
    double adjusted_time = base_processing_time * proc_variation * (1.0 + background_load);
    
    // Update task's internal processing time
    task.internal_processing_time = adjusted_time;
    task.processing_start_time = arrival_time;
    task.processing_end_time = arrival_time + adjusted_time;
    
    // Add queue delay if node is busy
    double queue_delay = 0.0;
    if (busy_until > arrival_time) {
        queue_delay = busy_until - arrival_time;
        task.queue_wait = queue_delay;
    }
    
    // Update node's busy status
    busy_until = arrival_time + queue_delay + adjusted_time;
    
    // Allocate resources to the task
    available_ram -= task.ram;
    available_mips -= task.mips;
    
    // Update utilization
    utilization = (1.0 - (available_mips / mips)) * 100.0;
    
    // Add to node's queue
    queue.push_back(task);
    
    // Update statistics
    total_processed++;
    cumulative_processed++;
    cumulative_utilization += utilization;
    
    // Schedule resource release
    resource_release_schedule.push_back({busy_until, adjusted_time});
    
    // Update task processing info
    task.is_served = true;
    task.processor_node = name;
    
    return adjusted_time;
}

void FogNode::update_resources(double current_time) {
    // Release resources from completed tasks
    auto it = resource_release_schedule.begin();
    while (it != resource_release_schedule.end()) {
        if (it->first <= current_time) {
            // Resources are released, reclaim them
            available_mips += mips * (it->second / busy_until);
            available_ram += ram * 0.1; // Approximate RAM release
            
            // Ensure we don't exceed total resources
            available_mips = min(available_mips, mips);
            available_ram = min(available_ram, ram);
            
            it = resource_release_schedule.erase(it);
        } else {
            ++it;
        }
    }
    
    // Update network conditions
    update_network_congestion(current_time);
    
    // Update node's busy status
    if (busy_until <= current_time) {
        busy_until = 0.0;
    }
    
    // Update utilization
    utilization = (1.0 - (available_mips / mips)) * 100.0;
    
    // Update sim clock
    sim_clock = current_time;
}

void FogNode::reset() {
    queue.clear();
    resource_release_schedule.clear();
    available_mips = mips;
    available_ram = ram;
    utilization = 0.0;
    busy_until = 0.0;
    used_storage = 0;
    network_congestion = 1.0;
    background_load = 0.0;
    last_congestion_update = 0.0;
}

// CloudService implementation
CloudService::CloudService(const map<string, any>& config)
    : name("DefaultCloudService"), location({0.0, 0.0}), ram(32768), mips(50000.0),
    bw(1000.0), busy_until(0.0), current_load(0.0), max_queue_size(1000),
    network_latency(100.0), last_latency_update(0.0), latency_update_interval(20.0),
    cloud_congestion(1.0) {
    
    // Parse from configuration
    try {
        if (config.count("name") > 0) {
            name = any_cast<string>(config.at("name"));
        }
        
        if (config.count("location") > 0) {
            auto loc = any_cast<vector<double>>(config.at("location"));
            if (loc.size() >= 2) {
                location = {loc[0], loc[1]};
            }
        }
        
        if (config.count("ram") > 0) {
            ram = any_cast<int>(config.at("ram"));
        }
        
        if (config.count("mips") > 0) {
            mips = any_cast<double>(config.at("mips"));
        }
        
        if (config.count("bw") > 0) {
            bw = any_cast<double>(config.at("bw"));
        }
        
        if (config.count("max_queue_size") > 0) {
            max_queue_size = any_cast<int>(config.at("max_queue_size"));
        }
        
        if (config.count("network_latency") > 0) {
            network_latency = any_cast<double>(config.at("network_latency"));
        }
    } catch (const bad_any_cast& e) {
        cerr << "Error parsing CloudService configuration: " << e.what() << endl;
    }
}

void CloudService::reset() {
    queue.clear();
    busy_until = 0.0;
    current_load = 0.0;
    network_latency = 100.0;
    last_latency_update = 0.0;
    cloud_congestion = 1.0;
}

void CloudService::update_network_conditions(double current_time) {
    // Skip if not time to update yet
    if (current_time - last_latency_update < latency_update_interval) {
        return;
    }
    
    // Random latency update
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> latency_dist(50.0, 200.0);
    
    network_latency = latency_dist(gen);
    last_latency_update = current_time;
    
    // Update cloud congestion
    uniform_real_distribution<> congestion_dist(0.8, 1.2);
    cloud_congestion = congestion_dist(gen);
}

double CloudService::process(Task& task, double current_time, const string& policy_type) {
    // Update network conditions first
    update_network_conditions(current_time);
    
    // Calculate network transmission time
    double distance = haversine(location, task.location);
    double network_time = (distance / 1000.0) * network_latency + 10.0;
    
    // Calculate processing time based on task requirements and cloud capabilities
    double base_processing_time = (task.mips * task.size) / (mips * cloud_congestion);
    
    // Apply variation factors
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> proc_var_dist(0.9, 1.1); // Cloud has more stable performance
    double proc_variation = proc_var_dist(gen);
    
    // Total time including network transmission
    double adjusted_time = base_processing_time * proc_variation + network_time;
    
    // Update task's internal processing time
    task.internal_processing_time = adjusted_time;
    task.processing_start_time = current_time;
    task.processing_end_time = current_time + adjusted_time;
    
    // Add queue delay if cloud is busy
    double queue_delay = 0.0;
    if (busy_until > current_time) {
        queue_delay = busy_until - current_time;
        task.queue_wait = queue_delay;
    }
    
    // Update cloud's busy status
    busy_until = current_time + queue_delay + adjusted_time;
    
    // Add to cloud's queue
    queue.push_back(task);
    
    // Update current load (simplified model)
    current_load = (queue.size() / static_cast<double>(max_queue_size)) * 100.0;
    
    // Update task processing info
    task.is_served = true;
    task.is_cloud_served = true;
    task.processor_node = name;
    
    return adjusted_time;
}

// Implementation of haversine function
double haversine(const Location& loc1, const Location& loc2) {
    double lat1 = loc1.first * M_PI / 180.0;
    double lat2 = loc2.first * M_PI / 180.0;
    double dLat = (loc2.first - loc1.first) * M_PI / 180.0;
    double dLon = (loc2.second - loc1.second) * M_PI / 180.0;
    
    double a = sin(dLat/2) * sin(dLat/2) +
               cos(lat1) * cos(lat2) * 
               sin(dLon/2) * sin(dLon/2);
    double c = 2 * atan2(sqrt(a), sqrt(1-a));
    return 6371 * c; // Radius of the Earth in kilometers
}

// Task methods implementation
bool Task::should_go_to_fog() const {
    return fog_candidate;
}

bool Task::is_small_task() const {
    return size < 10; // Assuming size is in MB, small tasks are < 10MB
}

std::pair<double, double> Task::estimate_fog_processing_time(double fog_mips, double available_efficiency) const {
    // Estimate processing time (ms) based on size and MIPS requirement
    double base_time = (mips * size) / (fog_mips * available_efficiency);
    double variation = base_time * 0.1; // 10% variation
    return std::make_pair(base_time, variation);
}

std::pair<double, double> Task::estimate_transmission_time(double bandwidth, double network_factor) const {
    // Estimate transmission time (ms) based on size and bandwidth
    double base_time = (size * 8) / (bandwidth * network_factor); // Size in MB, bandwidth in Mbps
    double variation = base_time * 0.1; // 10% variation
    return std::make_pair(base_time, variation);
}

std::pair<double, double> Task::estimate_cloud_processing_time(double cloud_mips) const {
    // Cloud processing is generally faster but has more overhead
    double base_time = (mips * size) / cloud_mips;
    double variation = base_time * 0.05; // 5% variation in cloud
    return std::make_pair(base_time, variation);
}

std::string Task::get_size_category() const {
    if (size < 5) return "Small";
    else if (size < 20) return "Medium";
    else if (size < 100) return "Large";
    else return "Very Large";
}

// BaseGateway implementation
BaseGateway::BaseGateway(const std::vector<FogNode>& fog_nodes, const std::vector<CloudService>& cloud_services)
    : fog_nodes(fog_nodes), cloud_services(cloud_services), batch_size(10), current_batch(0), sim_clock(0.0) {
    
    // Initialize random number generator
    std::random_device rd;
    rng.seed(rd());
}

void BaseGateway::reset_nodes() {
    for (auto& fog : fog_nodes) {
        fog.reset();
    }
    
    for (auto& cloud : cloud_services) {
        cloud.reset();
    }
}

int BaseGateway::get_node_device_count(const std::string& node_name, int batch_id) {
    if (batch_assignments.count(batch_id) == 0 || 
        batch_assignments[batch_id].count(node_name) == 0) {
        return 0;
    }
    return batch_assignments[batch_id][node_name].size();
}

int BaseGateway::get_total_node_commitments(const std::string& node_name) {
    if (device_commitments.count(node_name) == 0) {
        return 0;
    }
    
    int total = 0;
    for (const auto& [batch_id, count] : device_commitments[node_name]) {
        total += count;
    }
    return total;
}

int BaseGateway::get_total_commitments() {
    int total = 0;
    for (const auto& [node_name, commitments] : device_commitments) {
        for (const auto& [batch_id, count] : commitments) {
            total += count;
        }
    }
    return total;
}

bool BaseGateway::is_fog_available(FogNode& fog, const Task& task, int current_batch) {
    // Check if this fog node can handle the task
    double current_time = sim_clock;
    return fog.can_accept_task(task, current_time);
}

void BaseGateway::commit_fog_resources(FogNode& fog, const Task& task, int current_batch) {
    // Commit device to the fog node for this batch
    batch_assignments[current_batch][fog.name].push_back(task);
    
    // Increment commitment count
    if (device_commitments.count(fog.name) == 0 || 
        device_commitments[fog.name].count(current_batch) == 0) {
        device_commitments[fog.name][current_batch] = 1;
    } else {
        device_commitments[fog.name][current_batch]++;
    }
}

bool BaseGateway::is_bulk_data(const Task& task) {
    return task.size > 50; // Assuming anything over 50MB is considered bulk data
}

std::vector<Task> BaseGateway::get_next_batch(const std::vector<Task>& all_tasks) {
    std::vector<Task> batch;
    
    // Start index for this batch
    int start_idx = current_batch * batch_size;
    
    // Ensure we don't exceed vector bounds
    if (start_idx >= all_tasks.size()) {
        return batch; // Return empty batch if we've processed all tasks
    }
    
    // End index for this batch (exclusive)
    int end_idx = std::min(start_idx + batch_size, static_cast<int>(all_tasks.size()));
    
    // Get the tasks for this batch
    for (int i = start_idx; i < end_idx; i++) {
        Task task = all_tasks[i];
        task.batch_id = current_batch;
        batch.push_back(task);
    }
    
    // Increment batch counter for next time
    current_batch++;
    
    return batch;
}

void BaseGateway::process_batch(std::vector<Task>& tasks) {
    // Process each task in the batch
    for (auto& task : tasks) {
        // Skip already processed tasks
        if (processed_tasks.count(task.id) > 0) {
            continue;
        }
        
        // Record start time
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Try to offload task to fog or cloud
        int result = offload_task(task);
        
        // Record end time for selection
        auto end_time = std::chrono::high_resolution_clock::now();
        auto selection_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        
        if (task.is_cloud_served) {
            metrics.cloud_times.push_back(task.internal_processing_time);
            metrics.cloud_selection_time.push_back(selection_time);
            metrics.cloud_direct_count++;
        } else {
            metrics.fog_times.push_back(task.internal_processing_time);
            metrics.node_selection_time.push_back(selection_time);
            metrics.fog_first_count++;
        }
        
        metrics.queue_delays.push_back(task.queue_delay);
        
        // Mark task as processed
        processed_tasks.insert(task.id);
    }
}

void BaseGateway::process_cloud(Task& task) {
    // Randomly select a cloud service if there are multiple
    if (cloud_services.empty()) {
        std::cerr << "Error: No cloud services available for processing." << std::endl;
        return;
    }
    
    // Select first available cloud service
    // For simplicity, always using the first cloud service, but could implement load balancing
    CloudService& cloud = cloud_services[0];
    
    // Process the task
    double current_time = sim_clock;
    double processing_time = cloud.process(task, current_time, "");
    
    // Update task status
    task.is_cloud_served = true;
    task.is_served = true;
    task.processor_node = cloud.name;
}

void BaseGateway::get_batch_metrics() {
    int fog_count = metrics.fog_first_count;
    int cloud_count = metrics.cloud_direct_count;
    int total = fog_count + cloud_count;
    
    if (total > 0) {
        metrics.batch_metrics.fog_batch_counts.push_back(fog_count);
        metrics.batch_metrics.cloud_batch_counts.push_back(cloud_count);
        
        double avg_fog_util = 0.0;
        for (const auto& fog : fog_nodes) {
            avg_fog_util += fog.utilization;
        }
        avg_fog_util /= fog_nodes.size();
        
        double avg_cloud_util = 0.0;
        for (const auto& cloud : cloud_services) {
            avg_cloud_util += cloud.current_load;
        }
        avg_cloud_util /= cloud_services.size();
        
        metrics.batch_metrics.fog_utilization.push_back(avg_fog_util);
        metrics.batch_metrics.cloud_utilization.push_back(avg_cloud_util);
    }
}

void BaseGateway::track_resource_commitments(const Task& task, int commitment_duration) {
    // Placeholder implementation - could be expanded based on requirements
}

void BaseGateway::print_commitment_stats() {
    // Print summary of resource commitments
    std::cout << "\n=== Resource Commitment Statistics ===" << std::endl;
    
    for (const auto& [node_name, commitments] : device_commitments) {
        int total_commitments = 0;
        for (const auto& [batch_id, count] : commitments) {
            total_commitments += count;
        }
        
        std::cout << "Node " << node_name << ": " << total_commitments << " total commitments" << std::endl;
    }
}

// BaseGateway::offload_task implementation (virtual base method)
int BaseGateway::offload_task(Task& task) {
    // Base implementation - always route to the cloud
    // Derived classes override this to implement specific algorithms
    process_cloud(task);
    return 0;
}

// Implementation of load_tasks
std::vector<Task> load_tasks(const std::string& filepath) {
    std::vector<Task> tasks;
    
    try {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filepath << std::endl;
            return tasks;
        }
        
        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string json_str = buffer.str();
        
        // Simple parsing of JSON array using string operations
        // This is a very basic parser and doesn't handle all JSON syntax
        size_t pos = json_str.find('[');
        size_t end_pos = json_str.rfind(']');
        
        if (pos == std::string::npos || end_pos == std::string::npos) {
            std::cerr << "Error: Invalid JSON format in " << filepath << std::endl;
            return tasks;
        }
        
        std::string tasks_array = json_str.substr(pos + 1, end_pos - pos - 1);
        
        // Split into individual task objects
        size_t task_start = 0;
        int brace_count = 0;
        bool in_string = false;
        
        for (size_t i = 0; i < tasks_array.size(); i++) {
            char c = tasks_array[i];
            
            // Handle string literals, accounting for escaped quotes
            if (c == '"' && (i == 0 || tasks_array[i-1] != '\\')) {
                in_string = !in_string;
                continue;
            }
            
            // Only count braces when not in a string
            if (!in_string) {
                if (c == '{') {
                    if (brace_count == 0) {
                        task_start = i;
                    }
                    brace_count++;
                } else if (c == '}') {
                    brace_count--;
                    
                    if (brace_count == 0) {
                        // We have a complete task object
                        std::string task_json = tasks_array.substr(task_start, i - task_start + 1);
                        
                        // Parse task attributes
                        std::string id = extract_json_string(task_json, "id");
                        std::string data_type = extract_json_string(task_json, "data_type");
                        double data_size = extract_json_double(task_json, "data_size");
                        double arrival_time = extract_json_double(task_json, "arrival_time");
                        double required_processing_time = extract_json_double(task_json, "required_processing_time");
                        bool should_go_to_fog = extract_json_bool(task_json, "should_go_to_fog");
                        
                        // Extract location
                        std::string loc_str = extract_json_array(task_json, "location");
                        std::vector<double> loc_values = parse_json_array_doubles(loc_str);
                        Location location = {0.0, 0.0};
                        if (loc_values.size() >= 2) {
                            location = {loc_values[0], loc_values[1]};
                        }
                        
                        // Create task
                        Task task(
                            id,
                            static_cast<int>(data_size),
                            "Task_" + id,
                            required_processing_time / data_size, // Estimated MIPS
                            1, // number_of_pes
                            static_cast<int>(data_size) * 2,  // Estimated RAM
                            static_cast<int>(data_size) * 8,  // Estimated bandwidth
                            data_type,
                            location,
                            "Generic"
                        );
                        
                        task.arrival_time = arrival_time;
                        task.fog_candidate = should_go_to_fog;
                        
                        tasks.push_back(task);
                    }
                }
            }
        }
        
        file.close();
        
    } catch (const std::exception& e) {
        std::cerr << "Error parsing JSON: " << e.what() << std::endl;
    }
    
    return tasks;
}

// Helper functions for JSON parsing

std::string extract_json_string(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) {
        return "";
    }
    
    pos = json.find(':', pos);
    if (pos == std::string::npos) {
        return "";
    }
    
    // Skip whitespace
    while (pos < json.size() && (json[pos] == ':' || std::isspace(json[pos]))) {
        pos++;
    }
    
    if (pos >= json.size() || json[pos] != '"') {
        // Not a string value
        return "";
    }
    
    pos++; // Skip opening quote
    
    std::string result;
    bool escaped = false;
    
    while (pos < json.size()) {
        char c = json[pos];
        
        if (escaped) {
            result += c;
            escaped = false;
        } else if (c == '\\') {
            escaped = true;
        } else if (c == '"') {
            break;
        } else {
            result += c;
        }
        
        pos++;
    }
    
    return result;
}

double extract_json_double(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) {
        return 0.0;
    }
    
    pos = json.find(':', pos);
    if (pos == std::string::npos) {
        return 0.0;
    }
    
    // Skip whitespace
    while (pos < json.size() && (json[pos] == ':' || std::isspace(json[pos]))) {
        pos++;
    }
    
    // Find end of number
    size_t end_pos = pos;
    while (end_pos < json.size() && 
           (std::isdigit(json[end_pos]) || json[end_pos] == '.' || 
            json[end_pos] == '-' || json[end_pos] == '+' || 
            json[end_pos] == 'e' || json[end_pos] == 'E')) {
        end_pos++;
    }
    
    std::string value = json.substr(pos, end_pos - pos);
    try {
        return std::stod(value);
    } catch (...) {
        return 0.0;
    }
}

bool extract_json_bool(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) {
        return false;
    }
    
    pos = json.find(':', pos);
    if (pos == std::string::npos) {
        return false;
    }
    
    // Skip whitespace
    while (pos < json.size() && (json[pos] == ':' || std::isspace(json[pos]))) {
        pos++;
    }
    
    // Check for "true" or "false"
    if (pos + 4 <= json.size() && json.substr(pos, 4) == "true") {
        return true;
    }
    
    return false;
}

std::string extract_json_array(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) {
        return "[]";
    }
    
    pos = json.find(':', pos);
    if (pos == std::string::npos) {
        return "[]";
    }
    
    // Skip whitespace
    while (pos < json.size() && (json[pos] == ':' || std::isspace(json[pos]))) {
        pos++;
    }
    
    if (pos >= json.size() || json[pos] != '[') {
        return "[]";
    }
    
    // Find matching closing bracket
    size_t start = pos;
    int bracket_count = 1;
    bool in_string = false;
    
    pos++; // Skip opening bracket
    
    while (pos < json.size() && bracket_count > 0) {
        char c = json[pos];
        
        if (c == '"' && (pos == 0 || json[pos-1] != '\\')) {
            in_string = !in_string;
        } else if (!in_string) {
            if (c == '[') {
                bracket_count++;
            } else if (c == ']') {
                bracket_count--;
            }
        }
        
        pos++;
    }
    
    if (bracket_count != 0) {
        return "[]";
    }
    
    return json.substr(start, pos - start);
}

std::vector<double> parse_json_array_doubles(const std::string& array_str) {
    std::vector<double> result;
    
    if (array_str.size() < 2 || array_str[0] != '[' || array_str[array_str.size() - 1] != ']') {
        return result;
    }
    
    std::string content = array_str.substr(1, array_str.size() - 2);
    
    // Split by commas
    size_t start = 0;
    size_t pos;
    
    while ((pos = content.find(',', start)) != std::string::npos) {
        std::string value = content.substr(start, pos - start);
        
        // Trim whitespace
        value.erase(0, value.find_first_not_of(" \t\n\r\f\v"));
        value.erase(value.find_last_not_of(" \t\n\r\f\v") + 1);
        
        try {
            result.push_back(std::stod(value));
        } catch (...) {
            // Skip invalid values
        }
        
        start = pos + 1;
    }
    
    // Last element
    std::string value = content.substr(start);
    value.erase(0, value.find_first_not_of(" \t\n\r\f\v"));
    value.erase(value.find_last_not_of(" \t\n\r\f\v") + 1);
    
    try {
        result.push_back(std::stod(value));
    } catch (...) {
        // Skip invalid values
    }
    
    return result;
}
