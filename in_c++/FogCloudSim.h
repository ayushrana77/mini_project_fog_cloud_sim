#ifndef FOG_CLOUD_SIM_H
#define FOG_CLOUD_SIM_H

#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <set>
#include <random>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <memory>
#include <functional>
#include <any>

// Configuration Constants
constexpr int BATCH_SIZE = 1000;
constexpr int BATCHES_BEFORE_RESET = 10;
constexpr int BATCH_RESET_DELAY = 100; // ms between batch resets

// Real-world Variability Factors
constexpr double NETWORK_CONGESTION_MIN = 0.6; // Network throughput multiplier (min)
constexpr double NETWORK_CONGESTION_MAX = 1.5; // Network throughput multiplier (max)
constexpr double PROCESSING_VARIATION_MIN = 0.7; // Processing time multiplier (min)
constexpr double PROCESSING_VARIATION_MAX = 1.8; // Processing time multiplier (max)
constexpr double BACKGROUND_LOAD_MIN = 0.0; // Additional background load (min)
constexpr double BACKGROUND_LOAD_MAX = 0.25; // Additional background load (max)
constexpr double HIGH_UTILIZATION_THRESHOLD = 85.0; // Force cloud migration when node utilization exceeds this
constexpr double NETWORK_CONGESTION_THRESHOLD = 1.15; // Force cloud migration when network congestion exceeds this

// Random factors for baseline calculations
constexpr double BASE_VARIATION_FACTOR = 0.4; // How much baseline processing times can vary

constexpr double TRANSMISSION_LATENCY = 1.0; // ms
constexpr double EARTH_RADIUS_KM = 6371;
constexpr double MAX_SIMULATION_TIME = 1000; // seconds
constexpr double NODE_CHECK_DELAY = 0.1; // ms
constexpr double CLOUD_SELECTION_DELAY = 2.5; // ms

// Location type (latitude, longitude)
using Location = std::pair<double, double>;

// Forward declarations
class Task;
class FogNode;
class CloudService;
class BaseGateway;

// Task class
class Task {
public:
    std::string id;
    int size;
    std::string name;
    double mips;
    int number_of_pes;
    int ram;
    int bw;
    std::string data_type;
    Location location;
    std::string device_type;
    double arrival_time = 0.0;
    int batch_id = 0;
    int current_allocated_size = 0;
    int current_allocated_ram = 0;
    int current_allocated_bw = 0;
    bool being_instantiated = true;
    int data_percentage = 100;
    bool is_reversed = false;
    bool is_server_found = false;
    bool is_cloud_served = false;
    bool is_served = false;
    double queue_delay = 0.0;
    double internal_processing_time = 0.0;
    std::vector<double> tuple_times;
    int fog_level_served = 0;
    bool is_served_by_fc_cloud = false;
    std::string creation_time = "";
    double queue_wait = 0.0;
    double temperature = 0.0;
    bool fog_candidate = true;
    double processing_start_time = 0.0;
    double processing_end_time = 0.0;
    std::string processor_node = "";

    // Constructor
    Task(const std::string& id, int size, const std::string& name, double mips, 
         int number_of_pes, int ram, int bw, const std::string& data_type, 
         const Location& location, const std::string& device_type);

    // Methods
    bool should_go_to_fog() const;
    bool is_small_task() const;
    std::pair<double, double> estimate_fog_processing_time(double fog_mips, double available_efficiency = 1.0) const;
    std::pair<double, double> estimate_transmission_time(double bandwidth, double network_factor = 1.0) const;
    std::pair<double, double> estimate_cloud_processing_time(double cloud_mips = 15000) const;
    std::string get_size_category() const;
};

// Fog Node class
class FogNode {
public:
    std::string name;
    Location location;
    double down_bw;
    double up_bw;
    double mips;
    int num_pes;
    int ram;
    int total_storage;
    int used_storage;
    std::vector<Task> queue;
    double utilization;
    std::vector<double> power_log;
    double busy_until;
    int num_devices;
    int available_ram;
    double available_mips;
    int max_queue_size;
    int total_processed;
    double sim_clock;
    std::vector<std::pair<double, double>> resource_release_schedule;
    int cumulative_processed;
    double cumulative_utilization;
    double network_congestion;
    double background_load;
    double last_congestion_update;
    double congestion_update_interval;

    // Constructor
    FogNode(const std::map<std::string, std::any>& config);

    // Methods
    double calculate_power();
    void update_network_congestion(double current_time);
    bool can_accept_task(const Task& task, double current_time);
    double process(Task& task, double arrival_time);
    void update_resources(double current_time);
    void reset();
};

// Cloud Service class
class CloudService {
public:
    std::string name;
    Location location;
    int ram;
    double mips;
    double bw;
    double busy_until;
    double current_load;
    std::vector<Task> queue;
    int max_queue_size;
    double network_latency;
    double last_latency_update;
    double latency_update_interval;
    double cloud_congestion;

    // Constructor
    CloudService(const std::map<std::string, std::any>& config);

    // Methods
    void reset();
    void update_network_conditions(double current_time);
    double process(Task& task, double current_time = 0.0, const std::string& policy_type = "");
};

// Helper function for haversine distance
double haversine(const Location& loc1, const Location& loc2);

// Helper functions for JSON parsing
std::string extract_json_string(const std::string& json, const std::string& key);
double extract_json_double(const std::string& json, const std::string& key);
bool extract_json_bool(const std::string& json, const std::string& key);
std::string extract_json_array(const std::string& json, const std::string& key);
std::vector<double> parse_json_array_doubles(const std::string& array_str);

// Base Gateway class
class BaseGateway {
public:
    std::vector<FogNode> fog_nodes;
    std::vector<CloudService> cloud_services;
    int batch_size;
    int current_batch;
    double sim_clock;
    std::map<int, std::map<std::string, std::vector<Task>>> batch_assignments;
    std::map<std::string, std::map<int, int>> device_commitments;
    std::set<std::string> processed_tasks;
    
    struct Metrics {
        std::vector<double> fog_times;
        std::vector<double> cloud_times;
        std::vector<double> node_selection_time;
        std::vector<double> cloud_selection_time;
        std::vector<double> queue_delays;
        int fog_first_count = 0;
        int cloud_direct_count = 0;
        
        struct BatchMetrics {
            std::vector<int> fog_batch_counts;
            std::vector<int> cloud_batch_counts;
            std::vector<double> batch_times;
            std::vector<double> fog_utilization;
            std::vector<double> cloud_utilization;
        } batch_metrics;
    } metrics;

    std::map<std::string, std::vector<double>> commitment_stats;
    
    // Random number generation
    std::mt19937 rng;

    // Constructor
    BaseGateway(const std::vector<FogNode>& fog_nodes, const std::vector<CloudService>& cloud_services);

    // Methods
    void reset_nodes();
    int get_node_device_count(const std::string& node_name, int batch_id);
    int get_total_node_commitments(const std::string& node_name);
    int get_total_commitments();
    bool is_fog_available(FogNode& fog, const Task& task, int current_batch);
    void commit_fog_resources(FogNode& fog, const Task& task, int current_batch);
    bool is_bulk_data(const Task& task);
    virtual std::vector<Task> get_next_batch(const std::vector<Task>& all_tasks);
    void process_batch(std::vector<Task>& tasks);
    void process_cloud(Task& task);
    void get_batch_metrics();
    void track_resource_commitments(const Task& task, int commitment_duration);
    void print_commitment_stats();
    
    // Virtual method for polymorphism
    virtual int offload_task(Task& task) = 0;
    virtual void print_fog_status() = 0;
    
    // Virtual destructor
    virtual ~BaseGateway() = default;
};

// FCFS Cooperation Gateway
class FCFSCooperationGateway : public BaseGateway {
public:
    std::map<std::string, std::map<std::string, int>> data_type_counts;
    bool verbose_output;

    // Constructor
    FCFSCooperationGateway(const std::vector<FogNode>& fog_nodes, const std::vector<CloudService>& cloud_services);

    // Override methods
    int offload_task(Task& task) override;
    void print_fog_status() override;
};

// FCFS No Cooperation Gateway
class FCFSNoCooperationGateway : public BaseGateway {
public:
    std::map<std::string, std::map<std::string, int>> data_type_counts;
    bool verbose_output;

    // Constructor
    FCFSNoCooperationGateway(const std::vector<FogNode>& fog_nodes, const std::vector<CloudService>& cloud_services);

    // Override methods
    int offload_task(Task& task) override;
    void print_fog_status() override;
};

// Random Cooperation Gateway
class RandomCooperationGateway : public BaseGateway {
public:
    std::map<std::string, std::map<std::string, int>> data_type_counts;
    std::map<std::string, std::map<std::string, int>> task_commitment_stats;
    bool verbose_output;

    // Constructor
    RandomCooperationGateway(const std::vector<FogNode>& fog_nodes, const std::vector<CloudService>& cloud_services);

    // Override methods
    int offload_task(Task& task) override;
    void print_fog_status() override;
    
    // Random task selection
    std::vector<Task> get_next_batch(const std::vector<Task>& all_tasks) override;
};

// Random No Cooperation Gateway
class RandomNoCooperationGateway : public BaseGateway {
public:
    std::map<std::string, std::map<std::string, int>> data_type_counts;
    std::map<std::string, std::map<std::string, int>> task_commitment_stats;
    bool verbose_output;

    // Constructor
    RandomNoCooperationGateway(const std::vector<FogNode>& fog_nodes, const std::vector<CloudService>& cloud_services);

    // Override methods
    int offload_task(Task& task) override;
    void print_fog_status() override;
    
    // Random task selection
    std::vector<Task> get_next_batch(const std::vector<Task>& all_tasks) override;
};

// Load tasks from JSON file
std::vector<Task> load_tasks(const std::string& filepath);

// Global configurations for nodes
extern std::vector<std::map<std::string, std::any>> FOG_NODES_CONFIG;
extern std::vector<std::map<std::string, std::any>> CLOUD_SERVICES_CONFIG;

#endif // FOG_CLOUD_SIM_H
