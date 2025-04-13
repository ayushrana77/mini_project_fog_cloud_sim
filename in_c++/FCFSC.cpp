#include "FogCloudSim.h"
#include <iostream>
#include <iomanip>
#include <chrono>

using namespace std;

// FCFSCooperationGateway implementation
FCFSCooperationGateway::FCFSCooperationGateway(const vector<FogNode>& fog_nodes, const vector<CloudService>& cloud_services)
    : BaseGateway(fog_nodes, cloud_services) {
    // Initialize data type counts
    data_type_counts = {
        {"Abrupt", {{"fog", 0}, {"cloud", 0}}},
        {"Large", {{"fog", 0}, {"cloud", 0}}},
        {"LocationBased", {{"fog", 0}, {"cloud", 0}}},
        {"Bulk", {{"fog", 0}, {"cloud", 0}}},
        {"Medical", {{"fog", 0}, {"cloud", 0}}},
        {"SmallTextual", {{"fog", 0}, {"cloud", 0}}},
        {"Multimedia", {{"fog", 0}, {"cloud", 0}}}
    };
    verbose_output = false;
}

int FCFSCooperationGateway::offload_task(Task& task) {
    // Algorithm 1: Global Gateway With FCFS Tuples and Cooperation Policy
    
    auto start_time = chrono::high_resolution_clock::now();
    
    // Identify if task involves bulk data that should go directly to cloud
    if (is_bulk_data(task)) {
        auto end_time = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time).count() / 1000.0;
        metrics.node_selection_time.push_back(duration);
        
        // Process on cloud
        process_cloud(task);
        metrics.cloud_direct_count++;
        
        // Track allocation by data type
        if (data_type_counts.find(task.data_type) != data_type_counts.end()) {
            data_type_counts[task.data_type]["cloud"]++;
        }
        
        return 1; // Processed on cloud
    }
    
    // Check if task should go to fog nodes first
    if (task.should_go_to_fog()) {
        metrics.fog_first_count++;
        
        // Find best fog node
        FogNode* best_fog = nullptr;
        double min_distance = numeric_limits<double>::max();
        
        for (auto& fog : fog_nodes) {
            double distance = haversine(task.location, fog.location);
            if (distance < min_distance && is_fog_available(fog, task, current_batch)) {
                min_distance = distance;
                best_fog = &fog;
            }
        }
        
        auto end_time = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time).count() / 1000.0;
        metrics.node_selection_time.push_back(duration);
        
        if (best_fog) {
            // Process on the selected fog node
            double processing_time = best_fog->process(task, sim_clock);
            metrics.fog_times.push_back(processing_time);
            metrics.queue_delays.push_back(task.queue_delay);
            
            // Commit resources
            commit_fog_resources(*best_fog, task, current_batch);
            
            // Track allocation by data type
            if (data_type_counts.find(task.data_type) != data_type_counts.end()) {
                data_type_counts[task.data_type]["fog"]++;
            }
            
            return 0; // Processed on fog
        }
        
        // If no suitable fog node is found, try cooperation policy (look for any available node)
        start_time = chrono::high_resolution_clock::now();
        
        for (auto& fog : fog_nodes) {
            if (is_fog_available(fog, task, current_batch)) {
                best_fog = &fog;
                break;
            }
        }
        
        end_time = chrono::high_resolution_clock::now();
        duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time).count() / 1000.0;
        metrics.node_selection_time.push_back(duration);
        
        if (best_fog) {
            // Process on the alternative fog node
            double processing_time = best_fog->process(task, sim_clock);
            metrics.fog_times.push_back(processing_time);
            metrics.queue_delays.push_back(task.queue_delay);
            
            // Commit resources
            commit_fog_resources(*best_fog, task, current_batch);
            
            // Track allocation by data type
            if (data_type_counts.find(task.data_type) != data_type_counts.end()) {
                data_type_counts[task.data_type]["fog"]++;
            }
            
            return 0; // Processed on fog
        }
    }
    
    // If no fog node is available, process on cloud
    process_cloud(task);
    
    // Track allocation by data type
    if (data_type_counts.find(task.data_type) != data_type_counts.end()) {
        data_type_counts[task.data_type]["cloud"]++;
    }
    
    return 1; // Processed on cloud
}

void FCFSCooperationGateway::print_fog_status() {
    cout << "\n=== Fog Node Status ===" << endl;
    for (const auto& fog : fog_nodes) {
        cout << fog.name << ": Utilization = " << fixed << setprecision(2) << fog.utilization 
             << "%, Processed = " << fog.total_processed << endl;
    }
}

/*
int main() {
    cout << "=== FCFS Cooperation Algorithm ===" << endl;
    
    // Create fog nodes and cloud services from config
    vector<FogNode> fog_nodes;
    for (const auto& config : FOG_NODES_CONFIG) {
        fog_nodes.push_back(FogNode(config));
    }
    
    vector<CloudService> cloud_services;
    for (const auto& config : CLOUD_SERVICES_CONFIG) {
        cloud_services.push_back(CloudService(config));
    }
    
    // Create gateway
    FCFSCooperationGateway gateway(fog_nodes, cloud_services);
    
    try {
        // Load tasks from JSON file
        string task_file = "tasks.json";
        cout << "Loading tasks from " << task_file << "..." << endl;
        vector<Task> all_tasks = load_tasks(task_file);
        cout << "Loaded " << all_tasks.size() << " tasks." << endl;
        
        // Process batches of tasks
        int batches_to_process = 5; // Adjust this as needed
        cout << "\nProcessing " << batches_to_process << " batches..." << endl;
        
        for (int i = 0; i < batches_to_process; i++) {
            cout << "Processing batch " << i + 1 << "..." << endl;
            
            // Get next batch
            vector<Task> batch = gateway.get_next_batch(all_tasks);
            if (batch.empty()) {
                cout << "No more tasks to process." << endl;
                break;
            }
            
            // Process the batch
            gateway.process_batch(batch);
            
            // Update metrics
            gateway.get_batch_metrics();
        }
        
        // Calculate averages
        double avg_fog_time = 0.0;
        if (!gateway.metrics.fog_times.empty()) {
            avg_fog_time = accumulate(gateway.metrics.fog_times.begin(), 
                                     gateway.metrics.fog_times.end(), 0.0) / 
                                     gateway.metrics.fog_times.size();
        }
        
        double avg_cloud_time = 0.0;
        if (!gateway.metrics.cloud_times.empty()) {
            avg_cloud_time = accumulate(gateway.metrics.cloud_times.begin(), 
                                       gateway.metrics.cloud_times.end(), 0.0) / 
                                       gateway.metrics.cloud_times.size();
        }
        
        double avg_processing_time = 0.0;
        int total_processed = gateway.metrics.fog_times.size() + gateway.metrics.cloud_times.size();
        if (total_processed > 0) {
            double total_fog_time = accumulate(gateway.metrics.fog_times.begin(), 
                                              gateway.metrics.fog_times.end(), 0.0);
            double total_cloud_time = accumulate(gateway.metrics.cloud_times.begin(), 
                                                gateway.metrics.cloud_times.end(), 0.0);
            avg_processing_time = (total_fog_time + total_cloud_time) / total_processed;
        }
        
        // Print results
        cout << "\n=== Results ===" << endl;
        
        // Task distribution
        int total_processed_tasks = gateway.processed_tasks.size();
        int fog_count = 0;
        for (const auto& [data_type, counts] : gateway.data_type_counts) {
            fog_count += counts.at("fog");
        }
        int cloud_count = total_processed_tasks - fog_count;
        
        cout << "\n=== Processing Times (ms) ===" << endl;
        cout << "FCFSCooperation: Total = " << fixed << setprecision(2) << avg_processing_time 
             << ", Fog = " << avg_fog_time << ", Cloud = " << avg_cloud_time << endl;
        
        // Power consumption
        cout << "\n=== Power Consumption (W) ===" << endl;
        cout << "FCFSCooperation: [";
        for (size_t i = 0; i < fog_nodes.size(); i++) {
            double avg_power = accumulate(fog_nodes[i].power_log.begin(), 
                                         fog_nodes[i].power_log.end(), 0.0) / 
                                         fog_nodes[i].power_log.size();
            cout << avg_power;
            if (i < fog_nodes.size() - 1) cout << ", ";
        }
        cout << "]" << endl;
        
        // Queue delays
        double avg_queue_delay = 0.0;
        if (!gateway.metrics.queue_delays.empty()) {
            avg_queue_delay = accumulate(gateway.metrics.queue_delays.begin(), 
                                        gateway.metrics.queue_delays.end(), 0.0) / 
                                        gateway.metrics.queue_delays.size();
        }
        cout << "\n=== Queue Delays (ms) ===" << endl;
        cout << "FCFSCooperation: " << fixed << setprecision(2) << avg_queue_delay << endl;
        
        // Task distribution
        cout << "\n=== Task Distribution ===" << endl;
        double fog_percent = (total_processed > 0) ? (static_cast<double>(fog_count) / total_processed) * 100 : 0.0;
        double cloud_percent = (total_processed > 0) ? (static_cast<double>(cloud_count) / total_processed) * 100 : 0.0;
        cout << "FCFSCooperation: Fog = " << fog_count << " (" << fog_percent << "%), Cloud = " 
             << cloud_count << " (" << cloud_percent << "%)" << endl;
        
        // Data type distribution
        cout << "\n=== Data Type Distribution ===" << endl;
        cout << left << setw(15) << "Data Type" << " | " 
             << setw(10) << "Fog Count" << " | " 
             << setw(10) << "Cloud Count" << " | " 
             << setw(10) << "Total" << " | " 
             << setw(10) << "Fog %" << endl;
        cout << string(70, '-') << endl;
        
        for (const auto& [data_type, counts] : gateway.data_type_counts) {
            int fog_type_count = counts.at("fog");
            int cloud_type_count = counts.at("cloud");
            int type_total = fog_type_count + cloud_type_count;
            double fog_type_pct = (type_total > 0) ? (static_cast<double>(fog_type_count) / type_total) * 100 : 0.0;
            
            if (type_total > 0) {
                cout << left << setw(15) << data_type << " | " 
                     << setw(10) << fog_type_count << " | " 
                     << setw(10) << cloud_type_count << " | " 
                     << setw(10) << type_total << " | " 
                     << setw(10) << fixed << setprecision(1) << fog_type_pct << "%" << endl;
            }
        }
        
        // Selection times
        double avg_node_selection = 0.0;
        if (!gateway.metrics.node_selection_time.empty()) {
            avg_node_selection = accumulate(gateway.metrics.node_selection_time.begin(), 
                                           gateway.metrics.node_selection_time.end(), 0.0) / 
                                           gateway.metrics.node_selection_time.size();
        }
        
        double avg_alt_node_selection = 0.0; // Placeholder, we don't have separate metrics for alt node selection
        
        double avg_cloud_selection = 0.0;
        if (!gateway.metrics.cloud_selection_time.empty()) {
            avg_cloud_selection = accumulate(gateway.metrics.cloud_selection_time.begin(), 
                                            gateway.metrics.cloud_selection_time.end(), 0.0) / 
                                            gateway.metrics.cloud_selection_time.size();
        }
        
        cout << "\n=== Selection Times (ms) ===" << endl;
        cout << "FCFSCooperation: Node = " << fixed << setprecision(4) << avg_node_selection 
             << ", Alt Node = " << avg_alt_node_selection 
             << ", Cloud = " << avg_cloud_selection << endl;
        
        // Print resource commitment stats
        gateway.print_commitment_stats();
        
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
    }
    
    return 0;
}
*/
