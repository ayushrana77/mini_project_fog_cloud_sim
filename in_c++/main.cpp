#include "FogCloudSim.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <memory>
#include <map>
#include <string>

using namespace std;

// Structure to hold algorithm results for comparison
struct AlgorithmResults {
    double avg_processing_time = 0.0;
    double avg_fog_time = 0.0;
    double avg_cloud_time = 0.0;
    vector<double> avg_power_consumption;
    double avg_queue_delay = 0.0;
    int fog_count = 0;
    int cloud_count = 0;
    int total_processed = 0;
    double fog_percentage = 0.0;
    double cloud_percentage = 0.0;
    map<string, map<string, int>> data_type_counts;
    double avg_node_selection_time = 0.0;
    double avg_alt_node_selection_time = 0.0;
    double avg_cloud_selection_time = 0.0;
};

// Function to run simulations and collect results
AlgorithmResults run_algorithm(const string& algorithm_name, vector<Task>& all_tasks, int batches_to_process) {
    cout << "\n=== Running " << algorithm_name << " ===" << endl;
    
    // Create fog nodes and cloud services
    vector<FogNode> fog_nodes;
    for (const auto& config : FOG_NODES_CONFIG) {
        fog_nodes.push_back(FogNode(config));
    }
    
    vector<CloudService> cloud_services;
    for (const auto& config : CLOUD_SERVICES_CONFIG) {
        cloud_services.push_back(CloudService(config));
    }
    
    // Create appropriate gateway based on algorithm name
    unique_ptr<BaseGateway> gateway;
    
    if (algorithm_name == "FCFSCooperation") {
        gateway = make_unique<FCFSCooperationGateway>(fog_nodes, cloud_services);
    } else if (algorithm_name == "FCFSNoCooperation") {
        gateway = make_unique<FCFSNoCooperationGateway>(fog_nodes, cloud_services);
    } else if (algorithm_name == "RandomCooperation") {
        gateway = make_unique<RandomCooperationGateway>(fog_nodes, cloud_services);
    } else if (algorithm_name == "RandomNoCooperation") {
        gateway = make_unique<RandomNoCooperationGateway>(fog_nodes, cloud_services);
    } else {
        throw runtime_error("Unknown algorithm name: " + algorithm_name);
    }
    
    // Process batches
    for (int i = 0; i < batches_to_process; i++) {
        cout << "Processing batch " << i + 1 << "..." << endl;
        
        // Get next batch of tasks
        vector<Task> batch = gateway->get_next_batch(all_tasks);
        if (batch.empty()) {
            cout << "No more tasks to process." << endl;
            break;
        }
        
        // Process the batch
        gateway->process_batch(batch);
        
        // Update metrics
        gateway->get_batch_metrics();
    }
    
    // Collect results
    AlgorithmResults results;
    
    // Calculate average processing times
    if (!gateway->metrics.fog_times.empty()) {
        results.avg_fog_time = accumulate(gateway->metrics.fog_times.begin(), 
                                         gateway->metrics.fog_times.end(), 0.0) / 
                                         gateway->metrics.fog_times.size();
    }
    
    if (!gateway->metrics.cloud_times.empty()) {
        results.avg_cloud_time = accumulate(gateway->metrics.cloud_times.begin(), 
                                           gateway->metrics.cloud_times.end(), 0.0) / 
                                           gateway->metrics.cloud_times.size();
    }
    
    results.total_processed = gateway->metrics.fog_times.size() + gateway->metrics.cloud_times.size();
    if (results.total_processed > 0) {
        double total_fog_time = accumulate(gateway->metrics.fog_times.begin(), 
                                          gateway->metrics.fog_times.end(), 0.0);
        double total_cloud_time = accumulate(gateway->metrics.cloud_times.begin(), 
                                            gateway->metrics.cloud_times.end(), 0.0);
        results.avg_processing_time = (total_fog_time + total_cloud_time) / results.total_processed;
    }
    
    // Calculate power consumption
    for (const auto& fog : fog_nodes) {
        double avg_power = 0.0;
        if (!fog.power_log.empty()) {
            avg_power = accumulate(fog.power_log.begin(), fog.power_log.end(), 0.0) / fog.power_log.size();
        }
        results.avg_power_consumption.push_back(avg_power);
    }
    
    // Calculate queue delays
    if (!gateway->metrics.queue_delays.empty()) {
        results.avg_queue_delay = accumulate(gateway->metrics.queue_delays.begin(), 
                                            gateway->metrics.queue_delays.end(), 0.0) / 
                                            gateway->metrics.queue_delays.size();
    }
    
    // Calculate task distribution
    if (algorithm_name == "FCFSCooperation" || algorithm_name == "FCFSNoCooperation") {
        auto* fcfsGateway = dynamic_cast<FCFSCooperationGateway*>(gateway.get());
        if (fcfsGateway) {
            for (const auto& [data_type, counts] : fcfsGateway->data_type_counts) {
                results.fog_count += counts.at("fog");
                results.cloud_count += counts.at("cloud");
                results.data_type_counts[data_type] = counts;
            }
        } else {
            auto* fcfsNoCoop = dynamic_cast<FCFSNoCooperationGateway*>(gateway.get());
            if (fcfsNoCoop) {
                for (const auto& [data_type, counts] : fcfsNoCoop->data_type_counts) {
                    results.fog_count += counts.at("fog");
                    results.cloud_count += counts.at("cloud");
                    results.data_type_counts[data_type] = counts;
                }
            }
        }
    } else if (algorithm_name == "RandomCooperation" || algorithm_name == "RandomNoCooperation") {
        auto* randomGateway = dynamic_cast<RandomCooperationGateway*>(gateway.get());
        if (randomGateway) {
            for (const auto& [data_type, counts] : randomGateway->data_type_counts) {
                results.fog_count += counts.at("fog");
                results.cloud_count += counts.at("cloud");
                results.data_type_counts[data_type] = counts;
            }
        } else {
            auto* randomNoCoop = dynamic_cast<RandomNoCooperationGateway*>(gateway.get());
            if (randomNoCoop) {
                for (const auto& [data_type, counts] : randomNoCoop->data_type_counts) {
                    results.fog_count += counts.at("fog");
                    results.cloud_count += counts.at("cloud");
                    results.data_type_counts[data_type] = counts;
                }
            }
        }
    }
    
    if (results.total_processed > 0) {
        results.fog_percentage = (static_cast<double>(results.fog_count) / results.total_processed) * 100;
        results.cloud_percentage = (static_cast<double>(results.cloud_count) / results.total_processed) * 100;
    }
    
    // Calculate selection times
    if (!gateway->metrics.node_selection_time.empty()) {
        results.avg_node_selection_time = accumulate(gateway->metrics.node_selection_time.begin(), 
                                                   gateway->metrics.node_selection_time.end(), 0.0) / 
                                                   gateway->metrics.node_selection_time.size();
    }
    
    if (!gateway->metrics.cloud_selection_time.empty()) {
        results.avg_cloud_selection_time = accumulate(gateway->metrics.cloud_selection_time.begin(), 
                                                     gateway->metrics.cloud_selection_time.end(), 0.0) / 
                                                     gateway->metrics.cloud_selection_time.size();
    }
    
    return results;
}

// Function to display comparison between algorithms
void display_comparison(const map<string, AlgorithmResults>& all_results) {
    cout << "\n====================================================" << endl;
    cout << "             ALGORITHM COMPARISON RESULTS            " << endl;
    cout << "====================================================" << endl;
    
    // Processing times
    cout << "\n=== Processing Times (ms) ===" << endl;
    cout << left << setw(20) << "Algorithm" << " | " 
         << setw(10) << "Total" << " | " 
         << setw(10) << "Fog" << " | " 
         << setw(10) << "Cloud" << endl;
    cout << string(60, '-') << endl;
    
    for (const auto& [name, results] : all_results) {
        cout << left << setw(20) << name << " | " 
             << setw(10) << fixed << setprecision(2) << results.avg_processing_time << " | " 
             << setw(10) << results.avg_fog_time << " | " 
             << setw(10) << results.avg_cloud_time << endl;
    }
    
    // Power consumption
    cout << "\n=== Average Power Consumption (W) ===" << endl;
    cout << left << setw(20) << "Algorithm" << " | ";
    
    // Print header for each fog node
    const auto& firstResult = all_results.begin()->second;
    for (size_t i = 0; i < firstResult.avg_power_consumption.size(); i++) {
        cout << "Fog" << i + 1 << " ";
        if (i < firstResult.avg_power_consumption.size() - 1) cout << "| ";
    }
    cout << endl;
    cout << string(60, '-') << endl;
    
    for (const auto& [name, results] : all_results) {
        cout << left << setw(20) << name << " | ";
        for (size_t i = 0; i < results.avg_power_consumption.size(); i++) {
            cout << fixed << setprecision(2) << results.avg_power_consumption[i] << " ";
            if (i < results.avg_power_consumption.size() - 1) cout << "| ";
        }
        cout << endl;
    }
    
    // Queue delays
    cout << "\n=== Queue Delays (ms) ===" << endl;
    cout << left << setw(20) << "Algorithm" << " | " 
         << setw(15) << "Avg Delay" << endl;
    cout << string(40, '-') << endl;
    
    for (const auto& [name, results] : all_results) {
        cout << left << setw(20) << name << " | " 
             << setw(15) << fixed << setprecision(2) << results.avg_queue_delay << endl;
    }
    
    // Task distribution
    cout << "\n=== Task Distribution ===" << endl;
    cout << left << setw(20) << "Algorithm" << " | " 
         << setw(15) << "Fog" << " | " 
         << setw(15) << "Cloud" << " | " 
         << setw(10) << "Total" << endl;
    cout << string(70, '-') << endl;
    
    for (const auto& [name, results] : all_results) {
        cout << left << setw(20) << name << " | " 
             << setw(5) << results.fog_count << " (" << setw(6) << fixed << setprecision(1) << results.fog_percentage << "%) | " 
             << setw(5) << results.cloud_count << " (" << setw(6) << results.cloud_percentage << "%) | " 
             << setw(10) << results.total_processed << endl;
    }
    
    // Data type distribution for each algorithm
    cout << "\n=== Data Type Distribution ===" << endl;
    
    for (const auto& [alg_name, results] : all_results) {
        cout << "\n" << alg_name << ":" << endl;
        cout << left << setw(15) << "Data Type" << " | " 
             << setw(10) << "Fog Count" << " | " 
             << setw(10) << "Cloud Count" << " | " 
             << setw(10) << "Total" << " | " 
             << setw(10) << "Fog %" << endl;
        cout << string(70, '-') << endl;
        
        for (const auto& [data_type, counts] : results.data_type_counts) {
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
    }
    
    // Selection times
    cout << "\n=== Selection Times (ms) ===" << endl;
    cout << left << setw(20) << "Algorithm" << " | " 
         << setw(15) << "Node Selection" << " | " 
         << setw(15) << "Cloud Selection" << endl;
    cout << string(60, '-') << endl;
    
    for (const auto& [name, results] : all_results) {
        cout << left << setw(20) << name << " | " 
             << setw(15) << fixed << setprecision(4) << results.avg_node_selection_time << " | " 
             << setw(15) << results.avg_cloud_selection_time << endl;
    }
}

int main() {
    cout << "=== Fog-Cloud Simulation System ===" << endl;
    cout << "C++ Implementation" << endl;
    
    try {
        // Load tasks from JSON file
        string task_file = "Tupel100k.json";
        cout << "Loading tasks from " << task_file << "..." << endl;
        vector<Task> all_tasks = load_tasks(task_file);
        cout << "Loaded " << all_tasks.size() << " tasks." << endl;
        
        // Number of batches to process for each algorithm
        int batches_to_process = 5; // Adjust this as needed
        
        // Map to store results from all algorithms
        map<string, AlgorithmResults> all_results;
        
        // Run all algorithms and collect results
        vector<string> algorithms = {
            "FCFSCooperation", 
            "FCFSNoCooperation", 
            "RandomCooperation", 
            "RandomNoCooperation"
        };
        
        for (const auto& alg_name : algorithms) {
            vector<Task> tasks_copy = all_tasks; // Create a copy for each algorithm
            all_results[alg_name] = run_algorithm(alg_name, tasks_copy, batches_to_process);
        }
        
        // Display comparison between all algorithms
        display_comparison(all_results);
        
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}
