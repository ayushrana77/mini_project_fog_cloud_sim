#include "FogCloudSim.h"
#include <iostream>
#include <fstream>
#include <random>
#include <ctime>
#include <string>
#include <vector>
// #include <nlohmann/json.hpp>

// using json = nlohmann::json;
using namespace std;

// Structure to define task parameters
struct TaskTypeConfig {
    string data_type;
    double size_min;
    double size_max;
    double processing_time_min;
    double processing_time_max;
    bool fog_suitable;
    double fog_probability;
    double location_variability;
};

// Commenting out the main function to avoid multiple definition errors
/*
int main() {
    cout << "=== Task Generator for Fog-Cloud Simulation ===" << endl;
    
    // Define task types and their characteristics
    vector<TaskTypeConfig> task_types = {
        {"Abrupt", 1.0, 5.0, 10.0, 50.0, true, 0.8, 0.2},
        {"Large", 20.0, 50.0, 100.0, 500.0, false, 0.2, 0.1},
        {"LocationBased", 2.0, 10.0, 20.0, 100.0, true, 0.9, 0.8},
        {"Bulk", 100.0, 500.0, 500.0, 2000.0, false, 0.1, 0.1},
        {"Medical", 5.0, 20.0, 50.0, 200.0, true, 0.7, 0.3},
        {"SmallTextual", 0.5, 2.0, 5.0, 20.0, true, 0.9, 0.2},
        {"Multimedia", 10.0, 40.0, 80.0, 300.0, false, 0.3, 0.2}
    };
    
    // Define fog nodes location (lat, lon)
    vector<pair<double, double>> fog_locations = {
        {37.7749, -122.4194},  // San Francisco
        {40.7128, -74.0060},   // New York
        {51.5074, -0.1278},    // London
        {35.6762, 139.6503},   // Tokyo
        {28.6139, 77.2090}     // New Delhi
    };
    
    // Random number generation
    random_device rd;
    mt19937 gen(rd());
    
    // Task parameters
    int num_tasks = 100;           // Total number of tasks to generate
    double arrival_time_min = 0.0; // Minimum arrival time (ms)
    double arrival_time_max = 1000.0; // Maximum arrival time (ms)
    
    cout << "Generating " << num_tasks << " tasks..." << endl;
    
    // Create tasks
    string tasks_json = "[";
    
    for (int i = 0; i < num_tasks; i++) {
        // Randomly select a task type
        uniform_int_distribution<> task_type_dist(0, task_types.size() - 1);
        int type_index = task_type_dist(gen);
        TaskTypeConfig& task_type = task_types[type_index];
        
        // Generate random parameters
        uniform_real_distribution<> size_dist(task_type.size_min, task_type.size_max);
        uniform_real_distribution<> proc_dist(task_type.processing_time_min, task_type.processing_time_max);
        uniform_real_distribution<> arrival_dist(arrival_time_min, arrival_time_max);
        uniform_real_distribution<> fog_dist(0.0, 1.0);
        
        double data_size = size_dist(gen);
        double required_processing_time = proc_dist(gen);
        double arrival_time = arrival_dist(gen);
        bool should_go_to_fog = fog_dist(gen) < task_type.fog_probability;
        
        // Select a nearby fog node location with some variability
        uniform_int_distribution<> loc_dist(0, fog_locations.size() - 1);
        int base_loc_index = loc_dist(gen);
        
        // Add variability to location
        normal_distribution<> lat_var(0.0, task_type.location_variability);
        normal_distribution<> lon_var(0.0, task_type.location_variability);
        
        double lat = fog_locations[base_loc_index].first + lat_var(gen);
        double lon = fog_locations[base_loc_index].second + lon_var(gen);
        
        // Clamp coordinates to valid ranges
        lat = max(-90.0, min(90.0, lat));
        lon = max(-180.0, min(180.0, lon));
        
        // Create the task JSON
        string task = "{\"id\": " + to_string(i + 1) +
                      ", \"data_type\": \"" + task_type.data_type + "\"" +
                      ", \"data_size\": " + to_string(data_size) +
                      ", \"arrival_time\": " + to_string(arrival_time) +
                      ", \"required_processing_time\": " + to_string(required_processing_time) +
                      ", \"location\": [" + to_string(lat) + ", " + to_string(lon) + "]" +
                      ", \"should_go_to_fog\": " + (should_go_to_fog ? "true" : "false") + "}";
        
        tasks_json += task;
        if (i < num_tasks - 1) {
            tasks_json += ", ";
        }
    }
    
    tasks_json += "]";
    
    // Write to file
    string filename = "tasks.json";
    ofstream file(filename);
    if (file.is_open()) {
        file << tasks_json;
        file.close();
        cout << "Successfully generated " << num_tasks << " tasks and saved to " << filename << endl;
    } else {
        cerr << "Error: Could not open file for writing." << endl;
        return 1;
    }
    
    return 0;
}
*/
