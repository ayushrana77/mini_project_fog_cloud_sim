import json
import os
import numpy as np
from collections import defaultdict

def analyze_task_sizes():
    """Analyze the data types and sizes in the tuple100k.json file"""
    
    # Path to data file
    filepath = os.path.join(os.getcwd(), 'tuple100k.json')
    
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found")
        return
    
    print(f"Loading tasks from {filepath}...")
    
    try:
        # Load the JSON file
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
        
        # Check if data is in expected format
        if not isinstance(data, list):
            print("Error: Data is not in expected format (not a list)")
            return
        
        print(f"Loaded {len(data)} tasks")
        
        # Track data types and sizes
        data_types = defaultdict(list)
        
        # Analyze each task
        for item in data:
            if 'DataType' in item and 'Size' in item:
                data_type = item['DataType']
                size = int(item['Size'])
                data_types[data_type].append(size)
        
        # Calculate statistics
        print("\n=== Data Type Statistics ===")
        print(f"{'Data Type':<15} {'Count':<10} {'Avg Size':<15} {'Min Size':<15} {'Max Size':<15}")
        print("-" * 70)
        
        # Sort by count (most common first)
        sorted_types = sorted(data_types.items(), key=lambda x: len(x[1]), reverse=True)
        
        # Display statistics for each type
        for data_type, sizes in sorted_types:
            count = len(sizes)
            avg_size = np.mean(sizes)
            min_size = np.min(sizes)
            max_size = np.max(sizes)
            
            print(f"{data_type:<15} {count:<10} {avg_size:<15.2f} {min_size:<15} {max_size:<15}")
        
        # Overall statistics
        all_sizes = [size for sizes in data_types.values() for size in sizes]
        print("\n=== Overall Statistics ===")
        print(f"Total tasks: {len(all_sizes)}")
        print(f"Average size: {np.mean(all_sizes):.2f}")
        print(f"Min size: {np.min(all_sizes)}")
        print(f"Max size: {np.max(all_sizes)}")
        
    except FileNotFoundError:
        print(f"Error: File {filepath} not found")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in {filepath}: {str(e)}")
    except Exception as e:
        print(f"Error analyzing tasks: {str(e)}")

if __name__ == "__main__":
    analyze_task_sizes() 