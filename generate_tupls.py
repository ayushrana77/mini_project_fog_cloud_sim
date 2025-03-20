import json
import random

# Load the JSON file with utf-8-sig encoding
with open("tuples.json", "r", encoding="utf-8-sig") as file:
    data = json.load(file)  # Assuming it's a list of dictionaries

# Dictionary to store at least one tuple per unique DataType
data_type_samples = {}

# Iterate and collect at least one sample for each DataType
for tuple_data in data:
    dtype = tuple_data.get("DataType", "Unknown")
    if dtype not in data_type_samples:
        data_type_samples[dtype] = tuple_data
        if len(data_type_samples) == 6:  # Stop when 6 unique DataTypes are found
            break

# Convert dictionary values to a list (6 tuples)
selected_tuples = list(data_type_samples.values())

# Select 4 more random tuples to make a total of 10
remaining_tuples = [t for t in data if t not in selected_tuples]
random.shuffle(remaining_tuples)
selected_tuples.extend(remaining_tuples[:4])  # Add 4 more tuples

# Save to Tuple10.json
with open("Tuple10.json", "w", encoding="utf-8") as output_file:
    json.dump(selected_tuples, output_file, indent=4)

print("Saved 10 tuples to Tuple10.json")
