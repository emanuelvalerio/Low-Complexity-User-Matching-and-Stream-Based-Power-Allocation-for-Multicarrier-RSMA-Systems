import pandas as pd
import matplotlib.pyplot as plt
import os

# Simulation for Figure 3.B from reference paper

# Get the directory where this script (results.py) is saved
script_directory = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the CSV file
# Adjust 'Results' if the folder name is different on your machine
file_path = os.path.join(script_directory, 'Results', 'sum_rate_x_num_subcarriers_P_100W_8_users.csv')

print(f"Attempting to read file at: {file_path}") # Debugging info

try:
    # Load the CSV file
    df = pd.read_csv(file_path)
    print("File loaded successfully!")
except FileNotFoundError:
    print("\nERROR: The file was not found.")
    print("Check if the 'Results' folder exists in the same location as the script.")
    print("Check if the filename is exact (beware of hidden .csv.csv extensions).")
    exit()

# Group by the number of subcarriers ('N') and calculate the mean for each value
# This averages the results over all Monte Carlo iterations
avg_rates = df.groupby('N').mean()

# === LEGEND CONFIGURATION ===
# Map the raw CSV column names to the desired plot labels (Acronyms)
legend_names = {
    "Sum Rate Stream-Based-PA_lc": "WGOBUM-SBPA",
    "Sum Rate Stream-Based-PA_tum": "TUMUM-SBPA",
    "Sum Rate Stream-Based-PA_rand": "RUM-SBPA",
    # Add others if necessary:
    # "Sum Rate EPA_lc": "EPA (Low Complexity)",
}

# Plot the graph
plt.figure(figsize=(10, 6))

# Iterate through columns to plot each algorithm
for col in avg_rates.columns:
    if col == "Iteration": continue # Ensure we don't plot the iteration count
    
    # Try to get the mapped name from the dictionary. If not found, use the original (col)
    label_name = legend_names.get(col, col)
    
    plt.plot(avg_rates.index, avg_rates[col], marker='o', label=label_name)

# Chart Configuration
plt.xlabel('Number of Subcarriers', fontsize=12)
plt.ylabel('Average Sum Rate (bps/Hz)', fontsize=12)
plt.title('Average Sum Rate vs. Number of Subcarriers', fontsize=14)
plt.legend(title='Algorithms', loc='best', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7) # Softer grid lines
plt.tight_layout() # Adjust layout to prevent clipping
plt.show()