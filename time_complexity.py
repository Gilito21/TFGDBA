import numpy as np
import matplotlib.pyplot as plt

# Configuration
k = 10  # Number of neighbors per image (fixed small constant)

# Input size: small dataset range (up to 200)
input_sizes = np.linspace(50, 200, 100)

# Simulated input parameters
n = input_sizes
v = n              # Number of vertices
m = n / 10         # Assume 10% of vertices are damaged

# Time complexity simulations
feature_extraction_time = n                      # O(n)
sequential_matching_time = n * k                 # O(n × k)
vertex_distance_time = v                         # O(v)
dbscan_time = m * np.log2(m)                     # O(m log m)
incremental_mapping_time = n ** 2                # O(n²)

# Plotting
plt.figure(figsize=(12, 7))

plt.plot(input_sizes, feature_extraction_time, label="Feature Extraction (O(n))")
plt.plot(input_sizes, sequential_matching_time, label="Sequential Matching (O(n × k))")
plt.plot(input_sizes, vertex_distance_time, label="Vertex Distance (O(v))")
plt.plot(input_sizes, dbscan_time, label="DBSCAN Clustering (O(m log m))")
plt.plot(input_sizes, incremental_mapping_time, label="Incremental Mapping (O(n²))")

# Labels and styling
plt.xlabel("Input Size (n, v, m)")
plt.ylabel("Relative Computational Time (unitless)")
plt.title("Time Complexity Growth (Input Size up to 200)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
