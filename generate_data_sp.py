import os

import numpy as np
import pandas as pd

# CONFIGURATION (You must verify these against VolGAN.py)
# Total columns = number of moneyness points (m) * number of maturities (tau)
# For example, if the grid is 5 moneyness points by 4 maturities, flattened_size = 20.
flattened_size = 81  # REPLACE this with the actual vector length expected by the repo
num_samples = 5514 + 23  # Number of "days" or samples to simulate

# Create a directory for the data if it doesn't exist
output_dir = "surfacepath"  # The docs call this "surfacepath" [2]
os.makedirs(output_dir, exist_ok=True)

# Generate random data
# Implied volatility is usually between 0.0 and 1.0 (e.g., 20% = 0.2)
# We generate positive random numbers to simulate IVs.
data = np.random.uniform(low=0.1, high=0.5, size=(num_samples, flattened_size))

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
# The sources mention the file should be named 'surfaces_transform.csv' [2]
output_file = os.path.join(output_dir, "surfaces_transform.csv")
df.to_csv(output_file, index=False, header=False)  # Check if the repo expects a header

print(f"Generated {output_file} with shape {df.shape}")
