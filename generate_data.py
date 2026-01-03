import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from VolGAN import get_data_yahoo

# CONFIGURATION
# Sources say this file uses a fixed (delta, tau) grid [2]
output_dir = "datapath"
os.makedirs(output_dir, exist_ok=True)

# 1. Define the Grid
# OptionMetrics Standard "Surface" usually provides standard deltas and expirations
deltas = [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
]  # standard deltas
maturities_days = [
    30,
    60,
    91,
    122,
    152,
    182,
    273,
    365,
    547,
    730,
]  # standard days to expire

# 2. Simulation parameters
# start_date = datetime(2020, 1, 1)
# num_days = 100  # Number of trading dates to simulate
# startdate = datetime(1999, 12, 31)
# enddate = datetime(2000, 1, 1)
startdate = datetime(2000, 1, 3)
enddate = datetime(2022, 1, 1)

y_symbols = ["SPY"]
SPY = get_data_yahoo(y_symbols, start=startdate, end=enddate)

records = []

# for dt in pd.bdate_range(start=startdate, end=enddate):
for dt in SPY.index:
    date_str = dt.strftime("%d-%m-%Y")

    # For every date, we have entries for each maturity (tau)
    for days in maturities_days:
        # Create a row dictionary
        row = {
            "date": date_str,
            "days": days,  # This matches the column you saw in SPXData
        }

        # Populate columns for each Delta (representing the grid in [2])
        # Note: If the code expects a "long" format (a 'delta' column),
        # you would restructure this. But OptionMetrics is often "wide".
        # We will generate "wide" columns for deltas here based on standard formats.
        for d in deltas:
            # Generate a dummy IV (e.g., 0.20 for 20%)
            # Adding some random skew based on delta
            dummy_vol = 0.2 + (0.5 - d / 100.0) ** 2 + np.random.normal(0, 0.01)

            # Column naming depends on the specific parser in SPXData,
            # but standard OptionMetrics might use 'IVCall_20', 'IVPut_20', etc.
            # We will use generic names; you might need to check the code for exact keys.
            row[f"IV_{d}"] = abs(dummy_vol)

        records.append(row)

# 3. Create DataFrame
df = pd.DataFrame(records)

# 4. Save
output_file = os.path.join(output_dir, "data.csv")
df.to_csv(output_file, index=False)

print(f"Generated {output_file} with shape {df.shape}")
print("Columns:", df.columns.tolist())
