import matplotlib.pyplot as plt
import numpy as np
from sktime.utils.load_data import load_from_tsfile_to_dataframe

# 1. Point this to one of your training files
file_path = "data/mediapipe/BS/103007/TRAIN_default_X.ts"

# 2. Load the data
print(f"Loading data from {file_path}...")
X, y = load_from_tsfile_to_dataframe(file_path)

# 3. Select the very first video in the dataset (Row 0)
video_data = X.iloc[0]
label = y[0]

# 4. Plot the data
plt.figure(figsize=(12, 6))

for i in range(24, 28):
    # Grab the raw data array
    raw_data = video_data[i]

    # APPLY THE Z-SCORE NORMALIZATION! (The math from your paper)
    z_normalized_data = (raw_data - raw_data.mean()) / raw_data.std()

    # Plot the normalized data instead of the raw data
    plt.plot(z_normalized_data, label=f'Feature {i}')

# Update the titles to match your new paper narrative
plt.title(f'Pose Estimation Keypoints Over Time (Z-Normalized) (Label: {label})')
plt.xlabel('Frame / Time Index')
plt.ylabel('Normalized Coordinate Value (Z-Score)')
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the image
plt.savefig("timeseries_normalized_example.png", dpi=300)
print("Saved graph as timeseries_normalized_example.png!")
plt.show()