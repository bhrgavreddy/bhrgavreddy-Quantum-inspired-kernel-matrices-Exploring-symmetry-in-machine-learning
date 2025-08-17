import numpy as np
import pandas as pd

n_samples = 1000  # Set the number of samples

# Generate the first feature x0
x0_class_0 = np.random.uniform(0, np.pi / 3, n_samples // 2)
x0_class_1 = np.random.uniform(2 * np.pi / 3, np.pi, n_samples // 2)

# Combine the x0 feature for both classes
x0 = np.concatenate((x0_class_0, x0_class_1))

# Generate the other features x1 and x2
x1 = np.zeros(n_samples)
x2 = np.random.uniform(0+0.00001, np.pi-0.00001, n_samples)

# Generate the class labels
y = np.concatenate((np.zeros(n_samples // 2), np.ones(n_samples // 2)))

# Shuffle the dataset
indices = np.arange(n_samples)
np.random.shuffle(indices)

# Create the final dataset
x0 = x0[indices]
x1 = x1[indices]
x2 = x2[indices]
y = y[indices]

# Create a DataFrame
df = pd.DataFrame({
    'x0': x0,
    'x1': x1,
    'x2': x2,
    'Class': y.astype(int)
})

# Save to CSV
file_name = f"synthetic_data_{n_samples}.csv"
df.to_csv(file_name, index=False)
print(f"Dataset saved to {file_name}")
