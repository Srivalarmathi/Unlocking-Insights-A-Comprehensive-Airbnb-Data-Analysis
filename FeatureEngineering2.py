import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
file_path = r"C:\Users\valarsri\Downloads\processed_airbnb.csv"
df = pd.read_csv(file_path)

# Check for Infinity and NaN values
print("Checking for Infinity values...")
df.replace([np.inf, -np.inf], np.nan, inplace=True)

print("Checking for NaN values...")
df.fillna(0, inplace=True)

# Ensure price column is treated as string and convert to numeric
df['price'] = df['price'].astype(str).str.replace('$', '').str.replace(',', '').astype(float)

# Create New Features

# Review Frequency
df['review_frequency'] = df['number_of_reviews'] / df['availability_365']
df['review_frequency'] = df['review_frequency'].replace([np.inf, -np.inf], 0)  # Handle division by zero

# Price-to-Availability Ratio
df['price_to_availability'] = df['price'] / df['availability_365']
df['price_to_availability'] = df['price_to_availability'].replace([np.inf, -np.inf], 0)  # Handle division by zero

# Identify numerical columns for scaling
numerical_cols = ['reviews_per_month', 'rating', 'number_of_stays', '5_stars', 'review_frequency', 'price_to_availability']

# Apply MinMax Scaling
scaler = MinMaxScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Save the DataFrame with New Features
output_path = r"C:\Users\valarsri\Downloads\feature_engineered_airbnb.csv"
df.to_csv(output_path, index=False)
print(f"Feature engineering completed and saved to {output_path}")
