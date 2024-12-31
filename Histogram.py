#Python Script for Histogram Graph
import pandas as pd
import matplotlib.pyplot as plt


# Load the CSV file into a DataFrame
file_path = r"C:\Users\valarsri\Downloads\processed_airbnb.csv"
df = pd.read_csv(file_path)

# Plot histograms for all numeric columns
df.hist(figsize=(10, 8), bins=30, edgecolor='black')

# Set overall title for the histograms
plt.suptitle('Histograms of Numeric Columns', fontsize=16)

# Display the plot
plt.show()
