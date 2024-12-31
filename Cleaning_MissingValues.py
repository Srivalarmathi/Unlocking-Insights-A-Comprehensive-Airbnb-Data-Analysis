#Python Script for finding the missing values
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
file_path = r"C:\Users\valarsri\Downloads\airbnb - airbnb.csv"
df = pd.read_csv(file_path)

# Find the data types of each column
data_types = df.dtypes
print("Data Types:\n", data_types)

# Find the number of unique values in each column
unique_values = df.nunique()
print("\nUnique Values:\n", unique_values)

# Find the number of missing values in each column
missing_values = df.isnull().sum()
print("\nMissing Values:\n", missing_values)


# Create a heatmap to visualize missing values
plt.figure(figsize=(12, 8))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.title('Heatmap of Missing Values')
plt.xlabel('Columns')
plt.ylabel('Rows')
plt.show()
