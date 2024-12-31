# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno

# Load the CSV file into a DataFrame
file_path = r"C:\Users\valarsri\Downloads\processed_airbnb.csv"
df = pd.read_csv(file_path)

# Calculate the count of missing values in each column
missing_values_count = df.isnull().sum()

# Print the count of missing values for each column
print("Missing Values Count:\n", missing_values_count)

# Plot a heatmap to visualize the locations of missing values
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.title('Heatmap of Missing Values')
plt.show()

# Plot a matrix to visualize the missing values using missingno
msno.matrix(df, figsize=(12, 6))
plt.title('Missing Values Matrix')
plt.show()

# Plot a dendrogram to visualize the hierarchical clustering of the missing values
msno.dendrogram(df)
plt.title('Dendrogram of Missing Values')
plt.show()

# Save the count of missing values to a CSV file
missing_values_count.to_csv(r"C:\Users\valarsri\Downloads\missing_value_matrix_output.csv", header=["Missing Values"])

# Print a success message
print("Missing values analysis completed and visualized successfully.")
