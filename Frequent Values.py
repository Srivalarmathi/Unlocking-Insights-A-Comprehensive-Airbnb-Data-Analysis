# Python Script for finding most frequent values in the given csv file
import pandas as pd

# Load the CSV file into a DataFrame
file_path = r"C:\Users\valarsri\Downloads\processed_airbnb.csv"
df = pd.read_csv(file_path)


# Function to find most frequent values
def most_frequent_values(df):
    most_frequent = {}

    for column in df.columns:
        most_frequent_value = df[column].mode().iloc[0] if not df[column].mode().empty else None
        most_frequent[column] = most_frequent_value

    return most_frequent


# Calculate and display the most frequent values for each column
most_frequent = most_frequent_values(df)
for column, value in most_frequent.items():
    print(f"Column: {column}, Most Frequent Value: {value}")

# Optionally, you can save the results to a CSV file
output_path = r"C:\Users\valarsri\Downloads\Most_Frequent_values_Output.csv"
pd.DataFrame(most_frequent.items(), columns=['Column', 'Most Frequent Value']).to_csv(output_path, index=False)

