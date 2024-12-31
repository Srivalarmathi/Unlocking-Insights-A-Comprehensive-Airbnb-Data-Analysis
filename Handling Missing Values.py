import pandas as pd

# Load the dataset
file_path = r"C:\Users\valarsri\Downloads\airbnb - airbnb.csv"
df = pd.read_csv(file_path)

# Handling Missing Values

# 1. Handling categorical columns with mode imputation
df['name'] = df['name'].fillna(df['name'].mode()[0])
df['host_name'] = df['host_name'].fillna(df['host_name'].mode()[0])

# 2. Handling numerical columns with mean/median imputation
df['price'] = df['price'].str.replace('$', '').str.replace(',', '').astype(float)  # Convert price to numeric if needed
df['price'] = df['price'].fillna(df['price'].mean())  # Replace with mean

df['reviews_per_month'] = df['reviews_per_month'].fillna(df['reviews_per_month'].median())  # Replace with median
df['rating'] = df['rating'].fillna(df['rating'].mean())  # Replace with mean
df['number_of_stays'] = df['number_of_stays'].fillna(df['number_of_stays'].median())  # Replace with median
df['5_stars'] = df['5_stars'].fillna(df['5_stars'].mean())  # Replace with mean

# 3. Handling date columns with forward fill
df['last_review'] = pd.to_datetime(df['last_review'])
df['last_review'] = df['last_review'].ffill()  # Forward fill

# Check for remaining missing values
missing_values_after = df.isnull().sum()
print("Remaining Missing Values After Handling:\n", missing_values_after)

# Save the processed DataFrame
file_pathop = r"C:\Users\valarsri\Downloads\processed_airbnb.csv"
df.to_csv(file_pathop, index=False)

print("Missing values handled and saved to 'processed_airbnb.csv'.")
