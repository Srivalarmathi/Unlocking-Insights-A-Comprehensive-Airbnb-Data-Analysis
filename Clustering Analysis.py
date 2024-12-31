import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = r"C:\Users\valarsri\Downloads\processed_airbnb.csv"
df = pd.read_csv(file_path)

# Feature Engineering: Add review_frequency and price_to_availability
df['review_frequency'] = df['number_of_reviews'] / df['availability_365']
df['price_to_availability'] = df['price'] / df['availability_365']
df['review_frequency'] = df['review_frequency'].replace([float('inf'), -float('inf')], 0)
df['price_to_availability'] = df['price_to_availability'].replace([float('inf'), -float('inf')], 0)

# Select relevant features for clustering
features = ['price', 'availability_365', 'reviews_per_month', 'rating', 'number_of_reviews', 'review_frequency', 'price_to_availability']
X = df[features]

# Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Determine the optimal number of clusters using the Elbow Method
wcss = []  # Within-cluster sum of squares
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.show()

# Based on the Elbow Method, choose the optimal number of clusters (e.g., k=4)
optimal_clusters = 4
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
kmeans.fit(X_scaled)
df['Cluster'] = kmeans.labels_

# Visualize the Clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='price', y='availability_365', hue='Cluster', data=df, palette='viridis', s=100)
plt.xlabel('Price')
plt.ylabel('Availability (365 Days)')
plt.title('Clusters of Listings Based on Price and Availability')
plt.legend(title='Cluster')
plt.show()

# Summarize the Cluster Characteristics with Numeric Columns Only
numeric_cols = ['price', 'availability_365', 'reviews_per_month', 'rating', 'number_of_reviews', 'review_frequency', 'price_to_availability']
cluster_summary = df.groupby('Cluster')[numeric_cols].mean()
print(cluster_summary)
