import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob

# Load and preprocess dataset
file_path = r"C:\Users\valarsri\Downloads\processed_airbnb.csv"
df = pd.read_csv(file_path)

# Convert price to numeric
df['price'] = df['price'].astype(str).str.replace('$', '').str.replace(',', '').astype(float)

# Feature Engineering
df['review_frequency'] = df['number_of_reviews'] / df['availability_365']
df['price_to_availability'] = df['price'] / df['availability_365']
df.replace([np.inf, -np.inf], 0, inplace=True)
df.fillna(0, inplace=True)

# One-Hot Encode Categorical Variables
df = pd.get_dummies(df, columns=['room_type'], drop_first=True)

# Identify numerical columns for scaling
numerical_cols = ['reviews_per_month', 'rating', 'number_of_stays', '5_stars', 'review_frequency', 'price_to_availability']

# Apply MinMax Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Task 1: Price Prediction Model
X_price = df[['availability_365', 'rating'] + [col for col in df.columns if 'room_type_' in col]]
y_price = df['price']

X_train, X_test, y_train, y_test = train_test_split(X_price, y_price, test_size=0.2, random_state=42)
gb_model = GradientBoostingRegressor(random_state=42)
param_grid = {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5, 7]}
grid_search = GridSearchCV(estimator=gb_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
best_gb_model = grid_search.best_estimator_
y_pred = best_gb_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Price Prediction Model - Mean Squared Error: {mse}, Mean Absolute Error: {mae}, R-squared: {r2}')

# Plot Actual vs. Predicted Prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs. Predicted Prices (Gradient Boosting)')
plt.show()

# Plot Residuals
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Price')
plt.ylabel('Residuals')
plt.title('Residual Plot (Gradient Boosting)')
plt.show()

# Plot Histogram of Residuals
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.xlabel('Residuals')
plt.title('Distribution of Residuals (Gradient Boosting)')
plt.show()

# Task 2: Availability Prediction Model
X_availability = df[['price', 'rating'] + [col for col in df.columns if 'room_type_' in col]]
y_availability = df['availability_365'] > 180  # Example threshold for classification

X_train, X_test, y_train, y_test = train_test_split(X_availability, y_availability, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(random_state=42)
param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
best_rf_model = grid_search.best_estimator_
y_pred = best_rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f'Availability Prediction Model - Accuracy: {accuracy}, F1 Score: {f1}')

# Task 3: Sentiment Classification Model
# Assume sentiment analysis has already been done and sentiment labels are available
df['sentiment_label'] = df['rating'].apply(lambda x: 'positive' if x > 4 else 'neutral' if 3 <= x <= 4 else 'negative')

# Convert sentiment labels to numerical categories
df['sentiment'] = df['sentiment_label'].map({'negative': 0, 'neutral': 1, 'positive': 2})
X_sentiment = df[['review_frequency', 'price_to_availability', 'reviews_per_month', 'rating', 'number_of_reviews', 'number_of_stays']]
y_sentiment = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X_sentiment, y_sentiment, test_size=0.2, random_state=42)
logreg_model = LogisticRegression(random_state=42)
logreg_model.fit(X_train, y_train)
y_pred = logreg_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
print(f'Sentiment Classification Model - Accuracy: {accuracy}, F1 Score: {f1}')

# Task 4: Clustering Analysis
X_clustering = df[['price', 'availability_365', 'rating']]
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_clustering)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.show()

kmeans = KMeans(n_clusters=2, random_state=42)
df['cluster'] = kmeans.fit_predict(X_clustering)
plt.scatter(X_clustering['price'], X_clustering['availability_365'], c=df['cluster'], cmap='viridis')
plt.xlabel('Price')
plt.ylabel('Availability 365')
plt.title('K-Means Clustering with 2 Clusters')
plt.show()
