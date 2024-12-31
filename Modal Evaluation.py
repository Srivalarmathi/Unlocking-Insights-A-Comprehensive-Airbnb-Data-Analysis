import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score
import numpy as np
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

# Select relevant features for regression
features = ['availability_365', 'reviews_per_month', 'rating', 'number_of_reviews', 'number_of_stays', '5_stars', 'review_frequency', 'price_to_availability']
X = df[features]
y = df['price']

# Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42)
}

# Evaluate models
results = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    # Convert regression output to binary classification for accuracy and F1 score
    y_pred_binary = y_pred > y.median()
    y_test_binary = y_test > y.median()
    accuracy = accuracy_score(y_test_binary, y_pred_binary)
    f1 = f1_score(y_test_binary, y_pred_binary)
    results[model_name] = {'RMSE': rmse, 'MAE': mae, 'Accuracy': accuracy, 'F1 Score': f1}

# Display results
results_df = pd.DataFrame(results).T
print(results_df)

# Plot the evaluation metrics
metrics = ['RMSE', 'MAE', 'Accuracy', 'F1 Score']
plt.figure(figsize=(15, 10))
for i, metric in enumerate(metrics, 1):
    plt.subplot(2, 2, i)
    sns.barplot(x=results_df.index, y=results_df[metric])
    plt.title(metric)
    plt.xticks(rotation=45)
    plt.ylabel(metric)
    plt.xlabel('Model')
plt.tight_layout()
plt.show()

# Select the best model
best_model_rmse = results_df.idxmin()['RMSE']
best_model_accuracy = results_df.idxmax()['Accuracy']
print(f"Best Model by RMSE: {best_model_rmse}")
print(f"Best Model by Accuracy: {best_model_accuracy}")
