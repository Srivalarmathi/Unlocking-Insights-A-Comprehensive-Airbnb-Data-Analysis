import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = r"C:\Users\valarsri\Downloads\processed_airbnb.csv"
df = pd.read_csv(file_path)

# Feature Engineering: Add review_frequency and price_to_availability
df['review_frequency'] = df['number_of_reviews'] / df['availability_365']
df['price_to_availability'] = df['price'] / df['availability_365']
df['review_frequency'] = df['review_frequency'].replace([float('inf'), -float('inf')], 0)
df['price_to_availability'] = df['price_to_availability'].replace([float('inf'), -float('inf')], 0)

# Binarize availability: threshold example - available more than 180 days per year
df['availability_binary'] = df['availability_365'] > 180

# Select the relevant features
features = ['reviews_per_month', 'rating', 'number_of_reviews', 'number_of_stays', '5_stars', 'review_frequency', 'price_to_availability']

# Prepare Data for Modeling
X = df[features]
y = df['availability_binary']

# Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42, n_estimators=100)
rf_classifier.fit(X_train, y_train)

# Make Predictions
y_pred = rf_classifier.predict(X_test)

# Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

# Visualize the Confusion Matrix
plt.figure(figsize=(10, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
