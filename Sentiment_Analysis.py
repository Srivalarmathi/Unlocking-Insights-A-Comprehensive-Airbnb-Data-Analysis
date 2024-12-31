import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob

# Load the dataset
file_path = r"C:\Users\valarsri\Downloads\processed_airbnb.csv"
df = pd.read_csv(file_path)

# Convert price to numeric
df['price'] = df['price'].astype(str).str.replace('$', '').str.replace(',', '').astype(float)

# Perform Sentiment Analysis on Ratings
def categorize_sentiment(rating):
    if rating > 4:
        return 'positive'
    elif 3 <= rating <= 4:
        return 'neutral'
    else:
        return 'negative'

df['sentiment'] = df['rating'].apply(categorize_sentiment)

# Print the distribution of sentiment categories
print(df['sentiment'].value_counts())

# Visualize the sentiment distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='sentiment', data=df, palette='viridis', hue='sentiment', legend=False)
for p in plt.gca().patches:
    plt.gca().annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='baseline', fontsize=12, color='white')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Distribution of Sentiment Categories')
plt.show()

# One-Hot Encode the sentiment categories for correlation analysis
df = pd.get_dummies(df, columns=['sentiment'], drop_first=True)

# Verify the columns
print(df.columns)

# Include only numeric columns for correlation matrix
numeric_cols = ['price', 'reviews_per_month', 'rating', 'number_of_reviews', 'availability_365', 'number_of_stays', '5_stars', 'sentiment_positive']

# Calculate the correlation matrix
correlation_matrix = df[numeric_cols].corr()

# Extract relevant correlations with rating
relevant_correlations = correlation_matrix[['rating', 'sentiment_positive']].sort_values(by='rating', ascending=False)

# Print relevant correlations
print(relevant_correlations)

# Visualize the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, annot_kws={"size": 12, "color": "white"})
plt.title('Correlation Matrix')
plt.show()

# Visualize Relationships
# Box plot for reviews_per_month vs. rating by sentiment
plt.figure(figsize=(12, 8))
sns.boxplot(x='sentiment_positive', y='reviews_per_month', data=df, palette='viridis')
plt.xlabel('Sentiment (Positive)')
plt.ylabel('Reviews per Month')
plt.title('Reviews per Month vs. Rating by Sentiment')
plt.show()

# Enhanced scatter plots for numeric features vs. rating by sentiment
numeric_features = ['reviews_per_month', 'rating', 'number_of_reviews', 'number_of_stays', '5_stars']

for feature in numeric_features:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df[feature], y=df['rating'], hue=df['sentiment_positive'], palette='viridis', s=100, edgecolor='w', linewidth=0.5)
    plt.xlabel(feature)
    plt.ylabel('Rating')
    plt.title(f'{feature} vs. Rating by Sentiment')
    plt.legend(title='Sentiment', loc='best', title_fontsize='13', fontsize='10')
    plt.show()
