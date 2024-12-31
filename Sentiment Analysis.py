import pandas as pd
from textblob import TextBlob

# Load the CSV file into a DataFrame
file_path = r"C:\Users\valarsri\Downloads\processed_airbnb.csv"
df = pd.read_csv(file_path)

# Function to calculate sentiment polarity
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

# Assuming your DataFrame has a column named 'review_text' containing the reviews
df['sentiment'] = df['review_text'].apply(get_sentiment)

# Save the DataFrame with the sentiment scores
output_path = r"C:\Users\valarsri\Downloads\sentiment_analysis_output.csv"
df.to_csv(output_path, index=False)

print(f"Sentiment analysis completed and saved to {output_path}")
