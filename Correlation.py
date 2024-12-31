import pandas as pd

# Load the CSV file into a DataFrame
file_path = r"C:\Users\valarsri\Downloads\processed_airbnb.csv"
df = pd.read_csv(file_path)

# Select only the numeric columns from the DataFrame
numeric_df = df.select_dtypes(include='number')

# Calculate the correlation matrices using different methods
pearson_corr = numeric_df.corr(method='pearson')  # Pearson correlation
spearman_corr = numeric_df.corr(method='spearman')  # Spearman correlation
kendall_corr = numeric_df.corr(method='kendall')  # Kendall correlation

# Function to highlight highly correlated values in the correlation matrix
def highlight_highly_correlated(corr_matrix, threshold=0.8):
    return corr_matrix.map(lambda x: 'background-color: yellow' if abs(x) >= threshold else '')

# Apply the highlighting function to the correlation matrices
highlighted_pearson = pearson_corr.style.apply(highlight_highly_correlated, threshold=0.8, axis=None)
highlighted_spearman = spearman_corr.style.apply(highlight_highly_correlated, threshold=0.8, axis=None)
highlighted_kendall = kendall_corr.style.apply(highlight_highly_correlated, threshold=0.8, axis=None)

# Save the correlation matrices to CSV files
pearson_corr.to_csv(r"C:\Users\valarsri\Downloads\pearson_correlation.csv")
spearman_corr.to_csv(r"C:\Users\valarsri\Downloads\spearman_correlation.csv")
kendall_corr.to_csv(r"C:\Users\valarsri\Downloads\kendall_correlation.csv")

# Save the highlighted correlation matrices to HTML files
highlighted_pearson.to_html(r"C:\Users\valarsri\Downloads\highlighted_pearson_correlation.html")
highlighted_spearman.to_html(r"C:\Users\valarsri\Downloads\highlighted_spearman_correlation.html")
highlighted_kendall.to_html(r"C:\Users\valarsri\Downloads\highlighted_kendall_correlation.html")

# Print a success message
print("Correlation matrices and highlighted versions saved successfully.")
