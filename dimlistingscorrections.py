import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Date
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Load the dataset
file_path = r"C:\Users\valarsri\Downloads\dim_listings.csv"
df = pd.read_csv(file_path)

# Preprocess the data
df['listing_added'] = pd.to_datetime(df['listing_added'])

# Remove duplicates
df = df.drop_duplicates(subset='listing_id')

# Check for null values
print("Null values in the dataset:")
print(df.isnull().sum())

# Fill null values with the mean for numerical columns
df['listing_id'].fillna(df['listing_id'].mean(), inplace=True)

# For non-numerical columns, fill null values with mode or appropriate value
df['name'].fillna(df['name'].mode()[0], inplace=True)
df['coordinates'].fillna('(0, 0)', inplace=True)  # Example of filling coordinates with a placeholder
df['listing_added'].fillna(df['listing_added'].mean(), inplace=True)

# Verify that there are no null values
print("Null values after filling:")
print(df.isnull().sum())

# Create an engine to connect to the database
engine = create_engine('sqlite:///airbnb.db')  # You can replace 'sqlite:///airbnb.db' with your database URI

# Create a base class for declarative class definitions
Base = declarative_base()

# Define the Dimension Table
class DimListings(Base):
    __tablename__ = 'dim_listings'
    listing_id = Column(Integer, primary_key=True)
    name = Column(String)
    coordinates = Column(String)
    listing_added = Column(Date)

# Create the table
Base.metadata.create_all(engine)

# Create a session
Session = sessionmaker(bind=engine)
session = Session()

# Populate the DimListings table
listings_data = df[['listing_id', 'name', 'coordinates', 'listing_added']]
for row in listings_data.itertuples(index=False):
    session.add(DimListings(
        listing_id=row.listing_id,
        name=row.name,
        coordinates=row.coordinates,
        listing_added=row.listing_added
    ))

session.commit()

# Verify the table contents
dim_listings_df = pd.read_sql_table('dim_listings', engine)
print("DimListings table preview:")
print(dim_listings_df.head())

# Save the DimListings table to a CSV file
dim_listings_df.to_csv(r'C:\Users\valarsri\Downloads\dim_listings.csv', index=False)

print("DimListings table created and saved to CSV file successfully!")
