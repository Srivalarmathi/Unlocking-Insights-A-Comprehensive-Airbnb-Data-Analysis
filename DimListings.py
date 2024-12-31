import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Date
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Load the dataset
file_path = r"C:\Users\valarsri\Downloads\processed_airbnb.csv"
df = pd.read_csv(file_path)

# Preprocess the data
df['listing_added'] = pd.to_datetime(df['listing_added'])

# Remove duplicates
df = df.drop_duplicates(subset='listing_id')

# Verify data
print("Data preview:")
print(df.head())

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
try:
    listings_data = df[['listing_id', 'name', 'coordinates', 'listing_added']]
    for row in listings_data.itertuples(index=False):
        session.add(DimListings(
            listing_id=row.listing_id,
            name=row.name,
            coordinates=row.coordinates,
            listing_added=row.listing_added
        ))
    session.commit()
    print("Data successfully added to the DimListings table.")
except Exception as e:
    session.rollback()
    print(f"Error occurred: {e}")

# Verify the table contents
dim_listings_df = pd.read_sql_table('dim_listings', engine)
print("DimListings table preview:")
print(dim_listings_df.head())

# Save the DimListings table to a CSV file
dim_listings_df.to_csv(r'C:\Users\valarsri\Downloads\dim_listings.csv', index=False)

print("DimListings table created and saved to CSV file successfully!")
