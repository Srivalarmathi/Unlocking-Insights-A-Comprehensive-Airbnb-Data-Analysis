import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, Date, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Load the dataset
file_path = r"C:\Users\valarsri\Downloads\processed_airbnb.csv"
df = pd.read_csv(file_path)

# Change column name from 5_stars to five_stars
df.rename(columns={'5_stars': 'five_stars'}, inplace=True)

# Preprocess the data
df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)
df['last_review'] = pd.to_datetime(df['last_review'])
df['listing_added'] = pd.to_datetime(df['listing_added'])
df[['latitude', 'longitude']] = df['coordinates'].str.extract(r'\(([^,]+),\s([^)]+)\)').astype(float)
df['review_frequency'] = df['number_of_reviews'] / df['availability_365']
df['price_to_availability'] = df['price'] / df['availability_365']
df['review_frequency'] = df['review_frequency'].replace([float('inf'), -float('inf')], 0)
df['price_to_availability'] = df['price_to_availability'].replace([float('inf'), -float('inf')], 0)

# Ensure unique listing_id
df = df.drop_duplicates(subset='listing_id')

# Create an engine to connect to the database
engine = create_engine('sqlite:///airbnb.db')  # You can replace 'sqlite:///airbnb.db' with your database URI

# Create a base class for declarative class definitions
Base = declarative_base()

# Define the Dimension Tables
class DimListings(Base):
    __tablename__ = 'dim_listings'
    listing_id = Column(Integer, primary_key=True)
    name = Column(String)
    coordinates = Column(String)
    listing_added = Column(Date)

class DimHosts(Base):
    __tablename__ = 'dim_hosts'
    host_id = Column(Integer, primary_key=True)
    host_name = Column(String)

class DimNeighborhoods(Base):
    __tablename__ = 'dim_neighborhoods'
    neighbourhood_id = Column(Integer, primary_key=True)
    neighbourhood_full = Column(String)

class DimRoomTypes(Base):
    __tablename__ = 'dim_room_types'
    room_type_id = Column(Integer, primary_key=True)
    room_type = Column(String)

class DimDates(Base):
    __tablename__ = 'dim_dates'
    date_id = Column(Integer, primary_key=True)
    date = Column(Date, unique=True)

class DimLocations(Base):
    __tablename__ = 'dim_locations'
    location_id = Column(Integer, primary_key=True)
    latitude = Column(Float)
    longitude = Column(Float)

# Define the Fact Table
class FactListings(Base):
    __tablename__ = 'fact_listings'
    listing_id = Column(Integer, primary_key=True)
    price = Column(Float)
    number_of_reviews = Column(Integer)
    reviews_per_month = Column(Float)
    availability_365 = Column(Integer)
    rating = Column(Float)
    number_of_stays = Column(Float)
    five_stars = Column(Float)
    host_id = Column(Integer, ForeignKey('dim_hosts.host_id'))
    neighbourhood_id = Column(Integer, ForeignKey('dim_neighborhoods.neighbourhood_id'))
    room_type_id = Column(Integer, ForeignKey('dim_room_types.room_type_id'))
    last_review_date_id = Column(Integer, ForeignKey('dim_dates.date_id'))
    listing_added_date_id = Column(Integer, ForeignKey('dim_dates.date_id'))
    location_id = Column(Integer, ForeignKey('dim_locations.location_id'))

# Create all tables
Base.metadata.create_all(engine)

# Create a session
Session = sessionmaker(bind=engine)
session = Session()

# Populate dimension tables with no autoflush context
neighborhoods = df['neighbourhood_full'].unique()
hosts = df[['host_id', 'host_name']].drop_duplicates()  # Ensure unique hosts
room_types = df['room_type'].unique()
dates = pd.to_datetime(pd.concat([df['last_review'], df['listing_added']])).dropna().unique()

with session.no_autoflush:
    # Insert data into DimNeighborhoods
    for i, neighborhood in enumerate(neighborhoods, start=1):
        if not session.query(DimNeighborhoods).filter_by(neighbourhood_full=neighborhood).first():
            session.add(DimNeighborhoods(neighbourhood_id=i, neighbourhood_full=neighborhood))

    # Insert data into DimHosts
    for host_id, host_name in hosts.itertuples(index=False):
        if not session.query(DimHosts).filter_by(host_id=host_id).first():
            session.add(DimHosts(host_id=host_id, host_name=host_name))

    # Insert data into DimRoomTypes
    for i, room_type in enumerate(room_types, start=1):
        if not session.query(DimRoomTypes).filter_by(room_type=room_type).first():
            session.add(DimRoomTypes(room_type_id=i, room_type=room_type))

    # Insert data into DimDates
    for date in dates:
        if not session.query(DimDates).filter_by(date=date).first():
            session.add(DimDates(date_id=date.toordinal(), date=date))

    # Insert data into DimLocations
    locations = df[['latitude', 'longitude']].drop_duplicates()
    for i, (lat, lon) in enumerate(locations.itertuples(index=False), start=1):
        if not session.query(DimLocations).filter_by(latitude=lat, longitude=lon).first():
            session.add(DimLocations(location_id=i, latitude=lat, longitude=lon))

session.commit()

# Populate fact table
fact_listings_data = []
for row in df.itertuples(index=False):
    location_id = session.query(DimLocations.location_id).filter_by(latitude=row.latitude, longitude=row.longitude).first()[0]
    neighbourhood_id = session.query(DimNeighborhoods.neighbourhood_id).filter_by(neighbourhood_full=row.neighbourhood_full).first()[0]
    room_type_id = session.query(DimRoomTypes.room_type_id).filter_by(room_type=row.room_type).first()[0]
    last_review_date_id = session.query(DimDates.date_id).filter_by(date=row.last_review).first()[0]
    listing_added_date_id = session.query(DimDates.date_id).filter_by(date=row.listing_added).first()[0]

    # Ensure unique listing_id
    if not session.query(FactListings).filter_by(listing_id=row.listing_id).first():
        fact_listings_data.append({
            'listing_id': row.listing_id,
            'price': row.price,
            'number_of_reviews': row.number_of_reviews,
            'reviews_per_month': row.reviews_per_month,
            'availability_365': row.availability_365,
            'rating': row.rating,
            'number_of_stays': row.number_of_stays,
            'five_stars': row.five_stars,
            'host_id': row.host_id,
            'neighbourhood_id': neighbourhood_id,
            'room_type_id': room_type_id,
            'last_review_date_id': last_review_date_id,
            'listing_added_date_id': listing_added_date_id,
            'location_id': location_id
        })

session.bulk_save_objects([FactListings(**data) for data in fact_listings_data])
session.commit()

print("Dimension tables and fact table created successfully from the CSV file!")

# Save the dimension and fact tables to CSV files
dim_listings_df = pd.read_sql_table('dim_listings', engine)
dim_hosts_df = pd.read_sql_table('dim_hosts', engine)
dim_neighborhoods_df = pd.read_sql_table('dim_neighborhoods', engine)
dim_room_types_df = pd.read_sql_table('dim_room_types', engine)
dim_dates_df = pd.read_sql_table('dim_dates', engine)
dim_locations_df = pd.read_sql_table('dim_locations', engine)
fact_listings_df = pd.read_sql_table('fact_listings', engine)

dim_listings_df.to_csv(r'C:\Users\valarsri\Downloads\dim_listings.csv', index=False)
dim_hosts_df.to_csv(r'C:\Users\valarsri\Downloads\dim_hosts.csv', index=False)
dim_neighborhoods_df.to_csv(r'C:\Users\valarsri\Downloads\dim_neighborhoods.csv', index=False)
dim_room_types_df.to_csv(r'C:\Users\valarsri\Downloads\dim_room_types.csv', index=False)
dim_dates_df.to_csv(r'C:\Users\valarsri\Downloads\dim_dates.csv', index=False)
dim_locations_df.to_csv(r'C:\Users\valarsri\Downloads\dim_locations.csv', index=False)
fact_listings_df.to_csv(r'C:\Users\valarsri\Downloads\fact_listings.csv', index=False)

print("Dimension tables and fact table saved to CSV files!")
