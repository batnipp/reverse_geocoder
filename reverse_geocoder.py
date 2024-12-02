import streamlit as st
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import concurrent.futures
from functools import partial
import sqlite3
import time
from io import BytesIO
import folium
from streamlit_folium import st_folium

# Initialize cache database
def init_cache():
    conn = sqlite3.connect('geocoding_cache.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS geocoding_cache
        (address TEXT PRIMARY KEY, latitude REAL, longitude REAL)
    ''')
    conn.commit()
    conn.close()

@st.cache_data
def get_cached_coordinates(address):
    conn = sqlite3.connect('geocoding_cache.db')
    cursor = conn.cursor()
    cursor.execute('SELECT latitude, longitude FROM geocoding_cache WHERE address = ?', (address,))
    result = cursor.fetchone()
    conn.close()
    return result if result else None

def save_to_cache(address, lat, lon):
    if not (address and lat and lon):
        return
    conn = sqlite3.connect('geocoding_cache.db')
    cursor = conn.cursor()
    cursor.execute('INSERT OR REPLACE INTO geocoding_cache (address, latitude, longitude) VALUES (?, ?, ?)',
                  (address, lat, lon))
    conn.commit()
    conn.close()

def load_data(uploaded_file):
    """Load data from uploaded CSV or XLSX file."""
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:  # xlsx
        df = pd.read_excel(uploaded_file)
    return df

def geocode_address(address, geolocator):
    """Geocode a single address with caching"""
    # Check cache first
    cached_result = get_cached_coordinates(address)
    if cached_result:
        return cached_result
    
    # If not in cache, geocode
    max_retries = 3
    for attempt in range(max_retries):
        try:
            location = geolocator.geocode(address)
            if location:
                lat, lon = location.latitude, location.longitude
                # Save to cache
                save_to_cache(address, lat, lon)
                return lat, lon
            return None, None
        except GeocoderTimedOut:
            if attempt == max_retries - 1:
                return None, None
            time.sleep(1)
        except Exception:
            return None, None

def process_dataframe(df, progress_bar):
    """Process the dataframe using parallel execution and caching"""
    init_cache()
    geolocator = Nominatim(user_agent="csv_reverse_geocoder")
    
    # Create a partial function with the geolocator
    geocode_partial = partial(geocode_address, geolocator=geolocator)
    
    # Process addresses in parallel
    total_rows = len(df)
    processed_rows = 0
    
    df['latitude'] = None
    df['longitude'] = None
    
    # Initialize timing metrics
    start_time = time.time()
    time_per_batch = []
    
    # Process in smaller batches to update progress bar
    batch_size = 10
    status_placeholder = st.empty()
    
    for i in range(0, total_rows, batch_size):
        batch_start_time = time.time()
        batch = df['address'].iloc[i:i+batch_size].tolist()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(geocode_partial, batch))
            
        for j, (lat, lon) in enumerate(results):
            idx = i + j
            if idx < total_rows:
                df.at[idx, 'latitude'] = lat
                df.at[idx, 'longitude'] = lon
        
        # Update progress and timing estimates
        processed_rows += len(batch)
        progress = processed_rows / total_rows
        progress_bar.progress(min(progress, 1.0))
        
        # Calculate time estimates
        batch_time = time.time() - batch_start_time
        time_per_batch.append(batch_time)
        avg_time_per_batch = sum(time_per_batch) / len(time_per_batch)
        remaining_batches = (total_rows - processed_rows) / batch_size
        estimated_remaining_seconds = remaining_batches * avg_time_per_batch
        estimated_remaining_minutes = estimated_remaining_seconds / 60  # Convert to minutes
        elapsed_time = time.time() - start_time
        
        # Update status message with time in minutes
        status_placeholder.write(
            f"Processing addresses... This may take a few minutes.\n"
            f"Processed: {processed_rows}/{total_rows} addresses "
            f"({(progress*100):.1f}%)\n"
            f"Elapsed time: {elapsed_time:.1f} seconds\n"
            f"Estimated time remaining: {estimated_remaining_minutes:.1f} minutes"
        )
    
    return df

def create_map(df):
    """Create a Folium map with markers for each geocoded location."""
    # Filter out rows where geocoding failed
    df_clean = df.dropna(subset=['latitude', 'longitude'])
    
    if len(df_clean) > 0:
        # Calculate the center of the map
        center_lat = df_clean['latitude'].mean()
        center_lon = df_clean['longitude'].mean()
        
        # Create the map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=4)
        
        # Add markers
        for idx, row in df_clean.iterrows():
            folium.Marker(
                [row['latitude'], row['longitude']],
                popup=row['address']
            ).add_to(m)
        
        return m
    return None

def main():
    st.set_page_config(page_title="CSV Reverse Geocoder", layout="wide")
    
    # Title and description
    st.title("CSV Reverse Geocoder")
    st.markdown("""
    Upload a CSV or XLSX file containing addresses to get their latitude and longitude coordinates.
    The file must have an 'address' column.
    """)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Drop your CSV or XLSX file here",
        type=['csv', 'xlsx'],
        help="Make sure your file has an 'address' column"
    )
    
    if uploaded_file:
        try:
            # Load the data
            df = load_data(uploaded_file)
            
            # Validate the presence of 'address' column
            if 'address' not in df.columns:
                st.error("Error: The uploaded file must contain an 'address' column.")
                return
            
            # Process button
            if st.button("Process Addresses"):
                # Create a progress bar
                progress_bar = st.progress(0)
                st.write("Processing addresses... This may take a few minutes.")
                
                # Process the dataframe
                processed_df = process_dataframe(df, progress_bar)
                
                # Display the results
                st.subheader("Preview of Processed Data")
                st.dataframe(processed_df.head())
                
                # Create and display the map
                st.subheader("Location Map")
                map_obj = create_map(processed_df)
                if map_obj:
                    st_folium(map_obj, width=800, height=600)
                else:
                    st.warning("No valid coordinates to display on the map.")
                
                # Download button
                output = BytesIO()
                processed_df.to_csv(output, index=False)
                output.seek(0)
                st.download_button(
                    label="Download Processed CSV",
                    data=output,
                    file_name="geocoded_addresses.csv",
                    mime="text/csv"
                )
                
                # Display statistics
                st.subheader("Processing Statistics")
                total_rows = len(processed_df)
                successful_geocodes = processed_df['latitude'].notna().sum()
                st.write(f"Total addresses processed: {total_rows}")
                st.write(f"Successfully geocoded: {successful_geocodes}")
                st.write(f"Failed to geocode: {total_rows - successful_geocodes}")
        
        except Exception as e:
            st.error(f"An error occurred while processing the file: {str(e)}")
            return

if __name__ == "__main__":
    main()