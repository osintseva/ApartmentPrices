# master_run.py

import warnings
warnings.filterwarnings("ignore")  # Suppress all warnings

import pandas as pd
import os
from amenities_script import get_amenity_features
from tqdm import tqdm
import datetime

# Define AMENITIES here as well for master_run.py
AMENITIES = {
    'metro_stations': {'tags': {'subway': 'yes'}, 'radius': 3000},
    'sports_centers': {'tags': {'leisure': 'sports_centre'}, 'radius': 1000},
    'supermarkets': {'tags': {'shop': 'supermarket'}, 'radius': 300},
    'schools': {'tags': {'amenity': 'school'}, 'radius': 500},
    'kindergartens': {'tags': {'amenity': 'kindergarten'}, 'radius': 300},
    'cafes_restaurants': {'tags': {'amenity': ['cafe', 'restaurant']}, 'radius': 300},
    'public_transport': {'tags': {'highway': 'bus_stop'}, 'radius': 300},
    'woods_parks': {'tags': {'natural': 'wood', 'leisure': 'park'}, 'radius': 1000},
    'water_reservoirs': {'tags': {'natural': 'water'}, 'radius': 1000}
}

def process_database(input_database_path, output_database_path, max_entries=5, dump_interval=100):
    """
    Process the input CSV database to fetch amenity features for a limited number of entries
    and save the augmented data. Periodically dump the resulting database every 'dump_interval' entries.

    Parameters:
        input_database_path (str): Path to the input CSV database.
        output_database_path (str): Path to save the augmented RES_DATABASE CSV.
        max_entries (int): Maximum number of entries to process. Default is 5.
        dump_interval (int): Number of entries after which to dump the database. Default is 100.
    """
    # Check if input file exists
    if not os.path.isfile(input_database_path):
        return  # Silently exit if file does not exist

    # Determine the encoding of the file
    try:
        with open(input_database_path, encoding='utf-8') as f:
            encoding = 'utf-8'
    except UnicodeDecodeError:
        encoding = 'latin1'

    # Read the entire CSV file
    try:
        df = pd.read_csv(input_database_path, encoding=encoding)
    except Exception:
        return  # Silently exit if reading fails

    # Ensure Latitude and Longitude columns exist
    if 'Latitude' not in df.columns or 'Longitude' not in df.columns:
        return  # Silently exit if required columns are missing

    # Select the first 'max_entries' rows
    df_selected = df.head(max_entries).reset_index(drop=True)

    # Initialize list to store amenity features
    amenity_features_list = []

    # Initialize tqdm progress bar
    pbar = tqdm(total=max_entries, desc="Processing Entries")

    # Initialize counter for dump intervals
    dump_counter = 0

    # Ensure the DB_DUMPS directory exists
    dump_directory = "./DB_DUMPS"
    os.makedirs(dump_directory, exist_ok=True)

    # Iterate over each selected row
    for index, row in df_selected.iterrows():
        latitude = row['Latitude']
        longitude = row['Longitude']

        # Handle missing or invalid coordinates
        if pd.isnull(latitude) or pd.isnull(longitude):
            features = {f'{amenity}_count': 0 for amenity in AMENITIES.keys()}
            features.update({f'distance_to_nearest_{amenity}_m': 0 for amenity in AMENITIES.keys()})
        else:
            try:
                features = get_amenity_features(latitude, longitude)
            except Exception:
                # In case of any exception, set features to 0
                features = {f'{amenity}_count': 0 for amenity in AMENITIES.keys()}
                features.update({f'distance_to_nearest_{amenity}_m': 0 for amenity in AMENITIES.keys()})

        amenity_features_list.append(features)
        pbar.update(1)
        dump_counter += 1

        # Check if dump interval is reached
        if dump_counter % dump_interval == 0:
            # Create a temporary DataFrame up to the current entry
            current_amenities_df = pd.DataFrame(amenity_features_list)
            current_augmented_df = pd.concat([df_selected.iloc[:dump_counter].reset_index(drop=True), 
                                             current_amenities_df.reset_index(drop=True)], axis=1)

            # Generate a unique filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            dump_filename = f"dump_{dump_counter}_{timestamp}.csv"
            dump_path = os.path.join(dump_directory, dump_filename)

            # Save the current augmented DataFrame to the dump file
            try:
                current_augmented_df.to_csv(dump_path, index=False, encoding='utf-8')
            except Exception:
                pass  # Silently ignore any errors during dumping

    pbar.close()

    # Convert the list of dictionaries to a DataFrame
    amenities_df = pd.DataFrame(amenity_features_list)

    # Concatenate the original selected data with the amenities DataFrame
    augmented_df = pd.concat([df_selected, amenities_df], axis=1)

    # Write to the output CSV
    try:
        augmented_df.to_csv(output_database_path, index=False, encoding='utf-8')
        print(f"\nProcessing complete. Augmented data saved to '{output_database_path}'.")
    except Exception:
        pass  # Silently ignore any errors during final saving

# Example usage:
if __name__ == "__main__":
    process_database("df_with_new_features.csv", "df_with_amenities.csv", max_entries=13332)
