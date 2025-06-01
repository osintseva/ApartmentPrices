# amenities_script.py

import pandas as pd
import geopandas as gpd
import osmnx as ox
from shapely.geometry import Point
from geopy.distance import geodesic

# Constants and Configuration
AMENITIES = {
    'metro_stations': {'tags': {'subway': 'yes'}, 'radius': 3000},
    'sports_centers': {'tags': {'leisure': 'sports_centre'}, 'radius': 1000},
    'supermarkets': {'tags': {'shop': 'supermarket'}, 'radius': 300},
    'schools': {'tags': {'amenity': 'school'}, 'radius': 500},
    'kindergartens': {'tags': {'amenity': 'kindergarten'}, 'radius': 500},
    'cafes_restaurants': {'tags': {'amenity': ['cafe', 'restaurant']}, 'radius': 300},
    'public_transport': {'tags': {'highway': 'bus_stop'}, 'radius': 300},
    'woods_parks': {'tags': {'natural': 'wood', 'leisure': 'park'}, 'radius': 1000},  # Combined Amenity
    'water_reservoirs': {'tags': {'natural': 'water'}, 'radius': 1000}   # Water Reservoirs as Polygons
}

# Define minimum area for parks, woods, and water reservoirs in square meters
MIN_WOODS_PARKS_AREA_SQM = 6000  # You can adjust this value as needed

def get_projected_crs(gdf):
    """Determine an appropriate projected CRS based on the centroid of all properties."""
    centroid = gdf.geometry.unary_union.centroid
    utm_zone = int((centroid.x + 180) / 6) + 1
    if centroid.y >= 0:
        epsg_code = 32600 + utm_zone  # Northern hemisphere
    else:
        epsg_code = 32700 + utm_zone  # Southern hemisphere
    return f"EPSG:{epsg_code}"

def create_geodataframe(latitude, longitude):
    """Create a GeoDataFrame from the input coordinates."""
    data = {'geometry': [Point(longitude, latitude)]}
    gdf = gpd.GeoDataFrame(data, crs='EPSG:4326')
    return gdf

def fetch_amenities(point, amenity, radius):
    """Fetch amenities from OSM within a specified radius of a point."""
    try:
        tags = AMENITIES[amenity]['tags']
        return ox.features.features_from_point(point, tags=tags, dist=radius)
    except Exception:
        # Suppressed error message to prevent terminal output
        return gpd.GeoDataFrame()

def get_amenity_features(latitude, longitude):
    """
    Fetch and calculate amenities features for a given property coordinate.
    
    Parameters:
        latitude (float): Latitude of the property.
        longitude (float): Longitude of the property.
        
    Returns:
        dict: A dictionary containing counts and distances to nearest amenities.
    """
    # Create GeoDataFrame for the property
    gdf_orig = create_geodataframe(latitude, longitude)
    
    # Determine appropriate projected CRS
    projected_crs = get_projected_crs(gdf_orig)
    
    # Create a projected GeoDataFrame for accurate distance calculations
    gdf_proj = gdf_orig.to_crs(projected_crs)
    
    # Initialize dictionaries to store counts and distances
    amenities_dict = {amenity: 0 for amenity in AMENITIES.keys()}
    distances_dict = {f'distance_to_nearest_{amenity}_m': 0 for amenity in AMENITIES.keys()}  # Initialized to 0
    
    # Process each amenity
    for amenity, details in AMENITIES.items():
        point = (latitude, longitude)
        amenities_gdf = fetch_amenities(point, amenity, details['radius'])
        
        if amenities_gdf.empty:
            amenities_dict[amenity] = 0
            distances_dict[f'distance_to_nearest_{amenity}_m'] = 0  # Set to 0 instead of None
            continue
        
        # If the amenity is 'woods_parks' or 'water_reservoirs', apply area filtering
        if amenity in ['woods_parks', 'water_reservoirs']:
            # Reproject to projected CRS for accurate area calculation
            amenities_proj = amenities_gdf.to_crs(projected_crs).copy()
            
            # Calculate area in square meters
            amenities_proj.loc[:, 'area_sqm'] = amenities_proj.geometry.area
            
            # Filter amenities with area > min_area_sqm
            filtered_amenities = amenities_proj[amenities_proj['area_sqm'] > MIN_WOODS_PARKS_AREA_SQM].copy()
            
            # Update count after filtering
            count = len(filtered_amenities)
            amenities_dict[amenity] = count
            
            if count == 0:
                distances_dict[f'distance_to_nearest_{amenity}_m'] = 0  # Set to 0 instead of None
                continue
            
            # Calculate distances to the nearest amenity
            prop_point_proj = gdf_proj.geometry.iloc[0]
            # Use centroids for polygons
            filtered_amenities.loc[:, 'centroid'] = filtered_amenities.geometry.centroid
            distances = filtered_amenities['centroid'].distance(prop_point_proj)
            min_distance = distances.min()
            distances_dict[f'distance_to_nearest_{amenity}_m'] = round(min_distance, 1)
        else:
            # For other amenities, no area filtering
            # Reproject amenities to projected CRS for accurate distance calculations
            amenities_proj = amenities_gdf.to_crs(projected_crs).copy()
            prop_point_proj = gdf_proj.geometry.iloc[0]
            
            if 'geometry' in amenities_proj and not amenities_proj.empty:
                # Use centroids for non-point geometries
                if any(geom_type != 'Point' for geom_type in amenities_proj.geometry.geom_type):
                    amenities_proj.loc[:, 'centroid'] = amenities_proj.geometry.centroid
                    distances = amenities_proj['centroid'].distance(prop_point_proj)
                else:
                    distances = amenities_proj.geometry.distance(prop_point_proj)
                
                min_distance = distances.min()
                distances_dict[f'distance_to_nearest_{amenity}_m'] = round(min_distance, 1)
                
                # Update count of amenities
                amenities_dict[amenity] = len(amenities_gdf)
            else:
                distances_dict[f'distance_to_nearest_{amenity}_m'] = 0  # Set to 0 instead of None
                amenities_dict[amenity] = 0
    
    # Combine counts and distances into a single dictionary
    result = {}
    for amenity in AMENITIES.keys():
        result[f'{amenity}_count'] = amenities_dict[amenity]
        result[f'distance_to_nearest_{amenity}_m'] = distances_dict[f'distance_to_nearest_{amenity}_m']
    
    return result

# Example usage:
# if __name__ == "__main__":
#     latitude = 50.376064
#     longitude = 30.540911
#     features = get_amenity_features(latitude, longitude)
#     print(features)
