import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import folium
from streamlit_folium import folium_static
from streamlit_folium import st_folium
import geopandas as gpd
from shapely.geometry import Point


loaded_model = lgb.Booster(model_file='kyiv_apartment_price_model.txt')

st.title('Kyiv Apartment Price Predictor')

st.header('Enter Apartment Details')

col1, col2 = st.columns(2)

with col1:
    area_total = st.number_input('Total Area (sq m)', min_value=0.0)
    kitchen_area = st.number_input('Kitchen Area (sq m)', min_value=0.0)
    rooms = st.number_input('Number of Rooms', min_value=1, step=1)
    floor = st.number_input('Floor', min_value=1, step=1)
    storeys = st.number_input('Total Storeys', min_value=1, step=1)
    building_age = st.number_input('Building Age (years)', min_value=0, step=1)
    renovation_quality = st.slider('Renovation Quality', 1, 5, 3)

with col2:
    construction_type = st.selectbox('Construction Type', ['монолітно-каркасний', 'цегляний будинок', 'утеплена панель', 'панельні'])
    furnished = st.checkbox('Furnished')
    appliances = st.checkbox('Appliances Included')
    security_features = st.checkbox('Security Features')

st.header('Select Location on Map')


# Initialize session state for storing map click data
if 'location' not in st.session_state:
    st.session_state['location'] = None

# Function to create and update the map
def create_map():
    kyiv_map = folium.Map(location=[50.4501, 30.5234], zoom_start=11)
    folium.ClickForMarker().add_to(kyiv_map)
    return kyiv_map

# Create and display the map
kyiv_map = create_map()
map_data = st_folium(kyiv_map, width=700, height=500)

# Check if a location has been clicked on the map
if map_data['last_clicked']:
    lat = map_data['last_clicked']['lat']
    lon = map_data['last_clicked']['lng']
    st.session_state['location'] = (lat, lon)

# Display the selected latitude and longitude
if st.session_state['location']:
    lat, lon = st.session_state['location']
    st.write(f"Selected Location: Latitude {lat:.4f}, Longitude {lon:.4f}")
    st.write('if you click on marker twice, it will be deleted')

# Button to clear the marker
if st.button('Clear Marker'):
    st.session_state['location'] = None