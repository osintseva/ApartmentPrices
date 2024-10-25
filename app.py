import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import folium
from streamlit_folium import folium_static
from streamlit_folium import st_folium
import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial import distance
from sklearn.preprocessing import LabelEncoder


loaded_model = lgb.Booster(model_file='kyiv_apartment_price_model.txt')
df = pd.read_csv('df_full.csv')

ids = [
    10026800, 10026817, 10026650, 10026676, 10026675, 10026832, 10026671, 10026831, 10026833, 10026672,
    10026684, 10026668, 10026674, 10026673, 10026669, 10026834, 10026632, 10026802, 10026637, 10026799,
    10026801, 10026633, 10026635, 10026695, 10026634, 10026636, 10026804, 10026638, 10026639, 10026640,
    10026805, 10026641, 10026839, 10026660, 10026806, 10026803, 10026642, 10026643, 10026798, 10026812,
    10026814, 10026647, 10026810, 10026807, 10026644, 10026645, 10026652, 10026813, 10026677, 10026808,
    10026809, 10026811, 10026648, 10026651, 10026681, 10026680, 10026656, 10026655, 10026815, 10026657,
    10026653, 10026816, 10026678, 10026818, 10026654, 10026682, 10026658, 10026661, 10026823, 10026827,
    10026670, 10026830, 10026666, 10026825, 10026824, 10026683, 10026819, 10026687, 10026835, 10026836,
    10026688, 10026837, 10026829, 10026828, 10026667, 10026664, 10026820, 10026659, 10026821, 10026679,
    10026662, 10026826, 10026663, 10026665
]
district_names = [
    "Багринова гора", "Нижні сади", "Лісовий", "Шулявка", "Харківський", "Старий Київ", "Татарка", "Солдатська Слобідка", "Теремки", "Теремки-1",
    "Соломянка", "Соцмісто", "Троєщина", "Теремки-2", "Стара Дарниця", "Феофанія", "Академмістечко", "Берковець", "Биківня", "Біличе поле",
    "Байкова гора", "Біличі", "Бортничі", "Батиєва гора", "Березняки", "Борщагівка", "Верхня Теличка", "Вітряні Гори", "Виноградар", "Воскресенка",
    "Вишгородський масив", "Голосіїв", "Галагани", "Відрадний", "Дачі Осокорки", "Віта-Литовська", "ДВРЗ", "Деміївка", "Історичний центр", "Конча-Заспа",
    "Лівобережний масив", "Корчувате", "Кадетський гай", "Добрий шлях", "Жуляни", "КПІ", "Лукянівка", "Кудрявець", "Караваєві Дачі", "Залізничний",
    "Замковище", "Катеринівка", "Куренівка", "Липки", "Китаїв", "Звіринець", "Микільська Слобідка", "Нижній Печерськ", "Мишоловка", "Нова Дарниця",
    "Мінський", "Мостицький масив", "Микільська Борщагівка", "Нова забудова", "Нивки", "Новобіличі", "Оболонь", "Печерськ", "Почайна", "Русанівські сади",
    "Сирець", "Совки", "Саперна Слобідка", "Пріорка", "Пирогів", "Пуща-Водиця", "Південна Борщагівка", "Олександрівська Слобідка", "Царське село", "Черепанова гора",
    "Чоколівка", "Чорна Гора", "Село Шевченка", "Село Троєщина", "Святошино", "Райдужний", "Північно-Броварський", "Осокорки", "Паньківщина", "Первомайський",
    "Поділ", "Рибальський острів", "Позняки", "Русанівка"
]

# Create a dictionary mapping District IDs to their names
district_dict = dict(zip(ids, district_names))

# Function to calculate airline distance
def calculate_airline_distance(lat, lon, center_lat=50.4501, center_lon=30.5234):
    return np.sqrt((lat - center_lat) ** 2 + (lon - center_lon) ** 2)

metro_stations = {
    'Lisova': (50.46444, 30.645),
    'Chernihivska': (50.46, 30.63083),
    'Darnytsia': (50.45556, 30.61333),
    'Livoberezhna': (50.45194, 30.59833),
    'Hidropark': (50.44596, 30.57699),
    'Dnipro': (50.44111, 30.55917),
    'Arsenalna': (50.44449, 30.54543),
    'Khreshchatyk': (50.44722, 30.52278),
    'Teatralna': (50.44528, 30.51806),
    'Universytet': (50.4442, 30.5058),
    'Vokzalna': (50.44167, 30.48806),
    'Politekhnichnyi Instytut': (50.4508, 30.4661),
    'Shuliavska': (50.45508, 30.44537),
    'Beresteiska': (50.45861111, 30.41972222),
    'Nyvky': (50.4583, 30.404),
    'Sviatoshyn': (50.4575, 30.39194),
    'Zhytomyrska': (50.45617, 30.36587),
    'Akademmistechko': (50.4647, 30.35508),
    'Syrets': (50.47639, 30.43083),
    'Dorohozhychi': (50.47083, 30.44283),
    'Lukianivska': (50.4625, 30.48194),
    'Zoloti Vorota': (50.44608, 30.51553),
    'Palats Sportu': (50.43917, 30.51972),
    'Klovska': (50.4355, 30.5257),
    'Pecherska': (50.4275, 30.53889),
    'Druzhby Narodiv': (50.41975, 30.51955),
    'Vydubychi': (50.416775, 30.567579),
    'Slavutych': (50.3942, 30.60516),
    'Osokorky': (50.39519, 30.61625),
    'Pozniaky': (50.39806, 30.63333),
    'Kharkivska': (50.40083, 30.65222),
    'Vyrlytsia': (50.40333, 30.66611),
    'Boryspilska': (50.402776, 30.67633),
    'Chervonyi Khutir': (50.4092, 30.695),
    'Heroiv Dnipra': (50.52267, 30.4989),
    'Minska': (50.51222, 30.49861),
    'Obolon': (50.50138889, 30.49805556),
    'Pochaina': (50.48694444, 30.49777778),
    'Tarasa Shevchenka': (50.47306, 30.50528),
    'Kontraktova Ploshcha': (50.46583, 30.515),
    'Poshtova Ploshcha': (50.45917, 30.52494),
    'Maidan Nezalezhnosti': (50.45, 30.52444),
    'Ploshcha Ukrainskykh Heroiv': (50.4391, 30.5161),
    'Olimpiiska': (50.43222, 30.51611),
    'Palats Ukraina': (50.42083333, 30.52083333),
    'Lybidska': (50.41311, 30.52483),
    'Demiivska': (50.40491, 30.51675),
    'Holosiivska': (50.39767, 30.50833),
    'Vasylkivska': (50.39334, 30.48822),
    'Vystavkovyi Tsentr': (50.3825, 30.4775)
}

#___________________________________________________
#___________________________________________________

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


#dst to entries in dataset
df['distance'] = df.apply(lambda row: distance.euclidean((lat, lon), (row['Latitude'], row['Longitude'])), axis=1)

# Find the 5 nearest entries
nearest_entries = df.nsmallest(5, 'distance')

# Determine the most common District ID among these entries
most_common_district_id = nearest_entries['District ID'].mode()[0]

# Get the district name from the dictionary
district_name = district_dict.get(most_common_district_id)

st.write(f"Most Common District ID: {most_common_district_id}")
st.write(f"District Name: {district_name}")




# Calculate distance to city center
distance_to_center = calculate_airline_distance(lat, lon)
#st.write(f"Distance to City Center: {distance_to_center:.2f} units")

# Calculate distance to all metro stations and find the minimum distance
metro_distances = {station: distance.euclidean((lat, lon), coords) for station, coords in metro_stations.items()}
min_distance_to_metro = min(metro_distances.values())
min_station_name = min(metro_distances, key=metro_distances.get)

#st.write(f"Nearest Metro Station: {min_station_name} at {min_distance_to_metro:.2f} units")


def prepare_input_data():
    # Encode categorical features if necessary (example using LabelEncoder)
    
    le_construction_type = LabelEncoder()
    
    # Assuming you have fitted these encoders during training with all possible categories.
    
    construction_type_encoded = le_construction_type.fit_transform([construction_type])[0]
    
    return pd.DataFrame({
        'District ID': [most_common_district_id],
        'Construction Type': [construction_type_encoded],
        'furnished': [int(furnished)],
        'appliances': [int(appliances)],
        'security_features': [int(security_features)],
        'Rooms': [rooms],
        'Area_total': [area_total],
        'Kitchen_area': [kitchen_area],
        'Storeys': [storeys],
        'Floor': [floor],
        'Building_Age': [building_age],
        'distance_to_center': [distance_to_center],
        'renovation_quality': [renovation_quality],
        'distance_to_nearest_metro_stations_m': [min_distance_to_metro * 1000]  # Convert km to meters if needed
    })

if st.button('Predict Price'):
    input_data = prepare_input_data()
    
    prediction_log_price = loaded_model.predict(input_data)
    
    # Convert log price back to actual price
    predicted_price = np.exp(prediction_log_price)
    
    st.success(f"Predicted Apartment Price: {predicted_price[0]:,.2f} UAH")