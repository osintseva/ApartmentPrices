import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

import re

# Load the data
df = pd.read_csv('df_full.csv')

Q1 = df['Price'].quantile(0.25)
Q3 = df['Price'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers
df = df[(df['Price'] >= lower_bound) & (df['Price'] <= upper_bound)]

# Convert "Renovation State" from string to binary integer
df['Renovation State'] = df['Renovation State'].apply(lambda x: 1 if x == "З ремонтом" else 0)

# Calculate 'Building_Age'
df['Building_Age'] = 2024 - df['Construction Year']

# Calculate distance to the city center
center_lat = 50.4501
center_lon = 30.5234

def calculate_airline_distance(lat, lon, center_lat=center_lat, center_lon=center_lon):
    return np.sqrt((lat - center_lat) ** 2 + (lon - center_lon) ** 2)

df['distance_to_center'] = calculate_airline_distance(df['Latitude'], df['Longitude'])

# Calculate mean distance by District ID
mean_distance_by_district = df.groupby('District ID')['distance_to_center'].mean().reset_index()
mean_distance_by_district.rename(columns={'distance_to_center': 'mean_distance_to_center'}, inplace=True)

# Merge back into the original dataframe
df = df.merge(mean_distance_by_district, on='District ID', how='left')

categorical_features = ['District ID', 'Construction Type'] + [  'furnished', 'appliances', 'security_features']
numerical_features = ['Rooms', 'Area_total', 'Kitchen_area', 'Storeys', 'Floor', 'Building_Age', 'distance_to_center', 'renovation_quality', 'distance_to_nearest_metro_stations_m']

# Define target variable as Log_Price (log-transformed Price)
y = np.log(df['Price'])  # Log transform the Price
X = df[categorical_features + numerical_features]

# Encode categorical features as integers (LightGBM requires this)
for col in categorical_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Define the LightGBM dataset
lgb_train = lgb.Dataset(X, label=y, categorical_feature=categorical_features, free_raw_data=False)

# Define the model parameters
parameters = {
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.01,
    "num_threads": 10,
    "seed": 42,

    # Regularization parameters
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "subsample_freq": 1,
    "num_leaves": 50,
    "min_data_in_leaf": 20,

    # Handling categorical features
    "cat_smooth": 10,
    "min_data_per_group": 50
}

# Number of boosting rounds
n_rounds = 10000

# Setup KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

def custom_mae_eval(preds, train_data):
    """Custom evaluation function for MAE in original price scale."""
    labels = train_data.get_label()
    preds_transformed = np.exp(preds)
    labels_transformed = np.exp(labels)
    mae_value = mean_absolute_error(labels_transformed, preds_transformed)
    return ('mae_original_scale', mae_value, False)

# Perform cross-validation and train the model with custom evaluation function
results = lgb.cv(parameters, lgb_train, n_rounds,
                 folds=kf, stratified=False,
                 feval=custom_mae_eval,
                 eval_train_metric=True,
                 callbacks=[lgb.early_stopping(50), lgb.log_evaluation(period=100)])

# Check available keys in results
print(results.keys())

# Determine the number of boosting rounds based on cross-validation
num_boost_round = len(results['valid mae_original_scale-mean'])

# Train the model using the full dataset
model = lgb.train(parameters, lgb_train, num_boost_round=num_boost_round)

# Save the model
model.save_model('kyiv_apartment_price_model.txt')

print("Model has been saved to 'kyiv_apartment_price_model.txt'")