import numpy as np
import pandas as pd
import time  # Import the time module to track execution time
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import category_encoders as ce  
from scipy.stats import randint, uniform  # For defining distributions in RandomizedSearchCV

# ===========================================
# 1. Data Loading and Preprocessing
# ===========================================

# Load the data
df = pd.read_csv('df_full.csv')

# Delete outliers using IQR for 'Price'
Q1 = df['Price'].quantile(0.25)
Q3 = df['Price'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers
df = df[(df['Price'] >= lower_bound) & (df['Price'] <= upper_bound)]

# Update features
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

# ===========================================
# 2. Feature Engineering and Encoding
# ===========================================

# Features to be target encoded
target_encoded_features = ['District ID']

# Truly categorical features to be one-hot encoded
onehot_encoded_features = ['Construction Type']

# Binary categorical features treated as numerical
binary_features = ['balcony', 'wardrobe', 'view', 'furnished', 'appliances',
                   'floor_heating', 'air_conditioning', 'parking', 'security_features']

# Numerical features (including 'mean_distance_to_center' if intended)
numerical_features = ['Rooms', 'Area_total', 'Kitchen_area', 'Storeys', 'Floor',
                      'Renovation State', 'Building_Age', 'distance_to_center',
                      'renovation_quality', 'distance_to_nearest_metro_stations_m',
                      'sports_centers_count', 'supermarkets_count', 'schools_count',
                      'kindergartens_count', 'cafes_restaurants_count',
                      'public_transport_count',
                      'distance_to_nearest_woods_parks_m',
                      'distance_to_nearest_water_reservoirs_m',
                      'mean_distance_to_center'  # Ensure this is included if intended
                      ]

# Convert binary features to integer type (0/1)
for col in binary_features:
    df[col] = df[col].astype(int)

# Ensure 'Construction Type' is of categorical type
df['Construction Type'] = df['Construction Type'].astype('category')

# ===========================================
# 3. Defining the Preprocessor
# ===========================================

# Define the preprocessor with TargetEncoder and OneHotEncoder
preprocessor = ColumnTransformer(
    transformers=[
        ('target_enc', ce.TargetEncoder(cols=target_encoded_features, return_df=False), target_encoded_features),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'), onehot_encoded_features)
    ],
    remainder='passthrough'  # Keep numerical features and binary features as they are
)

# ===========================================
# 4. Defining the Target Variable and Feature Matrix
# ===========================================

# Define target variable as Log_Price (log-transformed Price)
y = np.log(df['Price'])  # Log-transform the Price

# Define feature matrix
X = df[target_encoded_features + onehot_encoded_features + binary_features + numerical_features]

# ===========================================
# 5. Setting Up the Pipeline and Randomized Search
# ===========================================

# Set up the Random Forest regression model pipeline with the updated preprocessor
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=200, random_state=42))
    ])

# Define the parameter distribution for Randomized Search
param_dist = {
    'regressor__max_depth': [None, 6, 8, 9],
    'regressor__min_samples_split': randint(2, 11),  # 2 to 10 inclusive
    'regressor__min_samples_leaf': randint(1, 5),    # 1 to 4 inclusive
    'regressor__max_features': uniform(0.7, 0.2)    # 0.7 to 0.9
}

# Prepare KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize lists to store training and validation MAE
train_maes = []
val_maes = []

# Track the start time
start_time = time.time()

# Perform cross-validation with Randomized Search
fold = 1
for train_index, val_index in kf.split(X):
    fold_start_time = time.time()  # Track time for the current fold

    # Split into train and validation sets with .copy() to avoid SettingWithCopyWarning
    X_train, X_val = X.iloc[train_index].copy(), X.iloc[val_index].copy()
    y_train, y_val = y.iloc[train_index].copy(), y.iloc[val_index].copy()

    # Initialize RandomizedSearchCV
    randomized_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=20,  # Number of parameter settings sampled
        cv=3,        # Inner cross-validation for hyperparameter tuning
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        random_state=42,
        verbose=0
    )

    # Fit RandomizedSearchCV on the training data
    randomized_search.fit(X_train, y_train)

    # Best estimator from randomized search
    best_model = randomized_search.best_estimator_

    # Predict on training and validation data using the best model
    y_train_pred_log = best_model.predict(X_train)
    y_val_pred_log = best_model.predict(X_val)

    # Inverse transform log predictions and actuals back to original price scale
    y_train_pred = np.exp(y_train_pred_log)
    y_val_pred = np.exp(y_val_pred_log)
    y_train_real = np.exp(y_train)
    y_val_real = np.exp(y_val)

    # Calculate MAE for training and validation data on original price scale
    train_mae = mean_absolute_error(y_train_real, y_train_pred)
    val_mae = mean_absolute_error(y_val_real, y_val_pred)

    train_maes.append(train_mae)
    val_maes.append(val_mae)

    # Calculate time taken for the current fold and estimate remaining time
    fold_time = time.time() - fold_start_time
    total_time_elapsed = time.time() - start_time
    estimated_remaining_time = (kf.get_n_splits() - fold) * fold_time

    # Print MAE and estimated remaining time for each fold
    print(f"Fold {fold} - Training MAE (Original Price): {train_mae:,.2f}, Validation MAE (Original Price): {val_mae:,.2f}")
    print(f"Best Parameters for Fold {fold}: {randomized_search.best_params_}")
    print(f"Time taken for Fold {fold}: {fold_time:.2f} seconds")
    print(f"Estimated remaining time: {estimated_remaining_time:.2f} seconds\n")

    fold += 1

# Calculate the average MAE across all folds
average_train_mae = np.mean(train_maes)
average_val_mae = np.mean(val_maes)

print(f"\nAverage Training MAE (Original Price): {average_train_mae:,.2f}")
print(f"Average Validation MAE (Original Price): {average_val_mae:,.2f}")
