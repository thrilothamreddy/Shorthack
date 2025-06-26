import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset
df = pd.read_csv(r"C:\Users\tnarannagari\smart-rent-estimator\House_Rent_Dataset.csv")

# Keep only necessary columns
required_columns = ['BHK', 'Size', 'StayFloor', 'Bathrooms', 
                    'Furnishing Status', 'Tenant Type', 'Area Type', 'City', 'Area Locality', 'Rent']
df = df[required_columns]

# Rename columns for easier handling
df.rename(columns={
    'BHK': 'bhk',
    'Size': 'size',
    'StayFloor': 'floor',
    'Bathrooms': 'bathroom',
    'Furnishing Status': 'furnishing',
    'Tenant Type': 'tenant',
    'Area Type': 'area_type',
    'City': 'city',
    'Area Locality': 'locality',
    'Rent': 'rent'
}, inplace=True)

# Drop rows with missing values
df.dropna(inplace=True)

# Encode categorical variables
furnishing_map = {'Furnished': 2, 'Semi-Furnished': 1, 'Unfurnished': 0}
tenant_map = {'Bachelors': 0, 'Family': 1, 'Bachelors/Family': 2}
area_type_map = {'Super Area': 0, 'Carpet Area': 1}
city_map = {'Kolkata': 0, 'Hyderabad': 1, 'Mumbai': 2, 'Chennai': 3, 'Delhi': 4, 'Bangalore': 5}

df['furnishing'] = df['furnishing'].map(furnishing_map)
df['tenant'] = df['tenant'].map(tenant_map)
df['area_type'] = df['area_type'].map(area_type_map)
df['city'] = df['city'].map(city_map)

# Encode locality using LabelEncoder
locality_encoder = LabelEncoder()
df['locality'] = locality_encoder.fit_transform(df['locality'])

# ✅ Save the locality encoder for use in app.py
joblib.dump(locality_encoder, "locality_encoder.pkl")
print("✅ Locality encoder saved as locality_encoder.pkl")

# Final feature list including locality
features = ['bhk', 'size', 'floor', 'bathroom', 'furnishing', 'tenant', 'area_type', 'city', 'locality']
X = df[features]
y = df['rent']

print(f"🧾 Total samples available: {len(X)}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save model and feature list
joblib.dump(model, "model.pkl")
print("✅ Model saved as model.pkl")

joblib.dump(features, "features.pkl")
print("✅ Features saved as features.pkl")
