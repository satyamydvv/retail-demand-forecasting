import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# 1. LOAD DATA
df = pd.read_csv("data/raw/train.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by=['store_nbr', 'family', 'date'])

# 2. FEATURE ENGINEERING
# Time features
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month

# Categorical Encoding for 'family'
# We store the categories to ensure the Streamlit app uses the same mapping
df['family'] = df['family'].astype('category')
family_mapping = dict(enumerate(df['family'].cat.categories))
df['family_code'] = df['family'].cat.codes

# Advanced Lag features
# Grouping by store and family to ensure lags don't "leak" from other products
df['lag_1'] = df.groupby(['store_nbr', 'family'])['sales'].shift(1)
df['lag_7'] = df.groupby(['store_nbr', 'family'])['sales'].shift(7)
df['lag_14'] = df.groupby(['store_nbr', 'family'])['sales'].shift(14)

# Rolling Statistics (Mean & Std Dev)
df['rolling_mean_7'] = df.groupby(['store_nbr', 'family'])['sales'] \
    .transform(lambda x: x.shift(1).rolling(7).mean())

df['rolling_std_7'] = df.groupby(['store_nbr', 'family'])['sales'] \
    .transform(lambda x: x.shift(1).rolling(7).std())

# Drop rows with NaN values created by shifting/rolling
df = df.dropna()

# 3. MODEL BUILDING
# Updated feature list including the new boosts
features = [
    'store_nbr', 'family_code', 'day_of_week', 'month', 
    'lag_1', 'lag_7', 'lag_14', 'rolling_mean_7', 'rolling_std_7'
]
target = 'sales'

X = df[features]
y = df[target]

# Split data (shuffle=False is important for Time Series)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Initialize and Train
# Change your RandomForest line to this:
model = RandomForestRegressor(
    n_estimators=100, 
    max_depth=12,         # <--- THIS IS THE FIX
    min_samples_leaf=5, 
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# 4. EVALUATION
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print("✅ Model trained successfully")
print(f"📊 MAE: {mae:.4f}")

# 5. SAVE ARTIFACTS
# Save the model
joblib.dump(model, "models/random_forest_model.pkl")

# Save the family mapping so app.py can translate text input to codes
joblib.dump(family_mapping, "models/family_mapping.pkl")

print("💾 Model and Mapping saved to models/ folder")
