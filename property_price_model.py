import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import joblib

# Load dataset
df = pd.read_csv('tamilnadu_property_dataset_with_price.csv')

# Preprocessing
# Check for missing values
if df.isnull().sum().any():
    df = df.dropna()

# Convert Y/N to binary
df['Nearby School (Y/N)'] = df['Nearby School (Y/N)'].map({'Y': 1, 'N': 0})
df['Hospital Nearby (Y/N)'] = df['Hospital Nearby (Y/N)'].map({'Y': 1, 'N': 0})

# Feature Engineering: Property Age
df['Property Age'] = 2025 - df['Year Built']

# Drop low-variance columns (Hall, Kitchen) if variance is low
if df['Hall'].var() < 0.1:
    df = df.drop('Hall', axis=1)
if df['Kitchen'].var() < 0.1:
    df = df.drop('Kitchen', axis=1)

# One-hot encode City
df = pd.get_dummies(df, columns=['City'], drop_first=True)

# Define features and target
X = df.drop(['Price (in Lakhs)', 'Year Built'], axis=1)
y = df['Price (in Lakhs)']

# Handle outliers using IQR for numeric columns only
numeric_cols = X.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 0:
    Q1 = X[numeric_cols].quantile(0.25)
    Q3 = X[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    mask = ~((X[numeric_cols] < (Q1 - 1.5 * IQR)) | (X[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
    X = X[mask]
    y = y[mask]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost model
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=6, learning_rate=0.1)
model.fit(X_train_scaled, y_train)

# Evaluate model
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MAE: {mae:.2f} Lakhs, R²: {r2:.4f}")

# Save model and scaler
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(X.columns, 'columns.pkl')

# Save feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
feature_importance.to_csv('feature_importance.csv', index=False)