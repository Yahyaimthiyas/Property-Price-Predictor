from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load model and data
try:
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    columns = joblib.load('columns.pkl')
    feature_importance = pd.read_csv('feature_importance.csv')
    if not all(col in feature_importance.columns for col in ['feature', 'importance']):
        raise ValueError("feature_importance.csv missing required columns")
    df = pd.read_csv('tamilnadu_property_dataset_with_price.csv')
    df['Property Age'] = 2025 - df['Year Built']
    df['Nearby School (Y/N)'] = df['Nearby School (Y/N)'].map({'Y': 1, 'N': 0})
    df['Hospital Nearby (Y/N)'] = df['Hospital Nearby (Y/N)'].map({'Y': 1, 'N': 0})
    r2 = 0.95  # Placeholder
    mae = 5.0  # Placeholder
except Exception as e:
    print(f"Error loading model/data: {e}")
    model, scaler, columns, feature_importance = None, None, [], pd.DataFrame()
    df = pd.DataFrame()
    r2, mae = 0, 5.0

cities = sorted(df['City'].unique()) if not df.empty else ['Chennai', 'Coimbatore', 'Madurai']

# Prepare chart data
def prepare_chart_data():
    chart_data = {}
    required_columns = ['City', 'Price (in Lakhs)', 'Sqft', 'Year Built', 'Property Age']
    if df.empty or not all(col in df.columns for col in required_columns):
        print("Warning: DataFrame is empty or missing required columns. Returning empty chart_data.")
        return chart_data
    try:
        # Average Price by City
        city_avg = df.groupby('City')['Price (in Lakhs)'].mean().to_dict()
        chart_data['avg_price_city'] = [{'city': k, 'price': round(v, 2)} for k, v in city_avg.items()]
        print(f"Avg Price by City: {len(chart_data['avg_price_city'])} cities")

        # Price vs. Sqft
        sample = df.sample(min(100, len(df)), random_state=42)[['Sqft', 'Price (in Lakhs)']].to_dict('records')
        chart_data['price_vs_sqft'] = [{'sqft': d['Sqft'], 'price': round(d['Price (in Lakhs)'], 2)} for d in sample]
        print(f"Price vs. Sqft: {len(chart_data['price_vs_sqft'])} points")

        # Price vs. Property Age
        sample = df.sample(min(100, len(df)), random_state=42)[['Property Age', 'Price (in Lakhs)']].to_dict('records')
        chart_data['price_vs_age'] = [{'age': d['Property Age'], 'price': round(d['Price (in Lakhs)'], 2)} for d in sample]
        print(f"Price vs. Age: {len(chart_data['price_vs_age'])} points")

        # Price Distribution by City
        chart_data['price_dist'] = {}
        for city in cities:
            prices = df[df['City'] == city]['Price (in Lakhs)'].values
            if len(prices) > 0:
                bins = np.histogram(prices, bins=10, range=(0, max(prices)))[1].tolist()
                hist = np.histogram(prices, bins=bins)[0].tolist()
                chart_data['price_dist'][city] = {'bins': [round(b, 2) for b in bins], 'counts': hist}
        print(f"Price Dist: {len(chart_data['price_dist'])} cities")

        # Price vs. Sqft vs. Age (2D scatter for simplicity)
        sample = df.sample(min(50, len(df)), random_state=42)[['Sqft', 'Property Age', 'Price (in Lakhs)']].to_dict('records')
        chart_data['scatter_3d'] = [{'sqft': d['Sqft'], 'age': d['Property Age'], 'price': round(d['Price (in Lakhs)'], 2)} for d in sample]
        print(f"Scatter 3D: {len(chart_data['scatter_3d'])} points")

        # Heatmap (simplified as bar chart)
        chart_data['heatmap'] = [{'city': city, 'price': round(city_avg.get(city, 0), 2)} for city in cities]
        print(f"Heatmap: {len(chart_data['heatmap'])} cities")

    except Exception as e:
        print(f"Error preparing chart data: {e}")
        chart_data = {}
    
    return chart_data

# Simulate price forecast
def simulate_price_forecast(city, current_price):
    default_forecast = [{'time': f'Q{i+1} 2025' if i < 4 else 'Q1 2026', 'price': 50.0} for i in range(5)]
    try:
        current_price = float(current_price)
        if not np.isfinite(current_price) or current_price <= 0:
            current_price = 50.0
        growth_rate = 0.02
        forecast = []
        price = current_price
        for i in range(5):
            price *= (1 + growth_rate)
            forecast.append({'time': f'Q{i+1} 2025' if i < 4 else 'Q1 2026', 'price': round(max(price, 1.0), 2)})
        print(f"Forecast for {city}: {forecast}")
        return forecast
    except Exception as e:
        print(f"Error in forecast for {city}: {e}")
        return default_forecast

# Shared prediction logic
def predict_price(data):
    try:
        city = data['city']
        if city not in cities:
            raise ValueError("Invalid city selected")
        bedroom = data['bedroom']
        if not bedroom.isdigit() or int(bedroom) < 1 or int(bedroom) > 5:
            raise ValueError("Bedrooms must be between 1 and 5")
        sqft = data['sqft']
        try:
            sqft = float(sqft)
            if sqft < 100 or sqft > 10000:
                raise ValueError
        except ValueError:
            raise ValueError("Square footage must be a number between 100 and 10,000")
        year_built = data['year_built']
        try:
            year_built = int(year_built)
            if year_built < 1900 or year_built > 2025:
                raise ValueError
        except ValueError:
            raise ValueError("Year built must be a number between 1900 and 2025")
        distance = data['distance']
        try:
            distance = float(distance)
            if distance < 0 or distance > 50:
                raise ValueError
        except ValueError:
            raise ValueError("Distance must be a number between 0 and 50")
        school = data['school']
        if school not in ['Y', 'N']:
            raise ValueError("Invalid school selection")
        hospital = data['hospital']
        if hospital not in ['Y', 'N']:
            raise ValueError("Invalid hospital selection")

        input_data = {
            'City': city,
            'Bedroom': int(bedroom),
            'Sqft': sqft,
            'Year Built': year_built,
            'Distance to City Center (km)': distance,
            'Nearby School (Y/N)': 1 if school == 'Y' else 0,
            'Hospital Nearby (Y/N)': 1 if hospital == 'Y' else 0,
            'Property Age': 2025 - year_built
        }

        input_df = pd.DataFrame([input_data])
        input_df = pd.get_dummies(input_df, columns=['City'], drop_first=False)
        for col in columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[columns]
        print(f"Input features for {city}: {input_df.to_dict('records')[0]}")
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        print(f"Raw model prediction for {city}: {prediction}")
        if not np.isfinite(prediction) or prediction <= 0:
            print(f"Invalid prediction: {prediction}, using default 50.0")
            prediction = 50.0
        else:
            prediction = float(prediction)
            prediction = max(prediction, 1.0)
        
        # Calculate dynamic feature importance
        try:
            feature_importances = model.feature_importances_ if hasattr(model, 'feature_importances_') else np.ones(len(columns)) / len(columns)
            feature_importance_list = [
                {'feature': col, 'importance': round(float(imp), 4)}
                for col, imp in zip(columns, feature_importances)
                if imp > 0  # Only include features with non-zero importance
            ]
            feature_importance_list = sorted(feature_importance_list, key=lambda x: x['importance'], reverse=True)[:5]
            print(f"Dynamic feature importance for {city}: {feature_importance_list}")
        except Exception as e:
            print(f"Error calculating feature importance: {e}")
            feature_importance_list = feature_importance.to_dict('records')[:5]

        lower_bound = max(prediction - mae, 1.0)
        upper_bound = prediction + mae
        similar = df[
            (df['City'] == city) &
            (df['Bedroom'] == input_data['Bedroom']) &
            (df['Sqft'].between(input_data['Sqft'] * 0.9, input_data['Sqft'] * 1.1))
        ]
        avg_price_similar = float(similar['Price (in Lakhs)'].mean()) if not similar.empty else None
        city_avg_price = float(df[df['City'] == city]['Price (in Lakhs)'].mean()) if not df[df['City'] == city].empty else None
        price_forecast = simulate_price_forecast(city, prediction)
        if not isinstance(price_forecast, list):
            print(f"price_forecast is not a list: {price_forecast}, using default forecast")
            price_forecast = [{'time': f'Q{i+1} 2025' if i < 4 else 'Q1 2026', 'price': 50.0} for i in range(5)]
        
        scale_factor = upper_bound * 1.2
        left_percent = max(0, min(100, (lower_bound / scale_factor) * 100))
        width_percent = max(0, min(100 - left_percent, ((upper_bound - lower_bound) / scale_factor) * 100))
        marker_percent = max(0, min(100, (prediction / scale_factor) * 100))
        
        return {
            'prediction': round(prediction, 2),
            'lower_bound': round(lower_bound, 2),
            'upper_bound': round(upper_bound, 2),
            'avg_price_similar': round(avg_price_similar, 2) if avg_price_similar else None,
            'city_avg_price': round(city_avg_price, 2) if city_avg_price else None,
            'city': city,
            'price_forecast': price_forecast,
            'input_data': input_data,
            'error_message': None,
            'left_percent': round(left_percent, 2),
            'width_percent': round(width_percent, 2),
            'marker_percent': round(marker_percent, 2),
            'feature_importance': feature_importance_list,
            'mae': mae
        }
    except Exception as e:
        print(f"Error in predict: {e}")
        return {
            'error_message': f"Error predicting price: {str(e)}",
            'prediction': None,
            'price_forecast': [{'time': f'Q{i+1} 2025' if i < 4 else 'Q1 2026', 'price': 50.0} for i in range(5)]
        }

@app.route('/', methods=['GET'])
def home():
    chart_data = prepare_chart_data()
    return render_template('index.html', 
                         cities=cities, 
                         metrics={'r2': r2, 'mae': mae},
                         feature_importance=feature_importance.to_dict('records'),
                         chart_data=chart_data)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    print(f"Predict input: {data}")
    prediction_data = predict_price(data)
    return jsonify(prediction_data)

@app.route('/compare', methods=['POST'])
def compare():
    try:
        data1 = {
            'city': request.form['city1'],
            'bedroom': request.form['bedroom1'],
            'sqft': request.form['sqft1'],
            'year_built': request.form['year_built1'],
            'distance': request.form['distance1'],
            'school': request.form['school1'],
            'hospital': request.form['hospital1']
        }
        data2 = {
            'city': request.form['city2'],
            'bedroom': request.form['bedroom2'],
            'sqft': request.form['sqft2'],
            'year_built': request.form['year_built2'],
            'distance': request.form['distance2'],
            'school': request.form['school2'],
            'hospital': request.form['hospital2']
        }
        print(f"Compare input: Property 1: {data1}, Property 2: {data2}")
        result1 = predict_price(data1)
        result2 = predict_price(data2)
        if result1['error_message'] or result2['error_message']:
            raise ValueError(result1['error_message'] or result2['error_message'])
        
        print(f"Compare results: Property 1: {result1['prediction']}, Property 2: {result2['prediction']}")
        return jsonify({
            'property1': result1,
            'property2': result2
        })
    except Exception as e:
        print(f"Error in compare: {e}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)