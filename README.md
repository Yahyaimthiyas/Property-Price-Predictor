# 🏠 Property Price Predictor

A machine learning web application to predict property prices in Tamil Nadu, India, based on various features. Built with Python, Flask, and scikit-learn.

## 🚀 Features

- Predicts property prices using a trained machine learning model
- User-friendly web interface
- Feature importance visualization
- Scalable and easy to use

## 📊 Dataset

The model is trained on the `tamilnadu_property_dataset_with_price.csv` dataset, which contains real estate data from Tamil Nadu, India.

## 🛠️ Tech Stack

- **Backend:** Python, Flask
- **Machine Learning:** scikit-learn
- **Frontend:** HTML, CSS (with custom styles)
- **Serialization:** pickle

## 📦 Project Structure

```
property_predictor/
│
├── app.py                        # Main Flask application
├── property_price_model.py       # Model training and prediction logic
├── model.pkl                     # Trained ML model
├── scaler.pkl                    # Feature scaler
├── columns.pkl                   # Feature columns used in the model
├── feature_importance.csv        # Feature importance data
├── tamilnadu_property_dataset_with_price.csv  # Dataset
├── static/
│   └── styles.css                # Custom CSS styles
└── templates/
    └── index.html                # Main HTML template
```

## ⚡ Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/property_predictor.git
cd property_predictor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** If `requirements.txt` is missing, install Flask and scikit-learn:
> ```bash
> pip install flask scikit-learn
> ```

### 3. Run the application

```bash
python app.py
```

The app will be available at [http://127.0.0.1:5000](http://127.0.0.1:5000).

## 🖥️ Usage

1. Open the web app in your browser.
2. Enter the required property details.
3. Click "Predict" to get the estimated property price.

## 📈 Model

- The model is trained using scikit-learn.
- Feature scaling and column transformation are handled using `scaler.pkl` and `columns.pkl`.
- Feature importance is visualized using data from `feature_importance.csv`.

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for improvements or bug fixes.

## 📄 License

This project is licensed under the MIT License.

## 📬 Contact

- **Developer:** Yahya Imthiyas  
- **Email:** yahyaimthiyas2005@gmail.com 
