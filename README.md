# 🚗 Car Price Prediction Model

Predict the resale value of a used car in seconds. This project uses machine learning, trained on real-world car listing data, to estimate a fair selling price based on details like the car's age, mileage, fuel type, and more — wrapped in a simple, interactive web app.

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-ML%20Model-F7931E?logo=scikitlearn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![License](https://img.shields.io/badge/License-MIT-blue)

---

## 📌 Overview

Buying or selling a used car often comes down to guesswork — this project replaces that guesswork with a data-driven estimate. The model is trained on historical car sales data and learns the relationship between a car's attributes and its market price, so you can plug in a few details and get an instant, reasonable price prediction.

## ✨ Features

- 🔍 **Instant price prediction** from key car attributes (year, fuel type, transmission, kms driven, etc.)
- 📊 **End-to-end ML pipeline** — data cleaning, exploratory analysis, feature engineering, and model training, all documented in a Jupyter notebook
- 🌐 **Interactive web app** for non-technical users to get predictions without touching code
- 🧩 **Simple, readable codebase** — easy to extend with new features or a different dataset

## 🗂️ Project Structure

```
car_price_app/
├── Car_Price_Prediction_Model.ipynb   # Data exploration, cleaning & model training
├── app.py                             # Web app entry point
├── car_app.py                         # App / prediction logic
├── car data.csv                       # Training dataset
├── requirements.txt                   # Python dependencies
└── README.md
```

## 🛠️ Tech Stack

| Category         | Tools |
|-------------------|-------|
| Language          | Python |
| Data Handling     | Pandas, NumPy |
| Machine Learning  | Scikit-learn |
| Visualization     | Matplotlib / Seaborn |
| Web App           | Streamlit |
| Notebook          | Jupyter |

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/tafrusaidev/Car-Price-Prediction-Model.git
cd Car-Price-Prediction-Model
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

> If this project uses Flask instead, run `python app.py` and open `http://localhost:5000` in your browser.

## 🧠 How It Works

1. **Data preprocessing** — the raw dataset (`car data.csv`) is cleaned, missing values are handled, and categorical features (fuel type, transmission, seller type, etc.) are encoded.
2. **Model training** — a regression model is trained in `Car_Price_Prediction_Model.ipynb` to learn patterns between car features and selling price.
3. **Prediction** — the trained model is loaded by the app to generate real-time price predictions from user input.

## 📈 Example

| Input | Value |
|-------|-------|
| Year | 2018 |
| Present Price | 8.5 Lakh |
| Kms Driven | 25,000 |
| Fuel Type | Petrol |
| Transmission | Manual |
| **Predicted Selling Price** | **₹ 6.2 Lakh** |

*(Sample only — actual output depends on your trained model.)*

## 🤝 Contributing

Contributions are welcome! If you'd like to improve the model, add new features, or clean up the app:

1. Fork the repo
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes
4. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

**Tafrus Qandeel**
GitHub: [@tafrusaidev](https://github.com/tafrusaidev)

---

⭐ If you found this project useful, consider giving it a star on GitHub!
