# Hotel Booking Cancellation Prediction

## Overview

This project is an end-to-end machine learning application that predicts whether a hotel booking is likely to be canceled before the customer's arrival. The goal is to help hotels identify high-risk bookings and make better business decisions such as overbooking strategies, targeted offers, and resource planning.

The project goes beyond model training by covering the complete ML workflow—from data exploration and feature engineering to model comparison, explainability with SHAP, and deployment as a REST API using FastAPI.

---

## Problem Statement

Hotel cancellations create operational and financial challenges. Empty rooms due to last-minute cancellations result in lost revenue, while excessive overbooking can negatively affect customer experience.

The objective of this project is to build a machine learning model that can predict the probability of a booking being canceled based on information available at the time of reservation.

---

## Dataset

The project uses a publicly available hotel booking dataset containing over **36,000 booking records** with a mix of numerical and categorical features.

Some of the important attributes include:

* Lead time
* Number of adults and children
* Number of week and weekend nights
* Meal type
* Room type
* Market segment
* Average price per room
* Previous booking history
* Special requests
* Reservation date

**Target Variable:**

* `booking status`

  * `0` → Not Canceled
  * `1` → Canceled

---

## Project Workflow

### 1. Exploratory Data Analysis (EDA)

* Checked dataset structure and data types.
* Verified missing values and duplicate records.
* Analyzed target class distribution.
* Investigated relationships between important features and booking cancellations.
* Identified and removed non-informative identifier columns.

### 2. Feature Engineering

* Converted reservation dates into datetime format.
* Created additional date-based features:

  * Reservation month
  * Day of week
  * Quarter
  * Weekend booking indicator
* Built separate numerical and categorical feature groups.

### 3. Preprocessing Pipeline

A production-ready Scikit-Learn pipeline was created using:

* `SimpleImputer`
* `OneHotEncoder`
* `ColumnTransformer`
* `Pipeline`

This ensures that the exact same preprocessing steps are applied during both training and inference.

### 4. Model Building

The following models were trained and evaluated:

* Decision Tree (Baseline)
* XGBoost
* LightGBM
* CatBoost

### 5. Model Explainability

SHAP (SHapley Additive exPlanations) was used to:

* Understand global feature importance.
* Explain individual predictions.
* Visualize how each feature contributes to the model's decisions.

### 6. Deployment

The best performing model was saved using `joblib` and deployed as a REST API using FastAPI. The API accepts booking details as input and returns both:

* Predicted booking status.
* Probability of cancellation.

---

## Model Performance

| Model         | Accuracy   | Precision  | Recall     | F1 Score   |
| ------------- | ---------- | ---------- | ---------- | ---------- |
| Decision Tree | 83.84%     | 77.30%     | 71.74%     | 74.42%     |
| XGBoost       | 86.34%     | 83.58%     | 72.58%     | 77.70%     |
| **LightGBM**  | **86.40%** | **83.42%** | **73.00%** | **77.86%** |
| CatBoost      | 85.94%     | 83.85%     | 70.73%     | 76.73%     |

**Best Model:** LightGBM

The final deployed model achieved an accuracy of approximately **86.4%** while maintaining a good balance between precision and recall.

---

## SHAP Insights

Model interpretability revealed several interesting patterns:

* Bookings made far in advance (`lead time`) are significantly more likely to be canceled.
* Customers making multiple `special requests` are generally less likely to cancel.
* `Online` bookings have a higher cancellation tendency compared to other booking channels.
* Repeat customers rarely cancel compared to first-time customers.
* Higher average room prices show a slight positive relationship with cancellations.

These insights align well with real-world hotel booking behavior and provide business value beyond prediction alone.

---

## API Example

### Request

```json
{
  "number_of_adults": 2,
  "number_of_children": 0,
  "number_of_weekend_nights": 1,
  "number_of_week_nights": 2,
  "type_of_meal": "Meal Plan 1",
  "car_parking_space": 0,
  "room_type": "Room_Type 1",
  "lead_time": 150,
  "market_segment_type": "Online",
  "repeated": 0,
  "P_C": 0,
  "P_not_C": 0,
  "average_price": 120.5,
  "special_requests": 0,
  "reservation_month": 6,
  "reservation_dayofweek": 2,
  "reservation_quarter": 2,
  "is_weekend_booking": 0
}
```

### Response

```json
{
  "prediction": "Canceled",
  "cancellation_probability": 0.7936
}
```

---

## Tech Stack

* **Language:** Python
* **Data Analysis:** Pandas, NumPy
* **Visualization:** Matplotlib
* **Machine Learning:** Scikit-Learn
* **Boosting Libraries:** XGBoost, LightGBM, CatBoost
* **Model Explainability:** SHAP
* **Model Serialization:** Joblib
* **API Framework:** FastAPI
* **Deployment:** Uvicorn
* **Version Control:** Git & GitHub

---

## Project Structure

```text
hotel-booking-cancellation/
│
├── api/
│   └── app.py
├── data/
├── models/
│   └── hotel_cancellation_model.pkl
├── notebooks/
│   └── 01_eda.ipynb
├── images/
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Running the Project Locally

### Clone the repository

```bash
git clone https://github.com/your-username/hotel-booking-cancellation-prediction.git
cd hotel-booking-cancellation-prediction
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Start the API

```bash
uvicorn api.app:app --reload
```

Open your browser and visit:

```text
http://127.0.0.1:8000/docs
```

to access the interactive Swagger documentation.

---

## Future Improvements

Some possible directions for extending this project:

* Hyperparameter tuning using Optuna.
* Dockerizing the application.
* Cloud deployment with Render or AWS.
* Building a simple Streamlit frontend.
* Adding model monitoring and drift detection.
* Automating retraining with an MLOps pipeline.

---

## Final Thoughts

The objective of this project was not only to train a high-performing machine learning model, but also to build a complete and reproducible ML workflow. It covers data analysis, feature engineering, model comparison, explainability, and deployment, closely reflecting the stages involved in real-world machine learning projects.

## Live API

URL: <https://hotel-booking-api-6jux.onrender.com>
Swagger Docs: <https://hotel-booking-api-6jux.onrender.com/docs>
