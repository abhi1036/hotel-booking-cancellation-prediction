from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Load trained pipeline
model = joblib.load("models/hotel_cancellation_model.pkl")

# Create FastAPI app
app = FastAPI(
    title="Hotel Booking Cancellation Prediction API",
    version="1.0"
)

# Input schema
class BookingRequest(BaseModel):
    number_of_adults: int
    number_of_children: int
    number_of_weekend_nights: int
    number_of_week_nights: int
    type_of_meal: str
    car_parking_space: int
    room_type: str
    lead_time: int
    market_segment_type: str
    repeated: int
    P_C: int
    P_not_C: int
    average_price: float
    special_requests: int
    reservation_month: int
    reservation_dayofweek: int
    reservation_quarter: int
    is_weekend_booking: int

@app.get("/")
def home():
    return {
        "message": "Hotel Booking Cancellation Prediction API is running!"
    }


@app.post("/predict")
def predict(data: BookingRequest):

    input_df = pd.DataFrame(
        [{
            "number of adults": data.number_of_adults,
            "number of children": data.number_of_children,
            "number of weekend nights": data.number_of_weekend_nights,
            "number of week nights": data.number_of_week_nights,
            "type of meal": data.type_of_meal,
            "car parking space": data.car_parking_space,
            "room type": data.room_type,
            "lead time": data.lead_time,
            "market segment type": data.market_segment_type,
            "repeated": data.repeated,
            "P-C": data.P_C,
            "P-not-C": data.P_not_C,
            "average price": data.average_price,
            "special requests": data.special_requests,
            "reservation_month": data.reservation_month,
            "reservation_dayofweek": data.reservation_dayofweek,
            "reservation_quarter": data.reservation_quarter,
            "is_weekend_booking": data.is_weekend_booking
        }]
    )

    prediction = int(model.predict(input_df)[0])
    probability = float(
        model.predict_proba(input_df)[0][1]
    )

    result = (
        "Canceled"
        if prediction == 1
        else "Not_Canceled"
    )

    return {
        "prediction": result,
        "cancellation_probability": round(
            probability,
            4
        )
    }