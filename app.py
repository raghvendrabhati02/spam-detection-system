import os
import sys
import warnings
from typing import List, Optional
import pandas as pd
import numpy as np

from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, Response, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from uvicorn import run as app_run

from src.pipeline.prediction_pipeline import PredictionPipeline
from src.pipeline.train_pipeline import TrainPipeline
from src.constant.application import APP_HOST, APP_PORT
from src.logger import logging

warnings.filterwarnings('ignore')

app = FastAPI(
    title="PrecisionCustomer API",
    description="AI-Powered Customer Segmentation and Personality Categorization System",
    version="1.0.0"
)

# Template and Static files setup
templates = Jinja2Templates(directory='templates')
os.makedirs("static/css", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic schemas for API validation
class CustomerFeatures(BaseModel):
    Age: int = Field(..., ge=18, le=120, description="Customer age")
    Education: int = Field(..., ge=0, le=4, description="Education level (0-4)")
    Marital_Status: int = Field(..., ge=0, le=1, description="Married/Partner status (0-1)")
    Parental_Status: int = Field(..., ge=0, le=1, description="Parental status (0-1)")
    Children: int = Field(..., ge=0, le=10, description="Number of children in household")
    Income: float = Field(..., ge=0, description="Yearly customer income")
    Total_Spending: float = Field(..., ge=0, description="Total customer spending in 2 years")
    Days_as_Customer: int = Field(..., ge=0, description="Days as a registered customer")
    Recency: int = Field(..., ge=0, description="Days since last purchase")
    Wines: int = Field(..., ge=0, description="Amount spent on wines")
    Fruits: int = Field(..., ge=0, description="Amount spent on fruits")
    Meat: int = Field(..., ge=0, description="Amount spent on meat")
    Fish: float = Field(..., ge=0, description="Amount spent on fish")
    Sweets: int = Field(..., ge=0, description="Amount spent on sweets")
    Gold: float = Field(..., ge=0, description="Amount spent on gold products")
    Web: int = Field(..., ge=0, description="Number of website purchases")
    Catalog: int = Field(..., ge=0, description="Number of catalog purchases")
    Store: int = Field(..., ge=0, description="Number of store purchases")
    Discount_Purchases: int = Field(..., ge=0, description="Number of purchases with discount")
    Total_Promo: int = Field(..., ge=0, description="Accepted campaign offers")
    NumWebVisitsMonth: int = Field(..., ge=0, description="Website visits in last month")


def get_cluster_persona(cluster_id: int) -> dict:
    """
    Returns marketing details for the predicted customer cluster
    """
    personas = {
        0: {
            "name": "Value-Seeking Family",
            "description": "Budget-conscious families with children. They have moderate income and moderate spending habits. Highly responsive to discounts, deals, and promotional campaigns.",
            "spending_tier": "Medium-Low",
            "strategy": "Target with coupons, seasonal sales, family-oriented bundle offers, and discount catalogs."
        },
        1: {
            "name": "Affluent Shopper",
            "description": "High-income individuals or couples with no children. Extremely high spending, particularly on premium items like wines, meat, and gold. They prioritize quality over price.",
            "spending_tier": "High",
            "strategy": "Target with premium loyalty clubs, exclusive events, high-end catalog offerings, and gourmet product recommendations."
        },
        2: {
            "name": "Young Frugal Starter",
            "description": "Younger demographics or new customers with low incomes and very low spending across all categories. They browse the website frequently but make small, essential purchases.",
            "spending_tier": "Low",
            "strategy": "Engage via digital web marketing, app notifications, low-threshold introductory rewards, and high-frequency low-cost campaigns."
        }
    }
    return personas.get(cluster_id, {
        "name": "General Cohort",
        "description": "Standard retail customer with mixed shopping behaviors.",
        "spending_tier": "Mixed",
        "strategy": "Standard general newsletter and basic retention campaigns."
    })


# Exception handler middleware for API routes
@app.middleware("http")
async def exception_handling_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        logging.error(f"Unhandled server error: {e}", exc_info=True)
        if request.url.path.startswith("/api/"):
            return JSONResponse(
                status_code=500,
                content={"status": False, "error": str(e)}
            )
        return Response(f"Internal Server Error: {e}", status_code=500)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "precision-customer-segmentation"}


@app.get("/train")
async def train_route():
    try:
        logging.info("Starting model training pipeline...")
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        return {"status": True, "message": "Model training completed and pushed successfully."}
    except Exception as e:
        return {"status": False, "error": str(e)}


@app.get("/")
@app.get("/predict")
async def predict_get_route(request: Request):
    return templates.TemplateResponse(
        "customer.html",
        {"request": request, "context": "Rendering", "persona": None}
    )


@app.post("/")
@app.post("/predict")
async def predict_form_route(
    request: Request,
    Age: int = Form(...),
    Education: int = Form(...),
    Marital_Status: int = Form(...),
    Parental_Status: int = Form(...),
    Children: int = Form(...),
    Income: float = Form(...),
    Total_Spending: float = Form(...),
    Days_as_Customer: int = Form(...),
    Recency: int = Form(...),
    Wines: int = Form(...),
    Fruits: int = Form(...),
    Meat: int = Form(...),
    Fish: float = Form(...),
    Sweets: int = Form(...),
    Gold: float = Form(...),
    Web: int = Form(...),
    Catalog: int = Form(...),
    Store: int = Form(...),
    Discount_Purchases: int = Form(...),
    Total_Promo: int = Form(...),
    NumWebVisitsMonth: int = Form(...)
):
    try:
        input_data = [
            Age, Education, Marital_Status, Parental_Status, Children, Income,
            Total_Spending, Days_as_Customer, Recency, Wines, Fruits, Meat,
            Fish, Sweets, Gold, Web, Catalog, Store, Discount_Purchases,
            Total_Promo, NumWebVisitsMonth
        ]
        
        prediction_pipeline = PredictionPipeline()
        prediction = prediction_pipeline.run_pipeline(input_data=input_data)
        cluster_id = int(prediction[0])
        persona_details = get_cluster_persona(cluster_id)
        
        return templates.TemplateResponse(
            "customer.html",
            {
                "request": request, 
                "context": cluster_id, 
                "persona": persona_details,
                "input_values": {
                    "Age": Age, "Education": Education, "Marital_Status": Marital_Status,
                    "Parental_Status": Parental_Status, "Children": Children, "Income": Income,
                    "Total_Spending": Total_Spending, "Days_as_Customer": Days_as_Customer,
                    "Recency": Recency, "Wines": Wines, "Fruits": Fruits, "Meat": Meat,
                    "Fish": Fish, "Sweets": Sweets, "Gold": Gold, "Web": Web,
                    "Catalog": Catalog, "Store": Store, "Discount_Purchases": Discount_Purchases,
                    "Total_Promo": Total_Promo, "NumWebVisitsMonth": NumWebVisitsMonth
                }
            }
        )
    except Exception as e:
        logging.error(f"Prediction form error: {e}")
        return templates.TemplateResponse(
            "customer.html",
            {"request": request, "context": "Error", "error_msg": str(e), "persona": None}
        )


@app.post("/api/predict")
async def api_predict(customer: CustomerFeatures):
    try:
        input_list = [
            customer.Age, customer.Education, customer.Marital_Status,
            customer.Parental_Status, customer.Children, customer.Income,
            customer.Total_Spending, customer.Days_as_Customer, customer.Recency,
            customer.Wines, customer.Fruits, customer.Meat, customer.Fish,
            customer.Sweets, customer.Gold, customer.Web, customer.Catalog,
            customer.Store, customer.Discount_Purchases, customer.Total_Promo,
            customer.NumWebVisitsMonth
        ]
        
        prediction_pipeline = PredictionPipeline()
        prediction = prediction_pipeline.run_pipeline(input_data=input_list)
        cluster_id = int(prediction[0])
        persona_details = get_cluster_persona(cluster_id)
        
        return {
            "status": True,
            "prediction": {
                "cluster": cluster_id,
                "persona": persona_details
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/demo")
async def api_demo_data():
    """
    Exposes realistic demo profiles to allow quick population of the form
    """
    demo_profiles = {
        "value_shopper": {
            "Age": 42, "Education": 2, "Marital_Status": 1, "Parental_Status": 1, "Children": 2,
            "Income": 45000.0, "Total_Spending": 350.0, "Days_as_Customer": 600, "Recency": 12,
            "Wines": 120, "Fruits": 30, "Meat": 80, "Fish": 20.0, "Sweets": 40, "Gold": 60.0,
            "Web": 4, "Catalog": 1, "Store": 5, "Discount_Purchases": 4, "Total_Promo": 1, "NumWebVisitsMonth": 7
        },
        "premium_aficionado": {
            "Age": 51, "Education": 4, "Marital_Status": 1, "Parental_Status": 0, "Children": 0,
            "Income": 85000.0, "Total_Spending": 1600.0, "Days_as_Customer": 1100, "Recency": 8,
            "Wines": 850, "Fruits": 120, "Meat": 450, "Fish": 80.0, "Sweets": 60, "Gold": 40.0,
            "Web": 8, "Catalog": 6, "Store": 12, "Discount_Purchases": 1, "Total_Promo": 4, "NumWebVisitsMonth": 2
        },
        "frugal_starter": {
            "Age": 28, "Education": 2, "Marital_Status": 0, "Parental_Status": 0, "Children": 0,
            "Income": 25000.0, "Total_Spending": 45.0, "Days_as_Customer": 120, "Recency": 45,
            "Wines": 10, "Fruits": 5, "Meat": 15, "Fish": 5.0, "Sweets": 5, "Gold": 5.0,
            "Web": 1, "Catalog": 0, "Store": 2, "Discount_Purchases": 1, "Total_Promo": 0, "NumWebVisitsMonth": 8
        }
    }
    return demo_profiles


@app.post("/api/predict/file")
async def api_predict_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")
    
    try:
        df = pd.read_csv(file.file)
        prediction_pipeline = PredictionPipeline()
        
        # Verify schema matches prediction schema keys
        # We need the 21 columns
        required_cols = [
            "Age", "Education", "Marital Status", "Parental Status", "Children",
            "Income", "Total_Spending", "Days_as_Customer", "Recency", "Wines",
            "Fruits", "Meat", "Fish", "Sweets", "Gold", "Web", "Catalog",
            "Store", "Discount Purchases", "Total Promo", "NumWebVisitsMonth"
        ]
        
        # Adjust input df columns names if they have underscores instead of spaces
        column_mapping = {col.replace(" ", "_"): col for col in required_cols}
        df.rename(columns=column_mapping, inplace=True)
        
        # Check missing columns
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise Exception(f"Uploaded CSV is missing columns: {missing_cols}")
            
        predictions = []
        for index, row in df[required_cols].iterrows():
            input_list = list(row)
            pred = prediction_pipeline.run_pipeline(input_data=input_list)
            predictions.append(int(pred[0]))
            
        df["predicted_cluster"] = predictions
        df["persona_name"] = [get_cluster_persona(cid)["name"] for cid in predictions]
        
        output_file = "batch_predictions.csv"
        df.to_csv(output_file, index=False)
        
        return FileResponse(
            path=output_file,
            filename="batch_predictions.csv",
            media_type="text/csv"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File prediction error: {str(e)}")


if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)
