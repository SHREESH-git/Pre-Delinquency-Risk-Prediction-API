import io
import base64
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import List

# Import your core pipeline logic (copied into the repo)
from model_loader import load_models
from feature_engineering import engineer_features
from predictor import prepare_inputs, predict_pd
from risk_engine import expected_loss, risk_bucket

# 1. Initialize FastAPI Application
app = FastAPI(
    title="EarlyShield Pre-Delinquency Engine API",
    description="Early warning system to detect customer financial stress weeks before default.",
    version="1.0.0"
)

# Load the models once on application startup
MODELS_DIR = "content/models"
try:
    models = load_models(MODELS_DIR)
except Exception as e:
    raise RuntimeError(f"Failed to load models from {MODELS_DIR}: {e}")


# 2. Define Request Validation Schemas (Pydantic)
class MonthlyRecord(BaseModel):
    customer_id: str = Field(..., example="CUST_001")
    month: str = Field(..., example="2026-05")
    customer_segment: str = Field(..., example="salaried")
    region_tier: str = Field(..., example="tier_1")
    product_type: str = Field(..., example="personal_loan")
    active_products_count: int = Field(..., ge=0, example=2)
    credit_card_utilization: float = Field(..., ge=0.0, le=1.0, example=0.45)
    total_monthly_obligation: float = Field(..., ge=0.0, example=15000)
    emi_amount: float = Field(..., ge=0.0, example=8000)
    days_to_emi: int = Field(..., ge=0, example=10)
    emi_to_income_ratio: float = Field(..., ge=0.0, le=1.0, example=0.30)
    salary_delay_days: int = Field(..., ge=0, example=2)
    weekly_balance_change_pct: float = Field(..., example=0.02)
    atm_withdrawal_amount: float = Field(..., ge=0.0, example=3000)
    monthly_income: float = Field(..., ge=1.0, example=35000)


class CustomerPredictionRequest(BaseModel):
    records: List[MonthlyRecord] = Field(
        ..., 
        description="Chronological historical records for a single customer (latest month last)."
    )


# 3. Define Endpoints

@app.get("/health", tags=["Status"])
def health_check():
    """Returns application health and configuration metadata."""
    return {
        "status": "healthy",
        "loaded_models": list(models.keys()),
        "blend_weight": models.get("best_weight", 0.6)
    }


@app.post("/predict", tags=["Inference"])
def predict_single_customer(request: CustomerPredictionRequest):
    """
    Accepts chronological customer history, computes engineered features,
    and runs the calibrated blended tree/LSTM ensemble to predict pre-delinquency risk.
    """
    if not request.records:
        raise HTTPException(status_code=400, detail="Record list cannot be empty.")

    try:
        # Convert Pydantic records to DataFrame
        df_raw = pd.DataFrame([r.dict() for r in request.records])
        
        # 1. Feature Engineering Pipeline
        df_engineered = engineer_features(df_raw)

        # 2. Category Handling (get_dummies)
        # Note: In production, align dummy columns explicitly with models['tree_feature_cols']
        cat_cols = ["customer_segment", "region_tier", "product_type"]
        df_engineered = pd.get_dummies(df_engineered, columns=[c for c in cat_cols if c in df_engineered])

        # 3. Model Scoring
        tree_input, lstm_input = prepare_inputs(models, df_engineered)
        pd_score = predict_pd(models, tree_input, lstm_input)

        # 4. Expected Loss Calculation (using latest month values)
        last_row = df_raw.iloc[-1]
        salary_flag = int(last_row["salary_delay_days"] > 5)
        util_flag = int(last_row["credit_card_utilization"] > 0.75)

        el, lgd, ead = expected_loss(
            models, 
            pd_score,
            last_row["emi_amount"],
            last_row["credit_card_utilization"],
            last_row["monthly_income"],
            salary_flag,
            util_flag
        )

        return {
            "customer_id": last_row["customer_id"],
            "probability_of_default": round(float(pd_score), 4),
            "risk_bucket": risk_bucket(pd_score),
            "expected_loss": round(float(el), 2),
            "lgd": round(float(lgd), 4),
            "ead": round(float(ead), 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/predict-batch", tags=["Inference"])
async def predict_batch_csv(file: UploadFile = File(...)):
    """
    Accepts a raw CSV file containing multi-customer portfolio history,
    runs the feature pipeline, and scores each customer in bulk.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    try:
        contents = await file.read()
        df_raw = pd.read_csv(io.BytesIO(contents))
        
        # Ensure customer history contains required columns
        required_cols = [col[0] for col in [
            ("customer_id", None), ("month", None), ("customer_segment", None), 
            ("region_tier", None), ("product_type", None)
        ]]
        for col in required_cols:
            if col not in df_raw.columns:
                raise HTTPException(status_code=400, detail=f"Missing required column: {col}")

        # Group by customer and score individually (since LSTM requires sequential client history)
        results = []
        for cust_id, group in df_raw.groupby("customer_id"):
            # Sort group chronologically
            group_sorted = group.sort_values("month")
            
            # Engineer features
            df_eng = engineer_features(group_sorted)
            cat_cols = ["customer_segment", "region_tier", "product_type"]
            df_eng = pd.get_dummies(df_eng, columns=[c for c in cat_cols if c in df_eng])

            # Predict
            tree_input, lstm_input = prepare_inputs(models, df_eng)
            pd_score = predict_pd(models, tree_input, lstm_input)

            last_row = group_sorted.iloc[-1]
            salary_flag = int(last_row["salary_delay_days"] > 5)
            util_flag = int(last_row["credit_card_utilization"] > 0.75)

            el, lgd, ead = expected_loss(
                models, pd_score,
                last_row["emi_amount"], last_row["credit_card_utilization"],
                last_row["monthly_income"], salary_flag, util_flag
            )

            results.append({
                "customer_id": cust_id,
                "probability_of_default": round(float(pd_score), 4),
                "risk_bucket": risk_bucket(pd_score),
                "expected_loss": round(float(el), 2),
                "lgd": round(float(lgd), 4),
                "ead": round(float(ead), 2)
            })

        return {"scored_portfolio": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch error: {str(e)}")


@app.post("/explain", tags=["Explainability"])
def explain_customer(request: CustomerPredictionRequest):
    """
    Computes SHAP explainability waterfall contribution values for the latest month
    of a customer's history. Returns the explanation as a Base64-encoded PNG image.
    """
    try:
        df_raw = pd.DataFrame([r.dict() for r in request.records])
        df_engineered = engineer_features(df_raw)
        
        # Align features
        tree_cols = models["tree_feature_cols"]
        df_engineered = pd.get_dummies(df_engineered)
        tree_input_all = df_engineered.reindex(columns=tree_cols, fill_value=0)
        tree_input = tree_input_all.iloc[[-1]]
        
        # LSTM input base logic
        _, lstm_input_base = prepare_inputs(models, df_engineered)

        # Vectorized ensemble wrapper function for SHAP explainer
        def ensemble_predict(X):
            df_X = pd.DataFrame(X, columns=tree_cols).reindex(columns=tree_cols, fill_value=0)
            n = len(df_X)
            
            # Trees
            px = models["xgb"].predict_proba(df_X)[:, 1]
            pl = models["lgb"].predict_proba(df_X)[:, 1]
            pc = models["cat"].predict_proba(df_X)[:, 1]
            tree_pd = (models["wx"] * px + models["wl"] * pl + models["wc"] * pc) / (models["wx"] + models["wl"] + models["wc"])
            
            # LSTM (replicate base sequence)
            lstm_batch = np.repeat(lstm_input_base, n, axis=0)
            device = next(models["lstm"].parameters()).device
            tensor_input = torch.tensor(lstm_batch, dtype=torch.float32).to(device)
            with torch.no_grad():
                lstm_pd = models["lstm"](tensor_input).cpu().numpy().reshape(-1)
            
            # Calibration
            blended = models.get("best_weight", 0.6) * tree_pd + (1 - models.get("best_weight", 0.6)) * lstm_pd
            calibrated = models["calibrator"].predict_proba(blended.reshape(-1, 1))[:, 1]
            return calibrated

        # Sample background and compute SHAP
        background = tree_input_all.sample(n=min(50, len(tree_input_all)), random_state=42)
        explainer = shap.KernelExplainer(ensemble_predict, background)
        shap_values = explainer.shap_values(tree_input)
        
        shap_arr = np.array(shap_values[0]) if isinstance(shap_values, list) else np.array(shap_values)
        shap_sample = shap_arr[0] if shap_arr.ndim == 2 else shap_arr

        # Generate Waterfall plot
        exp = shap.Explanation(
            values=shap_sample,
            base_values=explainer.expected_value,
            data=tree_input.iloc[0].values,
            feature_names=tree_cols
        )
        
        plt.figure(figsize=(10, 8))
        shap.plots.waterfall(exp, max_display=6, show=False)
        plt.title("PD Driver Contribution Analysis", fontsize=12, pad=15)
        
        # Save plot to buffer as base64 string
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()

        return {
            "mime": "image/png",
            "base64_data": img_str
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explainability error: {str(e)}")
