import streamlit as st
from model_loader import load_models

st.set_page_config(
    page_title="AI Pre-Delinquency Engine",
    page_icon="💳",
    layout="wide"
)

@st.cache_resource
def load_all_models():
    return load_models("content/models")

models = load_all_models()

# Make models available to all pages via session state
st.session_state["models"] = models

st.title(" AI-Powered Pre-Delinquency Engine")

st.markdown("""
### Hybrid AI Platform for Credit Risk Management

Predict customer delinquency risk **before** it happens using a production-grade ensemble:

| Model | Role |
|---|---|
| **XGBoost** | Tree ensemble — tabular features |
| **LightGBM** | Tree ensemble — gradient boosting |
| **CatBoost** | Tree ensemble — categorical handling |
| **LSTM** | Sequential temporal modelling |
| **Calibrator** | Probability calibration (Platt scaling) |
| **SHAP** | Model explainability |
| **Expected Loss Engine** | EAD × LGD × PD |

---

### 🗺️ Navigation Guide

Use the **sidebar** to move between pages:

1. ** Single Customer** — Predict risk for one customer (enter data manually)
2. ** Batch Prediction** — Upload a CSV and score your entire portfolio
3. ** SHAP Explainability** — Understand what drives each customer's risk
4. ** Portfolio Risk Analytics** — Visual analytics across the portfolio
""")

st.success(" Models loaded successfully. Use the sidebar to navigate.")

# Show model config summary
with st.expander(" Model Configuration"):
    cfg = models.get("config", {})
    st.json(cfg)