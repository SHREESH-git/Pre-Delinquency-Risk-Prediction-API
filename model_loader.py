import joblib
import torch
import json
import os
from catboost import CatBoostClassifier


class LSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.dropout = torch.nn.Dropout(0.2)
        self.fc = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = torch.sigmoid(self.fc(out))
        return out.squeeze()


def load_models(model_path="models"):

    xgb = joblib.load(os.path.join(model_path, "xgb_model.pkl"))
    lgb = joblib.load(os.path.join(model_path, "lgb_model.pkl"))

    cat = CatBoostClassifier()
    cat.load_model(os.path.join(model_path, "cat_model.cbm"))

    scaler = joblib.load(os.path.join(model_path, "lstm_scaler.pkl"))
    calibrator = joblib.load(os.path.join(model_path, "calibrator.pkl"))
    tree_feature_cols = joblib.load(os.path.join(model_path, "tree_feature_columns.pkl"))
    lstm_feature_cols = joblib.load(os.path.join(model_path, "lstm_feature_columns.pkl"))

    #  LOAD JSON 
    with open(os.path.join(model_path, "hybrid_config.json")) as f:
        full_config = json.load(f)

    # Extract nested configs
    model_info = full_config["model_info"]
    tree_cfg = full_config["tree_ensemble"]
    hybrid_cfg = full_config["hybrid_blend"]
    calib_cfg = full_config["calibration"]
    el_cfg = full_config["expected_loss_engine"]
    portfolio_cfg = full_config["portfolio_assumption"]

    #  BUILD LSTM 
    lstm = LSTMModel(
        input_size=len(lstm_feature_cols),
        hidden_size=model_info["hidden_size"]
    )

    lstm.load_state_dict(
        torch.load(os.path.join(model_path, "lstm_model_state.pt"),
                   map_location="cpu")
    )
    lstm.eval()

    return {
        "xgb": xgb,
        "lgb": lgb,
        "cat": cat,
        "lstm": lstm,
        "scaler": scaler,
        "calibrator": calibrator,
        "tree_feature_cols": tree_feature_cols,
        "lstm_feature_cols": lstm_feature_cols,

        # tree ensemble weights
        "wx": tree_cfg["wx"],
        "wl": tree_cfg["wl"],
        "wc": tree_cfg["wc"],

        # hybrid blending
        "best_weight": hybrid_cfg["best_tree_weight"],
        "seq_len": model_info["sequence_length"],

        # calibration
        "pd_floor": calib_cfg["pd_floor"],
        "pd_cap": calib_cfg["pd_cap"],

        # EL engine
        "ead_rules": el_cfg["ead_rules"],
        "lgd_rules": el_cfg["lgd_rules"],

        # portfolio assumption
        "real_world_default_rate": portfolio_cfg["real_world_default_rate"],

        # full config backup
        "config": full_config
    }