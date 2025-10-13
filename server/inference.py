from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

# ---------------- GRU head (ίδια υλοποίηση με training) ----------------
class GRUHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, num_layers: int = 2,
                 bidirectional: bool = False, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.0 if num_layers == 1 else dropout
        )
        out_dim = hidden * (2 if bidirectional else 1)
        self.fc = nn.Linear(out_dim, 1)

    def forward(self, x):           # x: [B, T, F]
        out, _ = self.gru(x)        # [B, T, H or 2H]
        out = self.fc(out)          # [B, T, 1]
        return out.squeeze(-1)      # [B, T]

# ---------------- SageMaker entrypoints ----------------
def model_fn(model_dir: str):
    """
    Φορτώνει: meta (για in_dim & features & thr), scaler, GRU state_dict.
    Επιστρέφει dict με όλα όσα χρειάζεται το predict_fn.
    """
    meta_path   = os.path.join(model_dir, "gru_cls_meta.json")
    scaler_path = os.path.join(model_dir, "gru_cls_scaler.joblib")
    state_path  = os.path.join(model_dir, "gru_cls.pt")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    feature_cols = list(meta["feature_cols"])
    model = GRUHead(
        in_dim=int(meta.get("in_dim", len(feature_cols))),
        hidden=int(meta["hidden"]),
        num_layers=int(meta["num_layers"]),
        bidirectional=bool(meta["bidirectional"]),
        dropout=float(meta["dropout"]),
    )
    state = torch.load(state_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    scaler = joblib.load(scaler_path)

    # Threshold override από env αν έχει οριστεί (π.χ. στο endpoint config)
    thr_env = os.environ.get("THR_OVERRIDE", "")
    thr = float(thr_env) if thr_env else float(meta.get("thr", 0.5))

    return {"model": model, "scaler": scaler, "meta": meta, "thr": thr, "feature_cols": feature_cols}

def input_fn(request_body, content_type="application/json"):
    """
    Παίρνει JSON -> dict με numpy array X:[B,W,F], thr(optional), feature_cols(optional)
    """
    if content_type != "application/json":
        raise ValueError(f"Unsupported content_type: {content_type}")

    payload = json.loads(request_body)

    # windows: required
    if "windows" not in payload:
        raise ValueError("Missing 'windows' in payload.")
    X = np.array(payload["windows"], dtype=np.float32)  # [B,W,F]
    if X.ndim != 3:
        raise ValueError(f"'windows' must be 3D [B,W,F], got shape={X.shape}")

    # optional override thr per-request
    thr = payload.get("thr", None)

    # optional: feature order from client (αλλιώς χρησιμοποιούμε του meta)
    client_feats = payload.get("feature_cols", None)

    return {"X": X, "thr": thr, "client_feats": client_feats}

def predict_fn(inputs, model_artifacts):
    """
    Scale -> GRU -> sigmoid(last) -> prob/pred
    """
    X = inputs["X"]                   # [B,W,F] raw features
    thr_req = inputs["thr"]
    client_feats = inputs["client_feats"]

    model = model_artifacts["model"]
    scaler = model_artifacts["scaler"]
    meta = model_artifacts["meta"]
    thr = float(thr_req) if thr_req is not None else float(model_artifacts["thr"])

    feature_cols = model_artifacts["feature_cols"]
    # Αν ο client έστειλε feature_cols, διασφάλισε ότι ταιριάζουν 1-1 με τα meta
    if client_feats is not None:
        if list(client_feats) != list(feature_cols):
            raise ValueError(f"Feature order mismatch. Expected {feature_cols}, got {client_feats}")

    B, W, F = X.shape
    if F != len(feature_cols):
        raise ValueError(f"Feature count mismatch: given F={F}, expected {len(feature_cols)}")

    # scale ανά batch
    flat = X.reshape(B * W, F)
    df = pd.DataFrame(flat, columns=feature_cols)
    flat_scaled = scaler.transform(df).astype(np.float32)
    xb = torch.from_numpy(flat_scaled.reshape(B, W, F))

    with torch.no_grad():
        logits_bt = model(xb).detach().numpy()    # [B,W]
    last = logits_bt[:, -1]                       # [B]

    # sigmoid χωρίς SciPy
    last = np.clip(last, -20.0, 20.0)
    probs = 1.0 / (1.0 + np.exp(-last))
    preds = (probs >= thr).astype(np.int32)

    return {"probs": probs.tolist(), "preds": preds.tolist(), "thr": thr}

def output_fn(prediction, accept="application/json"):
    if accept != "application/json":
        raise ValueError(f"Unsupported accept: {accept}")
    return json.dumps(prediction), accept
