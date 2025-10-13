from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

#region Model definition (same head used during training)
class GRUHead(nn.Module):
    """
    Minimal GRU -> Linear head for per-timestep logits.
    We use the last timestep's logit for binary classification.
    """
    def __init__(
        self,
        in_dim: int,
        hidden: int = 64,
        num_layers: int = 2,
        bidirectional: bool = False,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,         # inputs shaped [B, T, F]
            bidirectional=bidirectional,
            # Dropout is only applied between GRU layers when num_layers > 1
            dropout=0.0 if num_layers == 1 else dropout,
        )
        out_dim = hidden * (2 if bidirectional else 1)
        self.fc = nn.Linear(out_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: float tensor of shape [B, T, F]
        Returns:
            logits: float tensor of shape [B, T] (one logit per timestep)
        """
        out, _ = self.gru(x)   # [B, T, H or 2H]
        out = self.fc(out)     # [B, T, 1]
        return out.squeeze(-1) # [B, T]

#endregion



#region SageMaker entry points
def model_fn(model_dir: str):
    """
    Load all artifacts produced during training:
      - meta.json (feature order, window length, threshold, architecture params)
      - scaler (e.g., StandardScaler)
      - GRU weights (state_dict)

    Returns a dictionary with everything predict_fn needs.
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

    # Load weights on CPU (the serving container may or may not have CUDA)
    state = torch.load(state_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()  # important: disable dropout, etc.

    scaler = joblib.load(scaler_path)

    # Allow an endpoint-level threshold override (set in the endpoint env)
    thr_env = os.environ.get("THR_OVERRIDE", "")
    thr = float(thr_env) if thr_env else float(meta.get("thr", 0.5))

    return {
        "model": model,
        "scaler": scaler,
        "meta": meta,
        "thr": thr,
        "feature_cols": feature_cols,
    }


def input_fn(request_body, content_type="application/json"):
    """
    Parse and validate the request payload.

    Expected JSON keys:
      - 'windows': float list with shape [B, W, F]
      - 'feature_cols': list[str] describing feature order (optional but recommended)
      - 'thr': float threshold override (optional)
    """
    if content_type != "application/json":
        raise ValueError(f"Unsupported content_type: {content_type}")

    payload = json.loads(request_body)

    if "windows" not in payload:
        raise ValueError("Missing 'windows' in payload.")
    X = np.array(payload["windows"], dtype=np.float32)  # [B, W, F]
    if X.ndim != 3:
        raise ValueError(f"'windows' must be 3D [B, W, F], got shape={X.shape}")

    thr = payload.get("thr", None)                       # optional per-request override
    client_feats = payload.get("feature_cols", None)     # optional order enforcement

    return {"X": X, "thr": thr, "client_feats": client_feats}

def predict_fn(inputs, model_artifacts):
    """
    Scale -> GRU -> sigmoid(last timestep) -> probability -> threshold -> prediction.

    Note:
      - We scale *per-feature across the batch* using the training scaler.
      - We enforce the exact feature order used at training to avoid silent misalignment.
    """
    X = inputs["X"]                   # [B, W, F] raw float32 features
    thr_req = inputs["thr"]
    client_feats = inputs["client_feats"]

    model: GRUHead = model_artifacts["model"]
    scaler = model_artifacts["scaler"]
    thr = float(thr_req) if thr_req is not None else float(model_artifacts["thr"])

    feature_cols = model_artifacts["feature_cols"]

    # If the client supplies feature order, it MUST match the training order 1:1.
    if client_feats is not None:
        if list(client_feats) != list(feature_cols):
            raise ValueError(
                f"Feature order mismatch.\nExpected: {feature_cols}\nGot:      {client_feats}"
            )

    B, W, F = X.shape
    if F != len(feature_cols):
        raise ValueError(f"Feature count mismatch: got F={F}, expected {len(feature_cols)}")

    # Flatten [B, W, F] -> [B*W, F] to apply the training scaler feature-wise
    flat = X.reshape(B * W, F)
    df = pd.DataFrame(flat, columns=feature_cols)

    # Important: use the scaler fit on training data to maintain distributional alignment
    flat_scaled = scaler.transform(df).astype(np.float32)

    # Back to [B, W, F] for the GRU
    xb = torch.from_numpy(flat_scaled.reshape(B, W, F))

    # Inference: no grads needed
    with torch.no_grad():
        logits_bt = model(xb).detach().numpy()  # [B, W]

    # We classify based on the last timestep's logit (end-of-window)
    last_logits = logits_bt[:, -1]              # [B]

    # Sigmoid in a numerically safe way (clipping helps avoid overflow)
    last_logits = np.clip(last_logits, -20.0, 20.0)
    probs = 1.0 / (1.0 + np.exp(-last_logits))  # [B] in [0, 1]
    preds = (probs >= thr).astype(np.int32)     # [B] 0/1

    return {"probs": probs.tolist(), "preds": preds.tolist(), "thr": thr}

def output_fn(prediction, accept="application/json"):
    """Serialize the prediction dictionary back to JSON."""
    if accept != "application/json":
        raise ValueError(f"Unsupported accept: {accept}")
    return json.dumps(prediction), accept

#endregion
