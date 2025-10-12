# tools/DeploymentSagemaker.py
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch, joblib
from scipy.special import expit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ---- Project bits
from utils.constants import (
    DATA_DIR, MODELS_DIR, DEVICE,
    METER_ID_COL, TIMESTAMP_COL, LEAK_FLAG_COL,
    CLS_FEATURE_COLS,
)
from utils.DataAggregation import (
    build_aggregated_dataset, time_based_group_split
)

# Try both module name variants just in case your tree uses different casing

from Algorithms.Gru import GRUHead


# ---------------- load model/scaler/meta ----------------
def load_meta_scaler_model():
    meta_path = MODELS_DIR / "gru_cls_meta.json"
    thr_path  = MODELS_DIR / "gru_cls_thr.txt"
    scaler_path = MODELS_DIR / "gru_cls_scaler.joblib"
    state_path  = MODELS_DIR / "gru_cls.pt"

    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing scaler at {scaler_path}")
    if not state_path.exists():
        raise FileNotFoundError(f"Missing model state_dict at {state_path}")

    scaler = joblib.load(scaler_path)

    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        thr = float(thr_path.read_text().strip()) if thr_path.exists() else 0.5
        meta = {
            "model": "GRUHead", "task": "cls",
            "thr": thr, "window": 48, "stride": 1,
            "in_dim": len(CLS_FEATURE_COLS),
            "feature_cols": list(CLS_FEATURE_COLS),
            "hidden": 64, "num_layers": 2, "bidirectional": False, "dropout": 0.2,
        }

    feature_cols = list(meta.get("feature_cols", CLS_FEATURE_COLS))

    # Ensure scaler was fit on the same columns and order
    if hasattr(scaler, "feature_names_in_"):
        scaler_cols = list(scaler.feature_names_in_)
        if scaler_cols != feature_cols:
            raise ValueError(
                f"Scaler was fit on {scaler_cols}, but meta feature_cols={feature_cols}. "
                "Align feature order or re-train."
            )

    # Build model from meta
    model = GRUHead(
        in_dim=int(meta.get("in_dim", len(feature_cols))),
        hidden=int(meta["hidden"]),
        num_layers=int(meta["num_layers"]),
        bidirectional=bool(meta["bidirectional"]),
        dropout=float(meta["dropout"]),
    ).to(DEVICE)
    state = torch.load(state_path, map_location=DEVICE)
    model.load_state_dict(state); model.eval()

    return meta, scaler, model

def load_frozen_test_df(test_path: Path) -> pd.DataFrame:
    if not test_path.exists():
        raise FileNotFoundError(
            f"Frozen test file not found: {test_path}\n"
            "Create it during training (save test_agg.parquet after the split)."
        )
    df = pd.read_parquet(test_path)
    # Basic sanity checks:
    missing = [c for c in CLS_FEATURE_COLS + [METER_ID_COL, TIMESTAMP_COL, LEAK_FLAG_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Frozen test is missing columns: {missing}")
    return df
# ---------------- windowing over test with META kept ----------------
def build_test_windows_with_meta(test_df: pd.DataFrame, feature_cols: list[str], window: int, stride: int = 1):
    """
    Δίνει windows [window,F] και meta του ΤΕΛΕΥΤΑΙΟΥ timestep:
      meter_id, timestamp, hour, y_true.
    Προϋπόθεση: test_df έχει τις feature_cols (raw, unscaled) + meta + Leak_Flag.
    """
    test_df = test_df.sort_values([METER_ID_COL, TIMESTAMP_COL]).reset_index(drop=True)
    X_windows, meta_rows = [], []
    for m, g in test_df.groupby(METER_ID_COL, sort=False):
        if len(g) < window: continue
        for end in range(window-1, len(g), stride):
            sl = g.iloc[end-window+1:end+1]
            X_windows.append(sl[feature_cols].to_numpy(dtype=np.float32))  # raw here; we will scale later
            meta_rows.append({
                "Meter_ID": m,
                "Timestamp_end": sl.iloc[-1][TIMESTAMP_COL],
                "Hour": int(sl.iloc[-1]["Hour"]) if "Hour" in sl.columns else -1,
                "y_true": int(sl.iloc[-1][LEAK_FLAG_COL]),
            })
    return np.array(X_windows, dtype=np.float32), pd.DataFrame(meta_rows)


# ---------------- batched inference on windows (last-step) ----------------
def score_windows_laststep(model, scaler, X_windows_raw: np.ndarray, feature_cols: list[str], thr: float, batch: int = 512):
    """
    Παίρνει windows σε RAW feature space, κάνει scaler.transform ανά batch,
    περνάει από GRU, κρατάει sigmoid(last_logit) και binary pred με thr.
    """
    n = len(X_windows_raw)
    probs = np.zeros(n, dtype=np.float32)
    preds = np.zeros(n, dtype=np.int32)
    for i in range(0, n, batch):
        sl = X_windows_raw[i:i+batch]                    # [B,window,F]
        B, W, F = sl.shape
        # reshape -> scale -> reshape
        flat = sl.reshape(B*W, F)
        flat_scaled = scaler.transform(pd.DataFrame(flat, columns=feature_cols))
        xb = torch.tensor(flat_scaled.reshape(B, W, F), dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            logits_bt = model(xb).cpu().numpy()          # [B, W]
        last = logits_bt[:, -1]
        p = expit(np.clip(last, -20, 20))
        probs[i:i+batch] = p
        preds[i:i+batch] = (p >= thr).astype(np.int32)
    return probs, preds


# ---------------- build combined table from held-out test ----------------
def build_combined_table(meta, scaler, model, test_df: pd.DataFrame, limit_each: int = 12, seed: int = 123):
    window = int(meta["window"])
    feature_cols = list(meta["feature_cols"])
    thr = float(meta["thr"])

    # 1) Windows από το ΗΔΗ έτοιμο test (frozen)
    Xw_raw, meta_df = build_test_windows_with_meta(test_df, feature_cols, window=window, stride=1)
    if len(Xw_raw) == 0:
        return pd.DataFrame(columns=["Meter_ID","Timestamp_end","Hour","y_true","prob","pred","scenario"]), \
               pd.DataFrame()

    # 2) Score (persisted scaler + model)
    probs, preds = score_windows_laststep(model, scaler, Xw_raw, feature_cols, thr)

    # 3) Συνένωση
    out = meta_df.copy()
    out["prob"] = probs
    out["pred"] = preds
    out["scenario"] = np.where(out["y_true"]==1, "LEAK", "NORMAL")

    # 4) Balanced **random** sample για εμφάνιση (όχι nlargest/nsmallest!)
    rng = np.random.default_rng(seed)
    leak_df = out[out["y_true"]==1]
    norm_df = out[out["y_true"]==0]

    leak_k = min(limit_each, len(leak_df))
    norm_k = min(limit_each, len(norm_df))

    leak_part = leak_df.sample(n=leak_k, random_state=seed) if leak_k > 0 else leak_df
    norm_part = norm_df.sample(n=norm_k, random_state=seed) if norm_k > 0 else norm_df

    display_df = pd.concat([norm_part, leak_part], ignore_index=True)
    display_df = display_df.sort_values(["scenario","Timestamp_end"]).reset_index(drop=True)
    return out, display_df



# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit_each", type=int, default=12, help="πόσες γραμμές να δείξουμε ανά κλάση (NORMAL/LEAK)")
    ap.add_argument("--save_csv", type=str, default="", help="αν δώσεις μονοπάτι, θα σωθεί πλήρης πίνακας (όλα τα windows)")
    ap.add_argument("--thr", type=float, default=None, help="override threshold (αλλιώς meta)")
    ap.add_argument("--test_path", type=str,
                    default=str(MODELS_DIR / "artifacts/test_agg.parquet"),
                    help="path to frozen held-out test parquet")
    ap.add_argument("--seed", type=int, default=123, help="random seed for sampling view")
    args = ap.parse_args()

    # Load model/scaler/meta
    meta, scaler, model = load_meta_scaler_model()
    if args.thr is not None:
        meta["thr"] = float(args.thr)
    thr = float(meta["thr"])

    # Load frozen TEST set (αυτό που αποθήκευσες στο training)
    test_df = load_frozen_test_df(Path(args.test_path))

    # Φτιάξε full πίνακα + balanced random sample για εμφάνιση
    full_table, sample_table = build_combined_table(
        meta, scaler, model, test_df=test_df, limit_each=args.limit_each, seed=args.seed
    )
    if isinstance(full_table, pd.DataFrame) and full_table.empty:
        print("No test windows available. Check your data.")
        return

    # --- SAMPLE VIEW (NORMAL + LEAK μαζί) ---
    print("\n=== MIXED SAMPLE (from held-out TEST) ===")
    print("Meter_ID | Timestamp_end        | Hour | y_true |  prob  | pred")
    if not sample_table.empty:
        for _, r in sample_table.iterrows():
            ts = r['Timestamp_end']
            print(f"{str(r['Meter_ID'])} | {ts} | {int(r['Hour']):02d}  |   {int(r['y_true'])}    | {r['prob']:6.3f} |  {int(r['pred'])}")
    else:
        print("[empty sample] try lowering --limit_each")

    # --- SAMPLE METRICS (μόνο στο δείγμα που τυπώθηκε) ---
    if not sample_table.empty:
        y_true_s = sample_table["y_true"].to_numpy()
        y_pred_s = sample_table["pred"].to_numpy()
        acc = accuracy_score(y_true_s, y_pred_s)
        prec = precision_score(y_true_s, y_pred_s, zero_division=0)
        rec = recall_score(y_true_s, y_pred_s, zero_division=0)
        f1 = f1_score(y_true_s, y_pred_s, zero_division=0)
        print(f"\n[SAMPLE METRICS] acc={acc:.3f} | prec={prec:.3f} | rec={rec:.3f} | f1={f1:.3f} | thr={thr:.3f}")
    else:
        print("\n[SAMPLE METRICS] (no sample rows)")

    # --- FULL-TEST METRICS (σε ΟΛΑ τα test windows) ---
    y_true_full = full_table["y_true"].to_numpy()
    y_pred_full = full_table["pred"].to_numpy()
    acc_f = accuracy_score(y_true_full, y_pred_full)
    prec_f = precision_score(y_true_full, y_pred_full, zero_division=0)
    rec_f = recall_score(y_true_full, y_pred_full, zero_division=0)
    f1_f = f1_score(y_true_full, y_pred_full, zero_division=0)
    print(f"\n[FULL-TEST METRICS] acc={acc_f:.3f} | prec={prec_f:.3f} | rec={rec_f:.3f} | f1={f1_f:.3f} | thr={thr:.3f}")

    # --- Class counts ---
    n_total = len(full_table)
    n_pos = int(full_table['y_true'].sum())
    n_neg = n_total - n_pos
    print(f"[FULL-TEST COUNTS] total_windows={n_total} | leaks={n_pos} | normal={n_neg}")

    # --- Confusion counts ---
    tp = int(((full_table['y_true']==1) & (full_table['pred']==1)).sum())
    fp = int(((full_table['y_true']==0) & (full_table['pred']==1)).sum())
    tn = int(((full_table['y_true']==0) & (full_table['pred']==0)).sum())
    fn = int(((full_table['y_true']==1) & (full_table['pred']==0)).sum())
    print(f"[CONFUSION] TP={tp} | FP={fp} | TN={tn} | FN={fn}")

    # --- Mean prob by class ---
    mean_prob_pos = full_table.loc[full_table['y_true']==1, 'prob'].mean()
    mean_prob_neg = full_table.loc[full_table['y_true']==0, 'prob'].mean()
    print(f"[CLASS PROB] mean_prob_leak={mean_prob_pos:.3f} | mean_prob_normal={mean_prob_neg:.3f}")

    # --- Near-threshold examples (top-10 by |prob - thr|) ---
    near = full_table.iloc[(full_table["prob"] - thr).abs().argsort()[:10]]
    print("\n[NEAR-THRESHOLD EXAMPLES] top-10 by |prob - thr|")
    for _, r in near.iterrows():
        print(f"{r['Meter_ID']} | {r['Timestamp_end']} | y={int(r['y_true'])} | prob={r['prob']:.3f} | pred={int(r['pred'])}")

    # --- Save full table if requested ---
    if args.save_csv:
        out_path = Path(args.save_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        full_table.to_csv(out_path, index=False)
        print(f"\n[Saved] Full predictions table written to: {out_path}")


if __name__ == "__main__":
    main()
