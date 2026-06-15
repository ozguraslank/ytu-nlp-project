import argparse
import json

import numpy as np
from catboost import CatBoostClassifier
from scipy.stats import linregress
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             roc_auc_score)
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def moving_average(x, w):
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        w = max(1, len(x))
    return np.convolve(x, np.ones(w) / w, mode="valid")


def extract_features(top10_sums, window=10):
    """Features describing the shape of the moving-average curve."""
    raw = np.asarray(top10_sums, dtype=float)
    ma = moving_average(raw, window)
    n = len(ma)
    t = np.arange(n)

    thirds = np.array_split(ma, 3)
    slope = linregress(t, ma).slope if n > 1 else 0.0
    diffs = np.diff(ma) if n > 1 else np.array([0.0])

    feats = {
        # global statistics of the curve
        "mean": ma.mean(),
        "std": ma.std(),
        "min": ma.min(),
        "max": ma.max(),
        "median": np.median(ma),
        "p05": np.percentile(ma, 5),
        "p25": np.percentile(ma, 25),
        "p75": np.percentile(ma, 75),
        "slope": slope,                                   # shape / trend
        "first_third_mean": thirds[0].mean(),
        "mid_third_mean": thirds[1].mean(),
        "last_third_mean": thirds[2].mean(),
        "last_minus_first": thirds[2].mean() - thirds[0].mean(),
        "argmin_rel": float(np.argmin(ma)) / n,           # where the worst dip is
        "max_drop": float(diffs.min()),                   # sharpest local drop
        "roughness": float(np.abs(diffs).mean()),         # local volatility
        "frac_below_090": float((ma < 0.90).mean()),
        "frac_below_095": float((ma < 0.95).mean()),
        "frac_below_099": float((ma < 0.99).mean()),
        "area_above_curve": float((1.0 - ma).sum()),      # total uncertainty
        "length": float(len(raw)),                        # length of the answer (in tokens)  
        "log_length": float(np.log1p(len(raw))),
    }
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="answers.jsonl")
    ap.add_argument("--window", type=int, default=15)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]

    feat_names = None
    by_q = {}   # qid -> {"X": [...], "y": [], "split": str}
    for r in rows:
        qid = r["question_id"]
        split = r.get("split", None)
        if split not in ("train", "test"):
            continue  # skip anything not labeled as train/test
        f = extract_features(r["top10_prob_sums"], args.window)
        if feat_names is None:
            feat_names = list(f.keys())
        by_q.setdefault(qid, {"X": [], "y": [], "split": split})
        by_q[qid]["X"].append([f[k] for k in feat_names])
        by_q[qid]["y"].append(r["label"])
        # If there is any ambiguity about split, keep the first

    def question_relative(M):
        M = np.asarray(M, dtype=float)
        mu, sd = M.mean(axis=0), M.std(axis=0) + 1e-9
        z = (M - mu) / sd
        rank = np.argsort(np.argsort(M, axis=0), axis=0) / (len(M) - 1)
        return np.hstack([M, z, rank])

    Xtr, ytr, Xte, yte = [], [], [], []
    n_train_q, n_test_q = 0, 0
    for qid, d in by_q.items():
        Xq = question_relative(d["X"])
        if d["split"] == "train":
            Xtr.append(Xq); ytr.extend(d["y"]); n_train_q += 1
        elif d["split"] == "test":
            Xte.append(Xq); yte.extend(d["y"]); n_test_q += 1

    Xtr, ytr = np.vstack(Xtr), np.array(ytr)
    Xte, yte = np.vstack(Xte), np.array(yte)
    all_names = (feat_names
                 + [f"{k}__zq" for k in feat_names]
                 + [f"{k}__rankq" for k in feat_names])

    print(f"Train: {len(ytr)} answers ({n_train_q} questions), "
          f"correct ratio {ytr.mean():.3f}")
    print(f"Test : {len(yte)} answers ({n_test_q} questions), "
          f"correct ratio {yte.mean():.3f}")

    models = {
        "XGBoost": XGBClassifier(
            random_state=42,
            n_jobs=-1),
        "CatBoost": CatBoostClassifier(
            random_seed=42,
            verbose=False),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(random_state=42),
        "LGBM": LGBMClassifier(random_state=42, verbose=-1)
   
    }

    for name, model in models.items():
        model.fit(Xtr, ytr)
        proba = model.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.5).astype(int)
        print(f"=== {name} ===")
        print(f"Accuracy : {accuracy_score(yte, pred):.4f}")
        print(f"F1       : {f1_score(yte, pred):.4f}")
        print(f"ROC-AUC  : {roc_auc_score(yte, proba):.4f}")
        print(classification_report(
            yte, pred, target_names=["incorrect", "correct"], digits=3))

    gb = models["GradientBoosting"]
    imp = sorted(zip(all_names, gb.feature_importances_),
                 key=lambda x: -x[1])[:10]
    print("Top-10 most informative graph features (GradientBoosting):")
    for k, v in imp:
        print(f"  {k:>24s}  {v:.3f}")


if __name__ == "__main__":
    main()