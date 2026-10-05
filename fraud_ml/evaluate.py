"""Step 1: compare models on the chronological validation split. The test split is counted, never scored."""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from .common import FEATURES, chronological_split, load_transactions, resample_weights, split_summary, versions, write_json
from .models import CHOSEN, candidates
from .plots import model_comparison


def bootstrap_ap(y, scores, n_boot, seed):
    """Average precision on resampled validation rows, using the same resamples for every model."""
    draws = []
    for weight in resample_weights(len(y), n_boot, seed):
        if (weight * y).any():  # Average precision is undefined without a fraud case.
            draws.append({key: average_precision_score(y, s, sample_weight=weight) for key, s in scores.items()})
    return pd.DataFrame(draws)


def default_threshold_counts(y, proba):
    """What predict() would flag: the 0.5 cut-off that the original script used."""
    tn, fp, fn, tp = confusion_matrix(y, proba >= .5, labels=[0, 1]).ravel()
    return {"alerts_at_0.5": int(tp + fp), "frauds_caught_at_0.5": int(tp), "false_alarms_at_0.5": int(fp),
            "precision_at_0.5": float(tp / (tp + fp)) if tp + fp else None, "recall_at_0.5": float(tp / (tp + fn))}


def run(args):
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    df, duplicates = load_transactions(args.data)
    labels, cuts = chronological_split(df, args.train_fraction, args.val_fraction)
    splits = split_summary(df, labels)
    splits.to_csv(out / "splits.csv", index=False)
    print(splits.to_string(index=False))
    train, val = df[labels == "train"], df[labels == "validation"]
    X_train, y_train = train[FEATURES].to_numpy(), train.Class.to_numpy()
    X_val, y_val = val[FEATURES].to_numpy(), val.Class.to_numpy()
    results, scores = [], {}
    for name, weighting, model in candidates(y_train, args.seed, args.threads):
        key = f"{name}_{weighting}"
        start = time.perf_counter()
        model.fit(X_train, y_train)
        seconds = time.perf_counter() - start
        scores[key] = model.predict_proba(X_val)[:, 1]
        results.append({"key": key, "model": name, "class_weighting": weighting,
                        "average_precision": float(average_precision_score(y_val, scores[key])),
                        "roc_auc": float(roc_auc_score(y_val, scores[key])),
                        **default_threshold_counts(y_val, scores[key]), "fit_seconds": round(seconds, 1)})
        print(f"Fitted {key} in {seconds:.1f}s")
    metrics = pd.DataFrame(results)
    draws = bootstrap_ap(y_val, scores, args.n_bootstrap, args.seed)
    best = metrics.loc[metrics.average_precision.idxmax(), "key"]
    metrics["ap_ci_low"] = draws[metrics.key].quantile(.025).to_numpy()
    metrics["ap_ci_high"] = draws[metrics.key].quantile(.975).to_numpy()
    # Paired gaps: on each resample, how far each model falls below the best validation model.
    metrics["ap_below_best"] = metrics.average_precision.max() - metrics.average_precision
    gaps = draws[best].to_numpy()[:, None] - draws[metrics.key].to_numpy()
    metrics["gap_ci_low"], metrics["gap_ci_high"] = np.quantile(gaps, [.025, .975], axis=0)
    metrics = metrics[["model", "class_weighting", "average_precision", "ap_ci_low", "ap_ci_high",
                       "ap_below_best", "gap_ci_low", "gap_ci_high", "roc_auc", "alerts_at_0.5",
                       "frauds_caught_at_0.5", "false_alarms_at_0.5", "precision_at_0.5", "recall_at_0.5",
                       "fit_seconds"]]
    metrics.to_csv(out / "metrics.csv", index=False)
    model_comparison(metrics, CHOSEN, out / "model_comparison.png")
    pd.DataFrame({"Time": val.Time.to_numpy(), "Amount": val.Amount.to_numpy(), "Class": y_val, **scores}).to_csv(
        out / "validation_scores.csv", index=False)
    write_json(out / "run.json", {
        "task": "Model comparison on the chronological validation split; the test split is not scored",
        "seed": args.seed, "threads": args.threads, "n_bootstrap": args.n_bootstrap,
        "bootstrap_draws_used": len(draws), "rows_after_deduplication": len(df), "duplicates_removed": duplicates,
        "features": FEATURES, "split_fractions": {"train": args.train_fraction, "validation": args.val_fraction},
        "split_cut_times_seconds": cuts, "splits": splits.to_dict("records"), "best_validation_model": best,
        "average_precision": "sklearn average_precision_score (step-wise, not trapezoidal)",
        "versions": versions(),
    })
    print(metrics.to_string(index=False))
