"""Step 2b: score the test period once, with the model and thresholds frozen on validation."""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .common import FEATURES, chronological_split, load_transactions, resample_weights, versions, write_json
from .costs import outcome, savings_curve
from .plots import savings_curves

INTERVALS = ["precision", "recall", "estimated_savings", "savings_share"]


def run(args):
    from xgboost import XGBClassifier
    source = Path(args.policy)
    policy = json.loads((source / "policy.json").read_text(encoding="utf-8"))
    if policy["features"] != FEATURES:
        raise ValueError("The frozen policy was built with different features")
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    df, _ = load_transactions(args.data)
    fractions = policy["split_fractions"]
    labels, cuts = chronological_split(df, fractions["train"], fractions["validation"])
    if cuts != policy["split_cut_times_seconds"]:
        raise ValueError("This data gives different split times from the run that froze the thresholds")
    model = XGBClassifier(); model.load_model(source / "model.json")
    periods = {}
    for period, name in [("Validation", "validation"), ("Test", "test")]:
        part = df[labels == name]
        periods[period] = (part.Class.to_numpy(), part.Amount.to_numpy(),
                           model.predict_proba(part[FEATURES].to_numpy())[:, 1],
                           (part.Time.max() - part.Time.min()) / 3600, part.Time.to_numpy())
    y, amount, score, hours, times = periods["Test"]
    policies = [(f"review_cost_{cost}", float(cost), np.inf if t is None else t)
                for cost, t in policy["thresholds"].items()]
    policies.append(("default_0.5", float(policy["headline_review_cost"]), .5))
    flags = {name: score >= t for name, _, t in policies}
    rows = [{"policy": name, "review_cost": cost, "threshold": t, **outcome(y, amount, flags[name], cost, hours)}
            for name, cost, t in policies]
    # Resamples of the test period give intervals for the reported numbers.
    ap, draws = [], {name: [] for name, _, _ in policies}
    for weight in resample_weights(len(y), args.n_bootstrap, args.seed):
        if not (weight * y).any():
            continue
        ap.append(average_precision_score(y, score, sample_weight=weight))
        for name, cost, _ in policies:
            result = outcome(y, amount, flags[name], cost, hours, weight)
            draws[name].append([result[key] for key in INTERVALS])
    for row in rows:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # A policy that flags nothing has no precision.
            low, high = np.nanquantile(np.array(draws[row["policy"]], dtype=float), [.025, .975], axis=0)
        for key, lo, hi in zip(INTERVALS, low, high):
            row.update({f"{key}_ci_low": lo, f"{key}_ci_high": hi})
    table = pd.DataFrame(rows)
    table.to_csv(out / "test_policy.csv", index=False)
    metrics = pd.DataFrame([{"rows": len(y), "frauds": int(y.sum()), "fraud_rate": float(y.mean()),
                             "hours": float(hours), "average_precision": float(average_precision_score(y, score)),
                             "ap_ci_low": float(np.quantile(ap, .025)), "ap_ci_high": float(np.quantile(ap, .975)),
                             "roc_auc": float(roc_auc_score(y, score))}])
    metrics.to_csv(out / "test_metrics.csv", index=False)
    pd.DataFrame({"Time": times, "Amount": amount, "Class": y, "score": score}).to_csv(
        out / "test_scores.csv", index=False)
    # Both periods at the headline review cost, scored by the same frozen model.
    headline = float(policy["headline_review_cost"])
    frozen = policy["thresholds"][f"{headline:g}"]
    cutoffs = {"frozen": np.inf if frozen is None else frozen, "default": .5}
    curves = {}
    for period, (py, pa, ps, ph, _) in periods.items():
        points = {kind: outcome(py, pa, ps >= t, headline, ph) for kind, t in cutoffs.items()}
        curves[period] = (savings_curve(py, pa, ps, headline, ph),
                          {kind: (p["alerts_per_hour"], p["savings_share"]) for kind, p in points.items()})
    savings_curves(curves, headline, out / "savings_curve.png")
    write_json(out / "run.json", {
        "task": "Single test evaluation with the model and thresholds frozen on validation",
        "model": policy["model"], "thresholds": policy["thresholds"],
        "headline_review_cost": policy["headline_review_cost"], "seed": args.seed,
        "n_bootstrap": args.n_bootstrap, "bootstrap_draws_used": len(ap), "versions": versions(),
    })
    print(metrics.to_string(index=False))
    print(table.to_string(index=False))
