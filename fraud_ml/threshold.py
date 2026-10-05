"""Step 2a: fit the chosen model and freeze the cheapest validation threshold per review cost. Test stays unscored."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .common import FEATURES, chronological_split, load_transactions, versions, write_json
from .costs import choose_thresholds, outcome
from .models import CHOSEN, candidates

ASSUMPTIONS = ["Every alert costs one manual review of review_cost, whether or not it is a fraud.",
               "A caught fraud loses nothing; a missed fraud loses its full Amount.",
               "The dataset does not state the currency of Amount."]


def run(args):
    from xgboost import XGBClassifier
    if args.headline_cost not in args.review_costs:
        raise ValueError("--headline-cost must be one of --review-costs")
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    df, _ = load_transactions(args.data)
    labels, cuts = chronological_split(df, args.train_fraction, args.val_fraction)
    train, val = df[labels == "train"], df[labels == "validation"]
    models = {(name, weighting): model for name, weighting, model in
              candidates(train.Class.to_numpy(), args.seed, args.threads)}
    model = models[CHOSEN]
    model.fit(train[FEATURES].to_numpy(), train.Class.to_numpy())
    model.save_model(out / "model.json")
    # Thresholds come from the saved file, which is what the test run and the API load.
    model = XGBClassifier(); model.load_model(out / "model.json")
    y, amount = val.Class.to_numpy(), val.Amount.to_numpy()
    score = model.predict_proba(val[FEATURES].to_numpy())[:, 1]
    hours = (val.Time.max() - val.Time.min()) / 3600
    chosen, draws = choose_thresholds(y, amount, score, args.review_costs, args.n_bootstrap, args.seed)
    # How far the cheapest threshold moves when the validation period is resampled.
    low, high = np.quantile(draws, [.025, .975], axis=0, method="nearest")
    rows = [{"policy": f"review_cost_{cost:g}", "review_cost": cost, "threshold": t,
             **outcome(y, amount, score >= t, cost, hours),
             "threshold_range_low": lo, "threshold_range_high": hi,
             "alerts_at_range_low": int((score >= lo).sum()), "alerts_at_range_high": int((score >= hi).sum())}
            for cost, t, lo, hi in zip(args.review_costs, chosen, low, high)]
    # The original script's predict() cut-off, costed at the headline review cost for comparison.
    rows.append({"policy": "default_0.5", "review_cost": args.headline_cost, "threshold": .5,
                 **outcome(y, amount, score >= .5, args.headline_cost, hours)})
    table = pd.DataFrame(rows)
    table.to_csv(out / "validation_policy.csv", index=False)
    pd.DataFrame({"Time": val.Time.to_numpy(), "Amount": amount, "Class": y, "score": score}).to_csv(
        out / "validation_scores.csv", index=False)
    write_json(out / "policy.json", {
        "model": "_".join(CHOSEN), "features": FEATURES, "headline_review_cost": args.headline_cost,
        "split_fractions": {"train": args.train_fraction, "validation": args.val_fraction},
        "split_cut_times_seconds": cuts,
        # null flags nothing: no threshold was cheaper than leaving every transaction unreviewed.
        "thresholds": {f"{cost:g}": None if np.isinf(t) else float(t) for cost, t in zip(args.review_costs, chosen)},
    })
    write_json(out / "run.json", {
        "task": "Threshold choice on the chronological validation split; the test split is not scored",
        "seed": args.seed, "threads": args.threads, "n_bootstrap": args.n_bootstrap,
        "validation_average_precision": float(average_precision_score(y, score)),
        "validation_hours": float(hours), "cost_assumptions": ASSUMPTIONS, "versions": versions(),
    })
    print(table.to_string(index=False))
