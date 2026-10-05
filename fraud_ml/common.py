import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

# Raw Time only orders the transactions: test rows lie beyond every Time value seen in training.
FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]
SPLITS = ["train", "validation", "test"]


def require_columns(df, columns, source):
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{source}: missing columns {missing}")


def load_transactions(path):
    """Return the transactions ordered by Time and the number of exact duplicate rows removed."""
    df = pd.read_csv(path)
    require_columns(df, ["Time", "Class"] + FEATURES, path)
    if df[["Time", "Class"] + FEATURES].isna().any().any():
        raise ValueError(f"{path}: missing values in Time, Class or feature columns")
    if not df.Class.isin([0, 1]).all():
        raise ValueError(f"{path}: Class must be 0 or 1")
    before = len(df)
    df = df.drop_duplicates()
    # A stable sort keeps the file order of transactions recorded in the same second.
    df = df.sort_values("Time", kind="mergesort").reset_index(drop=True)
    df["Class"] = df.Class.astype(int)
    return df, before - len(df)


def chronological_split(df, train_fraction=.6, val_fraction=.2):
    """Label rows train, validation or test so that each split starts after the previous one ends."""
    if train_fraction <= 0 or val_fraction <= 0 or train_fraction + val_fraction >= 1:
        raise ValueError("Split fractions must be positive and leave rows for the test split")
    if not df.Time.is_monotonic_increasing:
        raise ValueError("Transactions must be sorted by Time")
    times = df.Time.to_numpy()
    cuts = [float(times[int(len(df) * f)]) for f in (train_fraction, train_fraction + val_fraction)]
    # Cut at Time values, not row positions, so transactions from the same second share a split.
    labels = np.where(times < cuts[0], "train", np.where(times < cuts[1], "validation", "test"))
    for name in SPLITS:
        if df.Class[labels == name].sum() == 0:
            raise ValueError(f"The {name} split has no fraud cases; adjust the split fractions")
    return labels, cuts


def split_summary(df, labels):
    rows = []
    for name in SPLITS:
        part = df[labels == name]
        rows.append({"split": name, "rows": len(part), "frauds": int(part.Class.sum()),
                     "fraud_rate": float(part.Class.mean()),
                     # Hours since the first transaction in the file; the dataset gives no clock time.
                     "start_hour": round(float(part.Time.min()) / 3600, 2),
                     "end_hour": round(float(part.Time.max()) / 3600, 2)})
    return pd.DataFrame(rows)


def resample_weights(n, n_boot, seed):
    """Bootstrap resamples as row counts; weighting rows by these counts equals drawing them with replacement."""
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        yield np.bincount(rng.integers(0, n, n), minlength=n)


def versions():
    return {"python": platform.python_version(),
            **{package: version(package) for package in ["numpy", "pandas", "scikit-learn", "xgboost"]}}


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
