"""Correctness checks and an end-to-end run on generated transactions, never the Kaggle data."""
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_ml.common import FEATURES, chronological_split, load_transactions
from fraud_ml.costs import choose_thresholds, outcome


def transactions(n=3000, seed=7):
    rng = np.random.default_rng(seed)
    fraud = rng.random(n) < .03
    # Frauds are shifted in feature space so that the models have something to learn.
    df = pd.DataFrame(rng.normal(size=(n, 28)) + fraud[:, None] * .4, columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", np.sort(rng.integers(0, 172800, n)).astype(float))
    df["Amount"] = rng.lognormal(3, 1, n).round(2)
    df["Class"] = fraud.astype(int)
    return df


class CorrectnessTests(unittest.TestCase):
    def test_loading_drops_exact_duplicates_and_orders_by_time(self):
        df = transactions(300)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "creditcard.csv"
            pd.concat([df, df.iloc[[5]]]).sample(frac=1, random_state=1).to_csv(path, index=False)
            loaded, removed = load_transactions(path)
        self.assertEqual(removed, 1)
        self.assertEqual(len(loaded), 300)
        self.assertTrue(loaded.Time.is_monotonic_increasing)

    def test_invalid_files_fail_cleanly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "creditcard.csv"
            transactions(50).drop(columns=["V5"]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "missing columns"):
                load_transactions(path)
            transactions(50).assign(Class=2).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Class must be 0 or 1"):
                load_transactions(path)

    def test_split_is_chronological_and_keeps_each_second_together(self):
        df = transactions(1000)
        # A burst of transactions in one second straddles the 60% cut.
        df.loc[550:650, "Time"] = df.Time[600]
        labels, cuts = chronological_split(df, .6, .2)
        times = {name: df.Time[labels == name] for name in ["train", "validation", "test"]}
        self.assertLess(times["train"].max(), times["validation"].min())
        self.assertLess(times["validation"].max(), times["test"].min())
        self.assertEqual(pd.Series(labels)[df.Time == df.Time[600]].nunique(), 1)
        self.assertEqual((labels == "train").sum(), 550)

    def test_every_split_needs_fraud_cases(self):
        df = transactions(1000)
        df.loc[df.index >= 800, "Class"] = 0
        with self.assertRaisesRegex(ValueError, "test split has no fraud"):
            chronological_split(df)
        with self.assertRaisesRegex(ValueError, "leave rows for the test split"):
            chronological_split(df, .7, .3)
        with self.assertRaisesRegex(ValueError, "sorted by Time"):
            chronological_split(df.iloc[::-1].reset_index(drop=True))

    def test_time_and_label_are_not_features(self):
        self.assertNotIn("Time", FEATURES)
        self.assertNotIn("Class", FEATURES)
        self.assertEqual(len(FEATURES), 29)

    def test_cheapest_threshold_flags_ties_together_and_prefers_fewer_alerts(self):
        y, amount = np.array([1, 0, 1, 0, 0]), np.array([100., 5., 3., 7., 2.])
        score = np.array([.9, .8, .8, .3, .1])
        # Costs for thresholds inf, .9, .8, .3, .1 are 103, c+3, 3c, 4c, 5c: both .8 scores are flagged together.
        # At c=1.5 thresholds .9 and .8 cost 4.5 each, and the one with fewer alerts wins.
        chosen, draws = choose_thresholds(y, amount, score, [1, 1.5, 50, 200], n_boot=5, seed=0)
        self.assertEqual(chosen, [.8, .9, .9, np.inf])
        self.assertEqual(draws.shape, (5, 4))

    def test_outcome_counts_costs_and_weighted_resamples(self):
        y, amount = np.array([1, 0, 1, 0, 0]), np.array([100., 5., 3., 7., 2.])
        flag = np.array([True, True, True, False, False])
        result = outcome(y, amount, flag, review_cost=1, hours=2)
        self.assertEqual((result["alerts"], result["frauds_caught"], result["false_alarms"]), (3, 2, 1))
        self.assertEqual((result["estimated_savings"], result["cost_with_model"], result["alerts_per_hour"]),
                         (100, 3, 1.5))
        # A weight of 2 on the first row is the same as drawing it twice.
        weighted = outcome(y, amount, flag, 1, 2, weight=np.array([2, 1, 1, 1, 1]))
        self.assertEqual((weighted["alerts"], weighted["fraud_amount_caught"], weighted["estimated_savings"]),
                         (4, 203, 199))


class IntegrationTests(unittest.TestCase):
    def test_evaluate_compares_all_models_on_validation_only(self):
        from fraud_ml.evaluate import run
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transactions().to_csv(root / "creditcard.csv", index=False)
            run(Namespace(data=root / "creditcard.csv", output=root / "evaluate", train_fraction=.6,
                          val_fraction=.2, n_bootstrap=50, seed=42, threads=1))
            metrics = pd.read_csv(root / "evaluate" / "metrics.csv")
            splits = pd.read_csv(root / "evaluate" / "splits.csv").set_index("split")
            scores = pd.read_csv(root / "evaluate" / "validation_scores.csv")
            info = json.loads((root / "evaluate" / "run.json").read_text(encoding="utf-8"))
            self.assertTrue((root / "evaluate" / "model_comparison.png").exists())
        self.assertEqual(len(metrics), 7)
        self.assertEqual(splits.rows.sum(), 3000)
        # Only validation rows are scored; the test split is counted but never passed to a model.
        self.assertEqual(len(scores), splits.loc["validation", "rows"])
        start, end = info["split_cut_times_seconds"]
        self.assertTrue(((scores.Time >= start) & (scores.Time < end)).all())
        self.assertIn("not scored", info["task"])
        # A constant score ranks nothing, so its average precision equals the validation fraud rate.
        dummy = metrics.set_index("model").loc["dummy_prior"]
        self.assertAlmostEqual(dummy.average_precision, splits.loc["validation", "fraud_rate"])
        self.assertGreater(metrics.average_precision.max(), 2 * splits.loc["validation", "fraud_rate"])
        self.assertTrue((metrics.ap_ci_low <= metrics.ap_ci_high).all())
        self.assertEqual(metrics.ap_below_best.min(), 0)
        self.assertEqual(info["bootstrap_draws_used"], 50)

    def test_thresholds_are_frozen_on_validation_then_applied_to_test(self):
        from fraud_ml.final import run as final
        from fraud_ml.threshold import run as threshold
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transactions().to_csv(root / "creditcard.csv", index=False)
            common = dict(data=root / "creditcard.csv", n_bootstrap=20, seed=42)
            threshold(Namespace(**common, output=root / "threshold", train_fraction=.6, val_fraction=.2,
                                review_costs=[1., 5., 25.], headline_cost=5., threads=1))
            # Choosing thresholds writes validation results only; nothing about the test period.
            self.assertEqual(sorted(path.name for path in (root / "threshold").iterdir()),
                             ["model.json", "policy.json", "run.json", "validation_policy.csv", "validation_scores.csv"])
            policy = json.loads((root / "threshold" / "policy.json").read_text(encoding="utf-8"))
            validation = pd.read_csv(root / "threshold" / "validation_policy.csv").set_index("policy")
            scores = pd.read_csv(root / "threshold" / "validation_scores.csv")
            final(Namespace(**common, policy=root / "threshold", output=root / "final"))
            test = pd.read_csv(root / "final" / "test_policy.csv").set_index("policy")
            metrics = pd.read_csv(root / "final" / "test_metrics.csv")
            self.assertTrue((root / "final" / "savings_curve.png").exists())
            # Thresholds frozen on one file cannot be applied to a file that splits differently.
            transactions(seed=8).to_csv(root / "other.csv", index=False)
            with self.assertRaisesRegex(ValueError, "different split times"):
                final(Namespace(**{**common, "data": root / "other.csv"}, policy=root / "threshold",
                                output=root / "other"))
        self.assertEqual(set(policy["thresholds"]), {"1", "5", "25"})
        # The frozen threshold is the cheapest of every candidate threshold on validation.
        frauds = scores[scores.Class == 1]
        costs = [5 * (scores.score >= t).sum() + frauds.Amount[frauds.score < t].sum()
                 for t in np.r_[np.inf, scores.score.unique()]]
        self.assertAlmostEqual(validation.loc["review_cost_5", "cost_with_model"], min(costs))
        # The test run applies the frozen thresholds; it never chooses its own. CSV keeps 16 digits, JSON all of them.
        for cost, t in policy["thresholds"].items():
            self.assertAlmostEqual(test.loc[f"review_cost_{cost}", "threshold"], np.inf if t is None else t, places=12)
        self.assertEqual(metrics.frauds.iloc[0], test.loc["review_cost_5", ["frauds_caught", "missed_frauds"]].sum())
        self.assertIn("default_0.5", test.index)
        self.assertTrue((test.recall_ci_low <= test.recall_ci_high).all())


if __name__ == "__main__":
    unittest.main()
