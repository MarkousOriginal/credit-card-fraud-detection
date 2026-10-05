"""Cost model: every alert costs one manual review; a caught fraud loses nothing, a missed fraud loses its Amount."""
import numpy as np
import pandas as pd

from .common import resample_weights


def outcome(y, amount, flag, review_cost, hours, weight=None):
    """Alerts, catches and estimated savings against flagging nothing, optionally on a weighted resample."""
    w = np.ones(len(y)) if weight is None else weight
    count = int if weight is None else float
    alerts, frauds, caught = count((w * flag).sum()), count((w * y).sum()), count((w * flag * y).sum())
    fraud_amount, caught_amount = float((w * y * amount).sum()), float((w * flag * y * amount).sum())
    savings = caught_amount - review_cost * alerts
    return {"alerts": alerts, "alerts_per_hour": alerts / hours, "frauds_caught": caught,
            "false_alarms": alerts - caught, "missed_frauds": frauds - caught,
            "precision": caught / alerts if alerts else None, "recall": caught / frauds if frauds else None,
            "fraud_amount_caught": caught_amount, "fraud_amount_missed": fraud_amount - caught_amount,
            "review_cost_total": review_cost * alerts, "cost_without_model": fraud_amount,
            "cost_with_model": fraud_amount - savings, "estimated_savings": savings,
            "savings_share": savings / fraud_amount if fraud_amount else None}


def ranked(y, amount, score):
    """Distinct thresholds from the highest down (np.inf flags nothing), with the rows sorted to match."""
    order = np.argsort(-score, kind="mergesort")
    s, loss = score[order], (y * amount)[order]
    # Transactions are flagged when score >= threshold, so tied scores are always flagged together.
    last = np.r_[s[1:] != s[:-1], True]
    return np.r_[np.inf, s[last]], last, loss


def totals(w, last, loss):
    """Alerts and fraud amount caught at every threshold, plus the total fraud amount, for row weights w."""
    return np.r_[0, np.cumsum(w)[last]], np.r_[0, np.cumsum(w * loss)[last]], (w * loss).sum()


def choose_thresholds(y, amount, score, review_costs, n_boot, seed):
    """Cheapest validation threshold for each review cost, and the thresholds chosen on bootstrap resamples."""
    thresholds, last, loss = ranked(y, amount, score)

    def cheapest(w):
        alerts, caught, fraud_amount = totals(w, last, loss)
        missed = fraud_amount - caught
        # argmin keeps the first minimum: among equal costs, the highest threshold and the fewest alerts.
        return [thresholds[np.argmin(cost * alerts + missed)] for cost in review_costs]

    draws = [cheapest(w) for w in resample_weights(len(loss), n_boot, seed)]
    return cheapest(np.ones(len(loss))), np.array(draws)


def savings_curve(y, amount, score, review_cost, hours):
    """Alerts per hour and the share of fraud losses avoided, net of reviews, at every distinct threshold."""
    thresholds, last, loss = ranked(y, amount, score)
    alerts, caught, fraud_amount = totals(np.ones(len(loss)), last, loss)
    return pd.DataFrame({"threshold": thresholds, "alerts_per_hour": alerts / hours,
                         "savings_share": (caught - review_cost * alerts) / fraud_amount})
