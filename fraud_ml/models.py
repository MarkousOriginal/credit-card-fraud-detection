"""Candidate classifiers, each with and without extra weight on the rare fraud class."""
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Step 1 left the top five models within bootstrap noise of each other on validation average
# precision; this one had the highest point estimate and trains about 25 times faster than the forest.
CHOSEN = ("xgboost", "balanced")


def candidates(y_train, seed, threads):
    from xgboost import XGBClassifier
    # Weighting frauds by the class ratio gives both classes the same total weight in training.
    ratio = float((y_train == 0).sum() / (y_train == 1).sum())
    forest = dict(n_estimators=300, random_state=seed, n_jobs=threads)
    boosting = dict(n_estimators=300, max_depth=5, learning_rate=.1, tree_method="hist",
                    random_state=seed, n_jobs=threads)
    # Only the linear model needs scaling; trees split on thresholds and ignore feature scale.
    return [
        ("dummy_prior", "none", DummyClassifier(strategy="prior")),
        ("logistic_regression", "none", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
        ("logistic_regression", "balanced",
         make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))),
        ("random_forest", "none", RandomForestClassifier(**forest)),
        ("random_forest", "balanced", RandomForestClassifier(**forest, class_weight="balanced_subsample")),
        ("xgboost", "none", XGBClassifier(**boosting)),
        ("xgboost", "balanced", XGBClassifier(**boosting, scale_pos_weight=ratio)),
    ]
