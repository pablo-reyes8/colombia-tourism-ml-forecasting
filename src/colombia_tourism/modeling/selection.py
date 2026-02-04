"""Model selection helpers."""

from __future__ import annotations

from sklearn.model_selection import RandomizedSearchCV


def tune_model(
    estimator,
    param_distributions,
    X,
    y,
    cv: int = 5,
    scoring: str = "r2",
    n_iter: int = 30,
    random_state: int = 42,
    n_jobs: int | None = None,
):
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    search.fit(X, y)
    return search
