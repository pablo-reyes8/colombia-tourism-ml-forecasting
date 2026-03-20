"""Model and feature selection helpers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from colombia_tourism.data import DEFAULT_FEATURES, DEFAULT_TARGET, FEATURE_GROUPS

DEFAULT_SELECTION_METHODS = [
    "lasso_polynomial",
    "adaptive_lasso",
    "sparse_group_lasso",
    "forward_selection",
    "backward_elimination",
    "pca_loadings",
    "pls_vip",
]


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


@dataclass
class FeatureSelectionResult:
    """Normalized output for one feature-selection method."""

    method: str
    selected_features: list[str]
    feature_scores: dict[str, float]
    selected_original_features: list[str]
    original_feature_scores: dict[str, float]
    model_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary_row(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "n_selected_features": len(self.selected_features),
            "n_selected_original_features": len(self.selected_original_features),
            "model_score": self.model_score,
            "selected_features": json.dumps(self.selected_features, ensure_ascii=False),
            "selected_original_features": json.dumps(
                self.selected_original_features,
                ensure_ascii=False,
            ),
            "metadata": json.dumps(self.metadata, ensure_ascii=False),
        }

    def ranking_frame(self, *, original_features: bool = False) -> pd.DataFrame:
        scores = self.original_feature_scores if original_features else self.feature_scores
        selected = (
            set(self.selected_original_features)
            if original_features
            else set(self.selected_features)
        )
        frame = pd.DataFrame(
            {
                "feature": list(scores.keys()),
                "score": list(scores.values()),
            }
        )
        frame["method"] = self.method
        frame["selected"] = frame["feature"].isin(selected)
        frame = frame.sort_values("score", ascending=False).reset_index(drop=True)
        frame["rank"] = np.arange(1, len(frame) + 1)
        frame["feature_space"] = "original" if original_features else "transformed"
        return frame


def _default_feature_selection_features(df: pd.DataFrame) -> list[str]:
    return [feature for feature in DEFAULT_FEATURES if feature in df.columns]


def prepare_selection_inputs(
    df: pd.DataFrame,
    *,
    target: str = DEFAULT_TARGET,
    feature_names: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Prepare the numeric selection dataset."""
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found")

    feature_names = feature_names or _default_feature_selection_features(df)
    feature_names = [feature for feature in feature_names if feature in df.columns]
    if not feature_names:
        raise ValueError("No usable features found for selection")

    X = df[feature_names].copy()
    y = df[target].copy()
    return X, y, feature_names


def make_polynomial_design_matrix(
    X: pd.DataFrame,
    *,
    poly_degree: int = 1,
    interaction_only: bool = False,
    include_bias: bool = False,
    impute_strategy: str = "mean",
    scale: bool = True,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Build an imputed, optionally polynomial-expanded design matrix."""
    imputer = SimpleImputer(strategy=impute_strategy)
    X_imputed = imputer.fit_transform(X)
    transformed_names = list(X.columns)
    poly = None
    if poly_degree and poly_degree > 1:
        poly = PolynomialFeatures(
            degree=poly_degree,
            include_bias=include_bias,
            interaction_only=interaction_only,
        )
        X_imputed = poly.fit_transform(X_imputed)
        transformed_names = poly.get_feature_names_out(X.columns).tolist()

    scaler = None
    X_final = X_imputed
    if scale:
        scaler = StandardScaler()
        X_final = scaler.fit_transform(X_imputed)

    return X_final, transformed_names, {
        "imputer": imputer,
        "poly": poly,
        "scaler": scaler,
    }


def _extract_original_features(term: str, original_features: list[str]) -> list[str]:
    matches: list[str] = []
    for feature in sorted(original_features, key=len, reverse=True):
        pattern = re.escape(feature) + r"(\^\d+)?"
        if re.search(pattern, term):
            matches.append(feature)
    unique_matches: list[str] = []
    for feature in matches:
        if feature not in unique_matches:
            unique_matches.append(feature)
    return unique_matches


def aggregate_transformed_scores_to_original(
    transformed_scores: dict[str, float],
    original_features: list[str],
) -> dict[str, float]:
    """Aggregate transformed-feature importance back to original variables."""
    aggregated = {feature: 0.0 for feature in original_features}
    for transformed_feature, score in transformed_scores.items():
        bases = _extract_original_features(transformed_feature, original_features)
        if not bases:
            continue
        for base in bases:
            aggregated[base] += float(score)
    return aggregated


def _safe_selected_features(
    scores: dict[str, float],
    *,
    threshold: float = 1e-8,
    top_k: int | None = None,
) -> list[str]:
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    selected = [feature for feature, score in ranked if score > threshold]
    if selected:
        if top_k is not None:
            return selected[:top_k]
        return selected
    if ranked:
        if top_k is None:
            return [ranked[0][0]]
        return [feature for feature, _ in ranked[:top_k]]
    return []


def _r2_cv_score(X, y, *, cv: int = 5, random_state: int = 42) -> float:
    splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    scores = cross_val_score(
        LinearRegression(),
        X,
        y,
        cv=splitter,
        scoring="r2",
    )
    return float(scores.mean())


def lasso_polynomial_selection(
    df: pd.DataFrame,
    *,
    target: str = DEFAULT_TARGET,
    feature_names: list[str] | None = None,
    poly_degree: int = 2,
    interaction_only: bool = False,
    cv: int = 5,
    random_state: int = 42,
    max_iter: int = 20_000,
    threshold: float = 1e-8,
    top_k_original: int | None = None,
) -> FeatureSelectionResult:
    """Select features using LassoCV over polynomial features."""
    X, y, original_features = prepare_selection_inputs(
        df,
        target=target,
        feature_names=feature_names,
    )
    X_matrix, transformed_names, _ = make_polynomial_design_matrix(
        X,
        poly_degree=poly_degree,
        interaction_only=interaction_only,
    )
    model = LassoCV(cv=cv, random_state=random_state, max_iter=max_iter)
    model.fit(X_matrix, y)
    transformed_scores = {
        name: float(abs(coef))
        for name, coef in zip(transformed_names, model.coef_, strict=False)
    }
    selected_features = _safe_selected_features(transformed_scores, threshold=threshold)
    original_scores = aggregate_transformed_scores_to_original(
        transformed_scores,
        original_features,
    )
    selected_original_features = _safe_selected_features(
        original_scores,
        threshold=threshold,
        top_k=top_k_original,
    )

    return FeatureSelectionResult(
        method="lasso_polynomial",
        selected_features=selected_features,
        feature_scores=transformed_scores,
        selected_original_features=selected_original_features,
        original_feature_scores=original_scores,
        model_score=float(model.score(X_matrix, y)),
        metadata={
            "alpha": float(model.alpha_),
            "poly_degree": poly_degree,
            "interaction_only": interaction_only,
            "transformed_feature_count": len(transformed_names),
        },
    )


def adaptive_lasso_selection(
    df: pd.DataFrame,
    *,
    target: str = DEFAULT_TARGET,
    feature_names: list[str] | None = None,
    poly_degree: int = 2,
    interaction_only: bool = False,
    cv: int = 5,
    random_state: int = 42,
    max_iter: int = 20_000,
    gamma: float = 1.0,
    ridge_alphas: np.ndarray | None = None,
    threshold: float = 1e-8,
    top_k_original: int | None = None,
) -> FeatureSelectionResult:
    """Adaptive Lasso using RidgeCV weights and LassoCV on re-weighted features."""
    X, y, original_features = prepare_selection_inputs(
        df,
        target=target,
        feature_names=feature_names,
    )
    X_matrix, transformed_names, _ = make_polynomial_design_matrix(
        X,
        poly_degree=poly_degree,
        interaction_only=interaction_only,
    )
    ridge_alphas = ridge_alphas if ridge_alphas is not None else np.logspace(-4, 4, 25)
    initial_model = RidgeCV(alphas=ridge_alphas)
    initial_model.fit(X_matrix, y)

    initial_coef = np.abs(np.ravel(initial_model.coef_))
    weights = 1.0 / np.clip(initial_coef, 1e-6, None) ** gamma
    weights = np.clip(weights, 1.0, 1e6)
    weighted_X = X_matrix / weights

    lasso = LassoCV(cv=cv, random_state=random_state, max_iter=max_iter)
    lasso.fit(weighted_X, y)
    final_coef = np.ravel(lasso.coef_) / weights

    transformed_scores = {
        name: float(abs(coef))
        for name, coef in zip(transformed_names, final_coef, strict=False)
    }
    selected_features = _safe_selected_features(transformed_scores, threshold=threshold)
    original_scores = aggregate_transformed_scores_to_original(
        transformed_scores,
        original_features,
    )
    selected_original_features = _safe_selected_features(
        original_scores,
        threshold=threshold,
        top_k=top_k_original,
    )

    return FeatureSelectionResult(
        method="adaptive_lasso",
        selected_features=selected_features,
        feature_scores=transformed_scores,
        selected_original_features=selected_original_features,
        original_feature_scores=original_scores,
        model_score=float(lasso.score(weighted_X, y)),
        metadata={
            "lasso_alpha": float(lasso.alpha_),
            "ridge_alpha": float(initial_model.alpha_),
            "gamma": gamma,
            "poly_degree": poly_degree,
            "interaction_only": interaction_only,
            "transformed_feature_count": len(transformed_names),
        },
    )


def infer_feature_group_labels(
    feature_names: list[str],
    groups: dict[str, tuple[str, ...]] | None = None,
) -> list[str]:
    """Map features to semantic groups used by sparse group lasso."""
    groups = groups or {
        name: values
        for name, values in FEATURE_GROUPS.items()
        if name != "target"
    }
    reverse_map: list[tuple[str, str]] = []
    for group_name, group_features in groups.items():
        for feature in group_features:
            reverse_map.append((feature, group_name))

    labels: list[str] = []
    for feature in feature_names:
        matches = [group for base, group in reverse_map if base in feature]
        if matches:
            labels.append(matches[0])
        else:
            labels.append("other")
    return labels


def _soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def _prox_sparse_group_lasso(
    values: np.ndarray,
    *,
    l1_penalty: float,
    group_penalty: float,
    groups: list[np.ndarray],
) -> np.ndarray:
    shrunk = _soft_threshold(values, l1_penalty)
    updated = shrunk.copy()
    for group in groups:
        group_values = shrunk[group]
        group_norm = np.linalg.norm(group_values, ord=2)
        if group_norm == 0:
            updated[group] = 0.0
            continue
        shrinkage = max(1.0 - group_penalty / group_norm, 0.0)
        updated[group] = shrinkage * group_values
    return updated


class SparseGroupLassoRegressor(BaseEstimator, RegressorMixin):
    """Simple proximal-gradient sparse group lasso for squared loss."""

    def __init__(
        self,
        *,
        alpha: float = 0.1,
        l1_ratio: float = 0.5,
        groups: list[int] | None = None,
        max_iter: int = 2_000,
        tol: float = 1e-6,
    ) -> None:
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.groups = groups
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n_samples, n_features = X.shape
        group_labels = self.groups or list(range(n_features))
        unique_groups = list(dict.fromkeys(group_labels))
        group_indices = [
            np.where(np.asarray(group_labels) == group)[0]
            for group in unique_groups
        ]

        X_mean = X.mean(axis=0)
        y_mean = y.mean()
        X_centered = X - X_mean
        y_centered = y - y_mean

        lipschitz = max(np.linalg.norm(X_centered, ord=2) ** 2 / n_samples, 1e-8)
        step = 1.0 / lipschitz
        beta = np.zeros(n_features, dtype=float)

        for iteration in range(self.max_iter):
            gradient = X_centered.T @ (X_centered @ beta - y_centered) / n_samples
            candidate = beta - step * gradient
            beta_next = _prox_sparse_group_lasso(
                candidate,
                l1_penalty=step * self.alpha * self.l1_ratio,
                group_penalty=step * self.alpha * (1.0 - self.l1_ratio),
                groups=group_indices,
            )
            if np.linalg.norm(beta_next - beta) <= self.tol * (np.linalg.norm(beta) + self.tol):
                beta = beta_next
                break
            beta = beta_next

        self.coef_ = beta
        self.intercept_ = float(y_mean - X_mean @ beta)
        self.n_iter_ = iteration + 1
        self.group_labels_ = group_labels
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_


def sparse_group_lasso_selection(
    df: pd.DataFrame,
    *,
    target: str = DEFAULT_TARGET,
    feature_names: list[str] | None = None,
    cv: int = 5,
    random_state: int = 42,
    alphas: np.ndarray | None = None,
    l1_ratios: tuple[float, ...] = (0.2, 0.5, 0.8),
    max_iter: int = 2_000,
    tol: float = 1e-6,
    threshold: float = 1e-8,
    top_k_original: int | None = None,
) -> FeatureSelectionResult:
    """Sparse group lasso using semantic feature groups from the project."""
    X, y, original_features = prepare_selection_inputs(
        df,
        target=target,
        feature_names=feature_names,
    )
    X_matrix, transformed_names, _ = make_polynomial_design_matrix(
        X,
        poly_degree=1,
    )
    group_labels = infer_feature_group_labels(transformed_names)
    unique_groups = {label: idx for idx, label in enumerate(dict.fromkeys(group_labels))}
    group_ids = [unique_groups[label] for label in group_labels]

    alphas = alphas if alphas is not None else np.logspace(-3, 0.5, 12)
    splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    best_score = -np.inf
    best_alpha = None
    best_l1_ratio = None
    for alpha in alphas:
        for l1_ratio in l1_ratios:
            scores = []
            for train_idx, test_idx in splitter.split(X_matrix):
                estimator = SparseGroupLassoRegressor(
                    alpha=float(alpha),
                    l1_ratio=float(l1_ratio),
                    groups=group_ids,
                    max_iter=max_iter,
                    tol=tol,
                )
                estimator.fit(X_matrix[train_idx], y.iloc[train_idx])
                predictions = estimator.predict(X_matrix[test_idx])
                scores.append(r2_score(y.iloc[test_idx], predictions))
            score = float(np.mean(scores))
            if score > best_score:
                best_score = score
                best_alpha = float(alpha)
                best_l1_ratio = float(l1_ratio)

    estimator = SparseGroupLassoRegressor(
        alpha=best_alpha,
        l1_ratio=best_l1_ratio,
        groups=group_ids,
        max_iter=max_iter,
        tol=tol,
    )
    estimator.fit(X_matrix, y)
    transformed_scores = {
        name: float(abs(coef))
        for name, coef in zip(transformed_names, estimator.coef_, strict=False)
    }
    selected_features = _safe_selected_features(transformed_scores, threshold=threshold)
    original_scores = transformed_scores.copy()
    selected_original_features = _safe_selected_features(
        original_scores,
        threshold=threshold,
        top_k=top_k_original,
    )

    return FeatureSelectionResult(
        method="sparse_group_lasso",
        selected_features=selected_features,
        feature_scores=transformed_scores,
        selected_original_features=selected_original_features,
        original_feature_scores=original_scores,
        model_score=best_score,
        metadata={
            "alpha": best_alpha,
            "l1_ratio": best_l1_ratio,
            "groups": unique_groups,
        },
    )


def forward_selection(
    df: pd.DataFrame,
    *,
    target: str = DEFAULT_TARGET,
    feature_names: list[str] | None = None,
    cv: int = 5,
    random_state: int = 42,
    tol: float = 1e-4,
    max_features: int | None = None,
) -> FeatureSelectionResult:
    """Greedy forward selection with linear regression and CV R²."""
    X, y, original_features = prepare_selection_inputs(
        df,
        target=target,
        feature_names=feature_names,
    )
    X_matrix, transformed_names, _ = make_polynomial_design_matrix(X, poly_degree=1)
    X_frame = pd.DataFrame(X_matrix, columns=transformed_names)

    remaining = transformed_names.copy()
    selected: list[str] = []
    history: list[dict[str, Any]] = []
    current_score = -np.inf

    while remaining and (max_features is None or len(selected) < max_features):
        best_candidate = None
        best_score = current_score
        for candidate in remaining:
            candidate_features = selected + [candidate]
            score = _r2_cv_score(
                X_frame[candidate_features],
                y,
                cv=cv,
                random_state=random_state,
            )
            if score > best_score + tol:
                best_score = score
                best_candidate = candidate

        if best_candidate is None:
            break

        selected.append(best_candidate)
        remaining.remove(best_candidate)
        current_score = best_score
        history.append(
            {
                "step": len(selected),
                "feature": best_candidate,
                "score": best_score,
            }
        )

    if selected:
        final_model = LinearRegression().fit(X_frame[selected], y)
        selected_scores = {
            feature: float(abs(coef))
            for feature, coef in zip(selected, final_model.coef_, strict=False)
        }
    else:
        selected_scores = {}

    full_scores = {feature: selected_scores.get(feature, 0.0) for feature in transformed_names}
    return FeatureSelectionResult(
        method="forward_selection",
        selected_features=selected,
        feature_scores=full_scores,
        selected_original_features=selected.copy(),
        original_feature_scores=full_scores,
        model_score=current_score if selected else None,
        metadata={"history": history, "tol": tol},
    )


def backward_elimination(
    df: pd.DataFrame,
    *,
    target: str = DEFAULT_TARGET,
    feature_names: list[str] | None = None,
    cv: int = 5,
    random_state: int = 42,
    tol: float = 1e-4,
    min_features: int = 1,
    target_feature_count: int | None = None,
) -> FeatureSelectionResult:
    """Greedy backward elimination with linear regression and CV R²."""
    X, y, original_features = prepare_selection_inputs(
        df,
        target=target,
        feature_names=feature_names,
    )
    X_matrix, transformed_names, _ = make_polynomial_design_matrix(X, poly_degree=1)
    X_frame = pd.DataFrame(X_matrix, columns=transformed_names)

    selected = transformed_names.copy()
    history: list[dict[str, Any]] = []
    current_score = _r2_cv_score(X_frame[selected], y, cv=cv, random_state=random_state)

    while len(selected) > max(min_features, target_feature_count or min_features):
        best_candidate_subset = None
        best_removed = None
        best_score = -np.inf
        for candidate in selected:
            candidate_subset = [feature for feature in selected if feature != candidate]
            score = _r2_cv_score(
                X_frame[candidate_subset],
                y,
                cv=cv,
                random_state=random_state,
            )
            if score > best_score:
                best_score = score
                best_candidate_subset = candidate_subset
                best_removed = candidate

        if best_candidate_subset is None:
            break
        if best_score < current_score - tol:
            break

        selected = best_candidate_subset
        current_score = best_score
        history.append(
            {
                "step": len(history) + 1,
                "removed_feature": best_removed,
                "score": best_score,
                "remaining_feature_count": len(selected),
            }
        )

    final_model = LinearRegression().fit(X_frame[selected], y)
    selected_scores = {
        feature: float(abs(coef))
        for feature, coef in zip(selected, final_model.coef_, strict=False)
    }
    full_scores = {feature: selected_scores.get(feature, 0.0) for feature in transformed_names}
    return FeatureSelectionResult(
        method="backward_elimination",
        selected_features=selected,
        feature_scores=full_scores,
        selected_original_features=selected.copy(),
        original_feature_scores=full_scores,
        model_score=current_score,
        metadata={"history": history, "tol": tol},
    )


def pca_loading_selection(
    df: pd.DataFrame,
    *,
    target: str = DEFAULT_TARGET,
    feature_names: list[str] | None = None,
    variance_threshold: float = 0.95,
    top_k: int | None = None,
) -> FeatureSelectionResult:
    """Rank original features by weighted PCA loadings."""
    X, _, original_features = prepare_selection_inputs(
        df,
        target=target,
        feature_names=feature_names,
    )
    X_matrix, transformed_names, _ = make_polynomial_design_matrix(X, poly_degree=1)
    pca = PCA()
    pca.fit(X_matrix)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumulative_variance, variance_threshold) + 1)
    weighted_loadings = (
        np.abs(pca.components_[:n_components]).T
        @ pca.explained_variance_ratio_[:n_components]
    )
    feature_scores = {
        feature: float(score)
        for feature, score in zip(transformed_names, weighted_loadings, strict=False)
    }
    selected_original_features = _safe_selected_features(
        feature_scores,
        threshold=float(np.mean(weighted_loadings)),
        top_k=top_k,
    )
    return FeatureSelectionResult(
        method="pca_loadings",
        selected_features=selected_original_features.copy(),
        feature_scores=feature_scores,
        selected_original_features=selected_original_features,
        original_feature_scores=feature_scores,
        model_score=float(cumulative_variance[n_components - 1]),
        metadata={
            "n_components": n_components,
            "explained_variance": float(cumulative_variance[n_components - 1]),
        },
    )


def _pls_vip(pls: PLSRegression) -> np.ndarray:
    t_scores = pls.x_scores_
    weights = pls.x_weights_
    y_loadings = pls.y_loadings_
    n_features = weights.shape[0]

    s = np.diag(t_scores.T @ t_scores @ y_loadings.T @ y_loadings).reshape(-1)
    total_s = np.sum(s)
    if total_s == 0:
        return np.zeros(n_features)

    weight_norm = np.sum(weights**2, axis=0)
    vip = np.sqrt(
        n_features
        * ((weights**2 / np.clip(weight_norm, 1e-12, None)) @ s)
        / total_s
    )
    return vip


def pls_vip_selection(
    df: pd.DataFrame,
    *,
    target: str = DEFAULT_TARGET,
    feature_names: list[str] | None = None,
    cv: int = 5,
    random_state: int = 42,
    max_components: int | None = None,
    top_k: int | None = None,
) -> FeatureSelectionResult:
    """Select features using PLS VIP scores."""
    X, y, original_features = prepare_selection_inputs(
        df,
        target=target,
        feature_names=feature_names,
    )
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)
    max_components = max_components or min(10, X_imputed.shape[1])
    max_components = max(1, min(max_components, X_imputed.shape[1]))
    splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    best_score = -np.inf
    best_components = 1
    for n_components in range(1, max_components + 1):
        model = PLSRegression(n_components=n_components, scale=True)
        scores = cross_val_score(model, X_imputed, y, cv=splitter, scoring="r2")
        score = float(scores.mean())
        if score > best_score:
            best_score = score
            best_components = n_components

    pls = PLSRegression(n_components=best_components, scale=True)
    pls.fit(X_imputed, y)
    vip_scores = _pls_vip(pls)
    feature_scores = {
        feature: float(score)
        for feature, score in zip(original_features, vip_scores, strict=False)
    }
    selected_original_features = _safe_selected_features(
        feature_scores,
        threshold=1.0,
        top_k=top_k,
    )
    return FeatureSelectionResult(
        method="pls_vip",
        selected_features=selected_original_features.copy(),
        feature_scores=feature_scores,
        selected_original_features=selected_original_features,
        original_feature_scores=feature_scores,
        model_score=best_score,
        metadata={"n_components": best_components},
    )


def aggregate_feature_selection_results(
    results: dict[str, FeatureSelectionResult],
) -> pd.DataFrame:
    """Aggregate consensus scores over original features."""
    all_features = sorted(
        {
            feature
            for result in results.values()
            for feature in result.original_feature_scores.keys()
        }
    )
    rows = []
    for feature in all_features:
        normalized_scores = []
        votes = 0
        methods = []
        for method, result in results.items():
            score = result.original_feature_scores.get(feature, 0.0)
            max_score = max(result.original_feature_scores.values(), default=0.0)
            normalized_scores.append(score / max_score if max_score > 0 else 0.0)
            if feature in result.selected_original_features:
                votes += 1
                methods.append(method)
        rows.append(
            {
                "feature": feature,
                "selection_votes": votes,
                "mean_normalized_score": float(np.mean(normalized_scores)),
                "selected_by": json.dumps(methods, ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["selection_votes", "mean_normalized_score"],
        ascending=[False, False],
    ).reset_index(drop=True)


def select_consensus_features(
    results: dict[str, FeatureSelectionResult],
    *,
    top_k: int | None = None,
    min_votes: int = 1,
) -> list[str]:
    """Return the consensus feature subset from the aggregated ranking."""
    consensus = aggregate_feature_selection_results(results)
    if consensus.empty:
        return []

    selected = consensus.loc[consensus["selection_votes"] >= min_votes, "feature"].tolist()
    if not selected:
        selected = consensus["feature"].tolist()
    if top_k is not None:
        selected = selected[:top_k]
    return selected


def run_feature_selection_suite(
    df: pd.DataFrame,
    *,
    target: str = DEFAULT_TARGET,
    feature_names: list[str] | None = None,
    methods: list[str] | None = None,
    cv: int = 5,
    random_state: int = 42,
    poly_degree: int = 2,
    max_features: int | None = 15,
) -> dict[str, FeatureSelectionResult]:
    """Run the full feature-selection suite over the final dataset."""
    methods = methods or DEFAULT_SELECTION_METHODS
    feature_names = feature_names or _default_feature_selection_features(df)

    results: dict[str, FeatureSelectionResult] = {}
    if "lasso_polynomial" in methods:
        results["lasso_polynomial"] = lasso_polynomial_selection(
            df,
            target=target,
            feature_names=feature_names,
            poly_degree=poly_degree,
            cv=cv,
            random_state=random_state,
            top_k_original=max_features,
        )
    if "adaptive_lasso" in methods:
        results["adaptive_lasso"] = adaptive_lasso_selection(
            df,
            target=target,
            feature_names=feature_names,
            poly_degree=poly_degree,
            cv=cv,
            random_state=random_state,
            top_k_original=max_features,
        )
    if "sparse_group_lasso" in methods:
        results["sparse_group_lasso"] = sparse_group_lasso_selection(
            df,
            target=target,
            feature_names=feature_names,
            cv=cv,
            random_state=random_state,
            top_k_original=max_features,
        )
    if "forward_selection" in methods:
        results["forward_selection"] = forward_selection(
            df,
            target=target,
            feature_names=feature_names,
            cv=cv,
            random_state=random_state,
            max_features=max_features,
        )
    if "backward_elimination" in methods:
        results["backward_elimination"] = backward_elimination(
            df,
            target=target,
            feature_names=feature_names,
            cv=cv,
            random_state=random_state,
            target_feature_count=max_features,
        )
    if "pca_loadings" in methods:
        results["pca_loadings"] = pca_loading_selection(
            df,
            target=target,
            feature_names=feature_names,
            top_k=max_features,
        )
    if "pls_vip" in methods:
        results["pls_vip"] = pls_vip_selection(
            df,
            target=target,
            feature_names=feature_names,
            cv=cv,
            random_state=random_state,
            top_k=max_features,
        )
    return results


def feature_selection_summary_frame(
    results: dict[str, FeatureSelectionResult],
) -> pd.DataFrame:
    return pd.DataFrame([result.summary_row() for result in results.values()])


def feature_selection_rankings_frame(
    results: dict[str, FeatureSelectionResult],
    *,
    original_features: bool = True,
) -> pd.DataFrame:
    frames = [
        result.ranking_frame(original_features=original_features)
        for result in results.values()
    ]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def save_feature_selection_report(
    results: dict[str, FeatureSelectionResult],
    output_dir: str | Path,
    *,
    consensus_top_k: int | None = None,
    consensus_min_votes: int = 1,
) -> dict[str, Path]:
    """Persist summary, rankings and consensus artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "feature_selection_summary.csv"
    ranking_path = output_dir / "feature_selection_rankings.csv"
    transformed_ranking_path = output_dir / "feature_selection_rankings_transformed.csv"
    consensus_path = output_dir / "feature_selection_consensus.csv"
    selected_json_path = output_dir / "selected_features.json"
    consensus_txt_path = output_dir / "consensus_features.txt"

    summary = feature_selection_summary_frame(results)
    rankings = feature_selection_rankings_frame(results, original_features=True)
    transformed_rankings = feature_selection_rankings_frame(
        results,
        original_features=False,
    )
    consensus = aggregate_feature_selection_results(results)
    consensus_features = select_consensus_features(
        results,
        top_k=consensus_top_k,
        min_votes=consensus_min_votes,
    )

    summary.to_csv(summary_path, index=False)
    rankings.to_csv(ranking_path, index=False)
    transformed_rankings.to_csv(transformed_ranking_path, index=False)
    consensus.to_csv(consensus_path, index=False)

    payload = {
        method: {
            "selected_features": result.selected_features,
            "selected_original_features": result.selected_original_features,
            "model_score": result.model_score,
            "metadata": result.metadata,
        }
        for method, result in results.items()
    }
    payload["consensus"] = {
        "selected_features": consensus_features,
        "top_k": consensus_top_k,
        "min_votes": consensus_min_votes,
    }
    selected_json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    consensus_txt_path.write_text(
        "\n".join(consensus_features),
        encoding="utf-8",
    )

    return {
        "summary": summary_path,
        "rankings": ranking_path,
        "transformed_rankings": transformed_ranking_path,
        "consensus": consensus_path,
        "selected_json": selected_json_path,
        "consensus_features": consensus_txt_path,
    }
