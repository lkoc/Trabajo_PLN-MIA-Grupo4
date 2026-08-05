"""Modelos y evaluación para las categorías gruesas del moderador.

Las etiquetas finas se usan únicamente para derivar el objetivo grueso y para
auditar/estratificar las particiones. Los flags transversales supervisan el
enrutamiento de casos dudosos, pero no son categorías temáticas del modelo.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable
import math

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight

from scripts_auxiliares.flujo_hibrido_moderador import (
    grouped_train_validation_test_split,
)


COARSE_ORDER = [
    "SEGURO",
    "RACISMO_DISCRIMINACION",
    "ACOSO_GENERO_IDENTIDAD",
    "ACOSO_PERSONAL",
    "AMENAZA_DIRECTA",
    "CONTENIDO_SEXUAL",
]
DAMAGE_ORDER = COARSE_ORDER[1:]
FINE_TO_COARSE = {
    "seguro": "SEGURO",
    "seguro_ironia_marcada": "SEGURO",
    "racismo_etnico_explicito": "RACISMO_DISCRIMINACION",
    "racismo_linguistico": "RACISMO_DISCRIMINACION",
    "clasismo_racial": "RACISMO_DISCRIMINACION",
    "discriminacion_regional": "RACISMO_DISCRIMINACION",
    "racismo_encubierto": "RACISMO_DISCRIMINACION",
    "misoginia_acoso_genero": "ACOSO_GENERO_IDENTIDAD",
    "homofobia_transfobia": "ACOSO_GENERO_IDENTIDAD",
    "acoso_personal": "ACOSO_PERSONAL",
    "amenaza_directa": "AMENAZA_DIRECTA",
    "sexual_explicito": "CONTENIDO_SEXUAL",
    "sexual_cosificacion": "CONTENIDO_SEXUAL",
    "sexual_no_consensual": "CONTENIDO_SEXUAL",
}
MODEL_ORDER = [
    "dummy_prior",
    "complement_nb",
    "logistic_regression",
    "linear_svm_word_char",
    "hist_gradient_boosting_svd",
]
MODEL_LABELS = {
    "dummy_prior": "Dummy (prior)",
    "complement_nb": "Complement NB",
    "logistic_regression": "Regresión logística",
    "linear_svm_word_char": "SVM lineal palabra+carácter",
    "hist_gradient_boosting_svd": "Gradient boosting + SVD",
}


class ConstantFeaturizer(BaseEstimator, TransformerMixin):
    """Una columna constante para el baseline que ignora el texto."""

    def fit(self, texts, y=None):
        return self

    def transform(self, texts):
        return np.zeros((len(texts), 1), dtype=np.float32)

    def fit_transform(self, texts, y=None):
        return self.transform(texts)


def add_coarse_targets(
    frame: pd.DataFrame,
    taxonomy: pd.DataFrame,
) -> pd.DataFrame:
    """Agrega ``coarse_labels`` sin convertir etiquetas finas en features."""
    taxonomy_labels = set(taxonomy.loc[taxonomy["categoria"] != "FLAG", "label"])
    if taxonomy_labels != set(FINE_TO_COARSE):
        missing = taxonomy_labels - set(FINE_TO_COARSE)
        extra = set(FINE_TO_COARSE) - taxonomy_labels
        raise ValueError(
            f"El mapeo grueso no coincide con la taxonomía; faltan={missing}, sobran={extra}."
        )

    def convert(labels: list[str]) -> list[str]:
        bases = {FINE_TO_COARSE[label] for label in labels}
        if not bases:
            raise ValueError(f"No se pudo derivar una categoría gruesa desde {labels}.")
        if "SEGURO" in bases and len(bases) > 1:
            raise ValueError(f"SEGURO coexiste con daño después del mapeo: {labels}.")
        return [base for base in COARSE_ORDER if base in bases]

    output = frame.copy()

    def resolve(row: pd.Series) -> list[str]:
        override = row.get("coarse_labels_override")
        if isinstance(override, (list, tuple, set)) and len(override) > 0:
            unknown = set(override) - set(COARSE_ORDER)
            if unknown:
                raise ValueError(f"Override grueso fuera de taxonomía: {unknown}.")
            if "SEGURO" in override and len(set(override)) > 1:
                raise ValueError("SEGURO coexiste con daño en un override humano.")
            return [base for base in COARSE_ORDER if base in set(override)]
        return convert(row["labels"])

    output["coarse_labels"] = output.apply(resolve, axis=1)
    output["has_transversal_flag"] = output["flags"].map(bool)
    return output


def target_matrix(frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(
        [
            [int(category in values) for category in COARSE_ORDER]
            for values in frame["coarse_labels"]
        ],
        dtype=np.int8,
    )


def stratification_matrix(
    frame: pd.DataFrame,
    fine_order: list[str],
    flag_order: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Matriz de auditoría; sus columnas nunca entran como predictores."""
    names = (
        [f"base::{name}" for name in COARSE_ORDER]
        + [f"fina::{name}" for name in fine_order]
        + [f"flag::{name}" for name in flag_order]
    )
    rows = []
    for row in frame.itertuples(index=False):
        coarse = set(row.coarse_labels)
        fine = set() if getattr(row, "fine_labels_reference_only", False) else set(row.labels)
        flags = set(row.flags)
        rows.append(
            [int(name in coarse) for name in COARSE_ORDER]
            + [int(name in fine) for name in fine_order]
            + [int(name in flags) for name in flag_order]
        )
    return np.asarray(rows, dtype=np.int8), names


def _split_balance_score(
    y: np.ndarray,
    split: dict[str, np.ndarray],
    target_sizes: dict[str, float],
) -> tuple[float, int]:
    global_prevalence = y.mean(axis=0)
    global_counts = y.sum(axis=0)
    eligible = global_counts >= 12
    scale = np.sqrt(np.maximum(global_prevalence, 1 / len(y)))
    score = 0.0
    missing = 0
    for name, indices in split.items():
        prevalence = y[indices].mean(axis=0)
        score += float(np.mean(np.abs(prevalence - global_prevalence) / scale))
        score += 2.0 * abs(len(indices) / len(y) - target_sizes[name])
        missing += int(((y[indices].sum(axis=0) == 0) & eligible).sum())
    return score + 5.0 * missing, missing


def balanced_group_split_search(
    frame: pd.DataFrame,
    fine_order: list[str],
    flag_order: list[str],
    seed: int = 42,
    test_size: float = 0.15,
    validation_size: float = 0.15,
    trials: int = 250,
) -> tuple[dict[str, np.ndarray], dict]:
    """Busca una división por video con mejor balance grueso/fino/transversal."""
    y, names = stratification_matrix(frame, fine_order, flag_order)
    target_sizes = {
        "train": 1 - test_size - validation_size,
        "validation": validation_size,
        "test": test_size,
    }
    best_split = None
    best_score = math.inf
    best_seed = None
    best_missing = None
    for trial in range(trials):
        candidate_seed = seed + trial
        split = grouped_train_validation_test_split(
            frame,
            seed=candidate_seed,
            test_size=test_size,
            validation_size=validation_size,
        )
        score, missing = _split_balance_score(y, split, target_sizes)
        if score < best_score:
            best_split = split
            best_score = score
            best_seed = candidate_seed
            best_missing = missing
    if best_split is None:
        raise RuntimeError("No se pudo construir una partición agrupada.")
    metadata = {
        "search_trials": trials,
        "initial_seed": seed,
        "selected_seed": best_seed,
        "balance_score": best_score,
        "missing_eligible_cells": best_missing,
        "stratification_columns": names,
        "note": "Las columnas finas y flags solo auditan la partición; no son features.",
    }
    return best_split, metadata


def split_prevalence_table(
    frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for split_name, frame in frames.items():
        for category in COARSE_ORDER:
            positives = int(frame["coarse_labels"].map(lambda x: category in x).sum())
            rows.append(
                {
                    "particion": split_name,
                    "categoria": category,
                    "chunks": len(frame),
                    "positivos": positives,
                    "prevalencia": positives / len(frame),
                    "videos_positivos": int(
                        frame.loc[
                            frame["coarse_labels"].map(lambda x: category in x), "video_id"
                        ].nunique()
                    ),
                }
            )
    return pd.DataFrame(rows)


def _make_featurizer(
    model_name: str,
    max_features: int = 50_000,
    model_parameters: dict | None = None,
):
    model_parameters = model_parameters or {}
    min_df = int(model_parameters.get("min_df", 2))
    common = dict(
        lowercase=True,
        strip_accents="unicode",
        min_df=min_df,
        sublinear_tf=True,
        dtype=np.float32,
    )
    if model_name == "dummy_prior":
        return ConstantFeaturizer()
    if model_name == "complement_nb":
        return TfidfVectorizer(
            ngram_range=(1, 1), max_features=max_features, **common
        )
    if model_name == "logistic_regression":
        return TfidfVectorizer(
            ngram_range=(1, 2), max_features=max_features, **common
        )
    if model_name == "linear_svm_word_char":
        return FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        ngram_range=(1, 2), max_features=max_features // 2, **common
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        max_features=max_features // 2,
                        **common,
                    ),
                ),
            ]
        )
    if model_name == "hist_gradient_boosting_svd":
        svd_components = int(model_parameters.get("svd_components", 96))
        return Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2), max_features=max_features, **common
                    ),
                ),
                (
                    "svd",
                    TruncatedSVD(
                        n_components=svd_components,
                        n_iter=int(model_parameters.get("svd_n_iter", 5)),
                        random_state=42,
                    ),
                ),
            ]
        )
    raise ValueError(f"Modelo no reconocido: {model_name}")


def _make_estimator(model_name: str, model_parameters: dict | None = None):
    model_parameters = model_parameters or {}
    regularization = float(model_parameters.get("C", 1.0))
    if model_name == "dummy_prior":
        return DummyClassifier(strategy="prior")
    if model_name == "complement_nb":
        return ComplementNB(alpha=float(model_parameters.get("alpha", 0.5)))
    if model_name == "logistic_regression":
        return LogisticRegression(
            C=regularization,
            max_iter=1_000,
            class_weight="balanced",
            solver="liblinear",
            random_state=42,
        )
    if model_name == "linear_svm_word_char":
        return LinearSVC(
            C=regularization,
            class_weight="balanced",
            max_iter=5_000,
            random_state=42,
        )
    if model_name == "hist_gradient_boosting_svd":
        return HistGradientBoostingClassifier(
            learning_rate=float(model_parameters.get("learning_rate", 0.08)),
            max_iter=int(model_parameters.get("max_iter", 100)),
            max_leaf_nodes=int(model_parameters.get("max_leaf_nodes", 31)),
            l2_regularization=float(model_parameters.get("l2_regularization", 1.0)),
            early_stopping=True,
            random_state=42,
        )
    raise ValueError(f"Modelo no reconocido: {model_name}")


def _scores_from_estimators(estimators: list, X) -> np.ndarray:
    columns = []
    for estimator in estimators:
        classes = np.asarray(estimator.classes_)
        if classes.size == 1:
            score = np.full(X.shape[0], float(classes[0]))
        elif hasattr(estimator, "predict_proba"):
            positive_index = int(np.where(classes == 1)[0][0])
            score = estimator.predict_proba(X)[:, positive_index]
        else:
            score = expit(estimator.decision_function(X))
        columns.append(score)
    return np.column_stack(columns)


def tune_thresholds(
    y_true: np.ndarray,
    scores: np.ndarray,
    minimum_positives: int = 10,
) -> np.ndarray:
    thresholds = np.full(y_true.shape[1], 0.50, dtype=float)
    grid = np.linspace(0.05, 0.95, 37)
    for column in range(y_true.shape[1]):
        if y_true[:, column].sum() < minimum_positives:
            continue
        candidates = []
        for threshold in grid:
            prediction = scores[:, column] >= threshold
            score = f1_score(y_true[:, column], prediction, zero_division=0)
            candidates.append((score, -abs(threshold - 0.50), threshold))
        thresholds[column] = max(candidates)[2]
    return thresholds


def constrained_coarse_predictions(
    scores: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """Daño tiene precedencia; si no se predice daño, la salida es SEGURO."""
    raw = scores >= thresholds
    output = np.zeros_like(raw, dtype=np.int8)
    damage = raw[:, 1:]
    output[:, 1:] = damage
    output[:, 0] = ~damage.any(axis=1)
    return output


def coarse_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
) -> tuple[dict, pd.DataFrame]:
    damage_true = y_true[:, 1:]
    damage_pred = y_pred[:, 1:]
    damage_scores = scores[:, 1:]
    summary = {
        "exact_match": float(accuracy_score(y_true, y_pred)),
        "jaccard_samples": float(
            jaccard_score(y_true, y_pred, average="samples", zero_division=1)
        ),
        "precision_micro": float(
            precision_score(y_true, y_pred, average="micro", zero_division=0)
        ),
        "recall_micro": float(
            recall_score(y_true, y_pred, average="micro", zero_division=0)
        ),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "pr_auc_macro": float(average_precision_score(y_true, scores, average="macro")),
        "damage_precision_micro": float(
            precision_score(damage_true, damage_pred, average="micro", zero_division=0)
        ),
        "damage_recall_micro": float(
            recall_score(damage_true, damage_pred, average="micro", zero_division=0)
        ),
        "damage_f1_micro": float(
            f1_score(damage_true, damage_pred, average="micro", zero_division=0)
        ),
        "damage_f1_macro": float(
            f1_score(damage_true, damage_pred, average="macro", zero_division=0)
        ),
        "damage_pr_auc_macro": float(
            average_precision_score(damage_true, damage_scores, average="macro")
        ),
        "n": int(len(y_true)),
    }
    report = pd.DataFrame(
        classification_report(
            y_true,
            y_pred,
            target_names=COARSE_ORDER,
            output_dict=True,
            zero_division=0,
        )
    ).T
    return summary, report


@dataclass
class CoarseTextModel:
    name: str
    featurizer: object
    estimators: list
    thresholds: np.ndarray
    review_margin: float = 0.05
    metadata: dict | None = None

    def predict_scores(self, texts: Iterable[str]) -> np.ndarray:
        text_list = list(texts)
        X = self.featurizer.transform(text_list)
        return _scores_from_estimators(self.estimators, X)

    def predict_matrix(self, texts: Iterable[str]) -> np.ndarray:
        return constrained_coarse_predictions(self.predict_scores(texts), self.thresholds)

    def review_mask(self, texts: Iterable[str]) -> np.ndarray:
        scores = self.predict_scores(texts)
        distances = np.min(np.abs(scores[:, 1:] - self.thresholds[1:]), axis=1)
        return distances <= self.review_margin

    def predict(self, texts: Iterable[str]) -> list[dict]:
        text_list = list(texts)
        scores = self.predict_scores(text_list)
        matrix = constrained_coarse_predictions(scores, self.thresholds)
        review = np.min(
            np.abs(scores[:, 1:] - self.thresholds[1:]), axis=1
        ) <= self.review_margin
        return [
            {
                "coarse_labels": [
                    name for name, selected in zip(COARSE_ORDER, matrix[row]) if selected
                ],
                "needs_review": bool(review[row]),
                "scores": {
                    name: float(value) for name, value in zip(COARSE_ORDER, scores[row])
                },
            }
            for row in range(len(text_list))
        ]


def fit_candidate(
    model_name: str,
    frame: pd.DataFrame,
    max_features: int = 50_000,
    text_column: str = "text",
    model_parameters: dict | None = None,
    flash_pseudo_weight: float | None = None,
) -> tuple[CoarseTextModel, dict]:
    if model_name not in MODEL_ORDER:
        raise ValueError(f"Modelo no reconocido: {model_name}")
    y = target_matrix(frame)
    if text_column not in frame:
        raise ValueError(f"No existe la columna de texto {text_column!r}.")
    source_weights = frame["sample_weight"].to_numpy(dtype=float)
    if flash_pseudo_weight is not None:
        if not 0 < flash_pseudo_weight <= 1:
            raise ValueError("flash_pseudo_weight debe estar en (0, 1].")
        source_weights = np.where(
            frame["label_source"].ne("flash_pseudo").to_numpy(),
            frame["sample_weight"].to_numpy(dtype=float),
            flash_pseudo_weight
            * frame["score_confianza_source"].to_numpy(dtype=float),
        )
    featurizer = _make_featurizer(
        model_name,
        max_features=max_features,
        model_parameters=model_parameters,
    )
    start = perf_counter()
    X = featurizer.fit_transform(frame[text_column].tolist())
    feature_seconds = perf_counter() - start
    estimators = []
    fit_start = perf_counter()
    for column in range(y.shape[1]):
        target = y[:, column]
        estimator = _make_estimator(model_name, model_parameters=model_parameters)
        weights = source_weights
        if model_name == "hist_gradient_boosting_svd":
            weights = source_weights * compute_sample_weight("balanced", target)
        estimator.fit(X, target, sample_weight=weights)
        estimators.append(estimator)
    fit_seconds = perf_counter() - fit_start
    model = CoarseTextModel(
        name=model_name,
        featurizer=featurizer,
        estimators=estimators,
        thresholds=np.full(len(COARSE_ORDER), 0.50),
        metadata={"model_label": MODEL_LABELS[model_name]},
    )
    diagnostics = {
        "model": model_name,
        "model_label": MODEL_LABELS[model_name],
        "features": int(X.shape[1]),
        "feature_seconds": feature_seconds,
        "fit_seconds": fit_seconds,
        "training_seconds": feature_seconds + fit_seconds,
        "text_column": text_column,
        "flash_pseudo_weight": flash_pseudo_weight,
        "model_parameters": model_parameters or {},
    }
    return model, diagnostics


def evaluate_candidate(
    model: CoarseTextModel,
    frame: pd.DataFrame,
    text_column: str = "text",
) -> tuple[dict, pd.DataFrame, np.ndarray]:
    start = perf_counter()
    scores = model.predict_scores(frame[text_column].tolist())
    inference_seconds = perf_counter() - start
    predictions = constrained_coarse_predictions(scores, model.thresholds)
    summary, report = coarse_metrics(target_matrix(frame), predictions, scores)
    summary["inference_seconds"] = inference_seconds
    summary["milliseconds_per_1000"] = 1000 * inference_seconds / len(frame) * 1000
    return summary, report, scores


def tune_candidate(
    model: CoarseTextModel,
    validation_frame: pd.DataFrame,
    text_column: str = "text",
) -> np.ndarray:
    scores = model.predict_scores(validation_frame[text_column].tolist())
    model.thresholds = tune_thresholds(target_matrix(validation_frame), scores)
    return scores


def review_routing_curve(
    scores: np.ndarray,
    thresholds: np.ndarray,
    flags: pd.Series,
    margins: np.ndarray | None = None,
) -> pd.DataFrame:
    """Evalúa si la incertidumbre del modelo captura flags anotados."""
    margins = margins if margins is not None else np.linspace(0.01, 0.30, 30)
    flag_any = flags.map(bool).to_numpy()
    distances = np.min(np.abs(scores[:, 1:] - thresholds[1:]), axis=1)
    rows = []
    for margin in margins:
        review = distances <= margin
        rows.append(
            {
                "margen": float(margin),
                "tasa_revision": float(review.mean()),
                "cobertura_automatica": float(1 - review.mean()),
                "captura_flags": float(review[flag_any].mean()) if flag_any.any() else math.nan,
                "precision_flags_revision": (
                    float(flag_any[review].mean()) if review.any() else math.nan
                ),
                "flags": int(flag_any.sum()),
            }
        )
    return pd.DataFrame(rows)


def select_review_margin(
    curve: pd.DataFrame,
    minimum_flag_capture: float = 0.80,
) -> tuple[pd.Series, bool]:
    eligible = curve.loc[curve["captura_flags"] >= minimum_flag_capture]
    if not eligible.empty:
        selected = eligible.sort_values(
            ["cobertura_automatica", "margen"], ascending=[False, True]
        ).iloc[0]
        return selected, True
    selected = curve.sort_values(
        ["captura_flags", "cobertura_automatica"], ascending=[False, False]
    ).iloc[0]
    return selected, False


def save_coarse_model(model: CoarseTextModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_coarse_model(path: Path) -> CoarseTextModel:
    return joblib.load(path)
