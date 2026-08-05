from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .device import resolve_device, torch_device_name
from .io import write_json_atomic
from .taxonomy import load_taxonomy
from .training import encode_targets


@dataclass(frozen=True)
class TrainingSpecification:
    family: str
    model_id: str
    max_length: int = 128
    learning_rate: float = 2e-5
    epochs: int = 3
    batch_size: int = 8
    gradient_accumulation: int = 1
    seed: int = 20260805


TRANSFORMER_SPECS = {
    "minilm": TrainingSpecification(
        family="transformer_encoder",
        model_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ),
    "e5": TrainingSpecification(
        family="transformer_encoder",
        model_id="intfloat/multilingual-e5-small",
    ),
    "qwen_lora": TrainingSpecification(
        family="qwen_lora",
        model_id="Qwen/Qwen3-0.6B-Base",
        learning_rate=1e-4,
        epochs=4,
        batch_size=2,
        gradient_accumulation=4,
    ),
}


def texts_and_targets(rows: Iterable[dict[str, Any]]) -> tuple[list[str], np.ndarray]:
    materialized = list(rows)
    texts = [str(row["text"]) for row in materialized]
    return texts, encode_targets(materialized)


def train_classical_suite(
    train_rows: Iterable[dict[str, Any]],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Entrena baselines de cinco salidas. El conjunto de prueba no interviene."""

    try:
        import joblib
        from sklearn.dummy import DummyClassifier
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression, SGDClassifier
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.naive_bayes import ComplementNB
        from sklearn.pipeline import Pipeline
        from sklearn.svm import LinearSVC
    except ImportError as exc:
        raise RuntimeError("Instale moderacion-peru[entrenamiento]") from exc

    texts, targets = texts_and_targets(train_rows)
    if not texts:
        raise ValueError("No existen filas de entrenamiento")
    candidates = {
        "dummy": DummyClassifier(strategy="prior"),
        "complement_nb": ComplementNB(alpha=1.0),
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "linear_svm": LinearSVC(class_weight="balanced"),
        "sgd_incremental": SGDClassifier(loss="log_loss", class_weight="balanced", random_state=20260805),
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    registry = {"schema_version": "2.0.0", "target_labels": list(load_taxonomy().target_labels), "models": {}}
    for name, estimator in candidates.items():
        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=150000, sublinear_tf=True)),
                ("classifier", OneVsRestClassifier(estimator, n_jobs=1)),
            ]
        )
        pipeline.fit(texts, targets)
        path = output / f"{name}.joblib"
        joblib.dump(pipeline, path)
        registry["models"][name] = {"path": str(path), "incremental": name == "sgd_incremental"}
    write_json_atomic(output / "registry.json", registry)
    return registry


def build_transformer_classifier(spec: TrainingSpecification, device: str = "auto"):
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Instale moderacion-peru[entrenamiento]") from exc
    taxonomy = load_taxonomy()
    hardware = resolve_device(device)
    id2label = {index: label for index, label in enumerate(taxonomy.target_labels)}
    label2id = {label: index for index, label in id2label.items()}
    tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        spec.model_id,
        num_labels=len(taxonomy.target_labels),
        id2label=id2label,
        label2id=label2id,
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True,
    )
    model.to(torch_device_name(hardware))
    return tokenizer, model, hardware


def build_qwen_lora_classifier(device: str = "auto"):
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise RuntimeError("Instale PEFT mediante moderacion-peru[entrenamiento]") from exc
    spec = TRANSFORMER_SPECS["qwen_lora"]
    tokenizer, model, hardware = build_transformer_classifier(spec, device)
    lora = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    return tokenizer, get_peft_model(model, lora), hardware


def save_training_specification(path: str | Path, spec: TrainingSpecification, device: str) -> None:
    taxonomy = load_taxonomy()
    write_json_atomic(
        path,
        {
            "schema_version": "2.0.0",
            "taxonomy_contract": taxonomy.contract_id,
            "target_labels": list(taxonomy.target_labels),
            "specification": asdict(spec),
            "device": resolve_device(device).model_dump(),
            "resume_supported": spec.family in {"transformer_encoder", "qwen_lora"},
            "selection_split": "validation",
            "test_used_for_selection": False,
        },
    )
