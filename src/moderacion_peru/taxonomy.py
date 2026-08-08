from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .paths import find_project_root


class CategoryDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    display_name: str
    definition: str
    include: tuple[str, ...]
    exclude: tuple[str, ...]
    counterexample: str
    fine_labels: tuple[str, ...]
    evidence_layers: tuple[str, ...]


class TaxonomyContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_id: str
    version: str
    title: str
    safe_label: str = "SEGURO"
    damage_labels: tuple[str, ...]
    flags: tuple[str, ...]
    categories: dict[str, CategoryDefinition]
    fine_label_mapping: dict[str, str]
    legacy_coarse_mapping: dict[str, str]

    @model_validator(mode="after")
    def validate_contract(self) -> "TaxonomyContract":
        targets = self.target_labels
        if len(targets) != 5 or len(set(targets)) != 5:
            raise ValueError("El contrato debe contener SEGURO y cuatro daños únicos")
        if set(self.categories) != set(targets):
            raise ValueError("Las definiciones no coinciden con las salidas entrenadas")
        if set(self.fine_label_mapping.values()) - set(targets):
            raise ValueError("Existe una etiqueta fina sin categoría válida")
        return self

    @property
    def target_labels(self) -> tuple[str, ...]:
        return (self.safe_label, *self.damage_labels)

    @property
    def fine_labels(self) -> tuple[str, ...]:
        return tuple(self.fine_label_mapping)

    @staticmethod
    def _canonicalize_identifiers(
        labels: Iterable[str],
        canonical_values: Iterable[str],
        *,
        error_label: str,
    ) -> tuple[str, ...]:
        canonical = tuple(canonical_values)
        by_casefold = {value.casefold(): value for value in canonical}
        normalized: list[str] = []
        unknown: list[str] = []
        for label in labels:
            match = by_casefold.get(label.casefold())
            if match is None:
                unknown.append(label)
            elif match not in normalized:
                normalized.append(match)
        if unknown:
            raise ValueError(f"{error_label}: {sorted(unknown, key=str.casefold)}")
        return tuple(normalized)

    def normalize_categories(self, labels: Iterable[str]) -> tuple[str, ...]:
        unique = set(
            self._canonicalize_identifiers(
                labels,
                self.target_labels,
                error_label="Categorías desconocidas",
            )
        )
        if self.safe_label in unique and len(unique) > 1:
            raise ValueError("SEGURO es mutuamente excluyente con cualquier daño")
        return tuple(label for label in self.target_labels if label in unique)

    def normalize_fine_labels(self, labels: Iterable[str]) -> tuple[str, ...]:
        return self._canonicalize_identifiers(
            labels,
            self.fine_labels,
            error_label="Etiquetas finas desconocidas",
        )

    def derive_categories(self, fine_labels: Iterable[str]) -> tuple[str, ...]:
        fine = set(self.normalize_fine_labels(fine_labels))
        derived = {self.fine_label_mapping[label] for label in fine}
        return self.normalize_categories(derived)

    def migrate_legacy_categories(self, labels: Iterable[str]) -> tuple[str, ...]:
        legacy = set(
            self._canonicalize_identifiers(
                labels,
                self.legacy_coarse_mapping,
                error_label="Categorías históricas desconocidas",
            )
        )
        mapped = {self.legacy_coarse_mapping[label] for label in legacy}
        return self.normalize_categories(mapped)


def load_taxonomy(path: str | Path | None = None) -> TaxonomyContract:
    taxonomy_path = Path(path) if path else find_project_root() / "config" / "taxonomia_v2.json"
    payload = json.loads(taxonomy_path.read_text(encoding="utf-8-sig"))
    return TaxonomyContract.model_validate(payload)
