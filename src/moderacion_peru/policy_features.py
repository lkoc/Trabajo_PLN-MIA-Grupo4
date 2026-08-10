from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from typing import Any

import numpy as np

POLICY_FEATURE_VERSION = "prompt-v3.2-heuristics-1"


def _fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    return "".join(
        character for character in normalized if not unicodedata.combining(character)
    )


_PATTERNS: tuple[tuple[str, str], ...] = (
    ("sexual_local", r"\b(cachar|cachando|tirar|coger|poto|potito|manosear|chapar)\b"),
    ("sexual_explicit", r"\b(sexo|sexual|penetraci[oó]n|desnudez|genital|pornograf)"),
    (
        "racism_local",
        r"\b(cholo|cholito|serrano|indio|chuncho|pune[nñ]o|blanqui[nñ]oso)\b",
    ),
    (
        "classism_education",
        r"\b(analfabet|no sabe (leer|escribir)|colegio estatal|universidad privada|pucp)\b",
    ),
    ("gender_slur", r"\b(maric[oó]n|cabro|machona|puta|perra|feminazi)\b"),
    (
        "directed_abuse",
        r"\b(hijo de puta|concha de tu (madre|abuela)|cholo de mierda|maricon de mierda|maldito perro|eres una mierda)\b",
    ),
    (
        "threat",
        r"\b(te voy a matar|s[eé] d[oó]nde vives|vas a morir|te voy a pegar|si denuncias|te encontrar[eé])\b",
    ),
    (
        "attribution_report",
        r"\b(dijo|afirm[oó]|declar[oó]|cont[oó]|relat[oó]|denunci[oó]|report[oó]|seg[uú]n|la v[ií]ctima|le dijeron)\b",
    ),
    (
        "condemnation",
        r"\b(conden[oó]|rechaz[oó]|critic[oó]|denunci[oó]|no se debe|es discriminaci[oó]n|es acoso)\b",
    ),
    (
        "informational",
        r"\b(noticia|informe|investigaci[oó]n|estudio|educaci[oó]n sexual|cl[ií]nic|jur[ií]dic)\b",
    ),
    (
        "affiliative",
        r"\b(amigo|amiga|causa|pata|cari[nñ]o|broma|entre amigos|mi familia)\b",
    ),
    ("second_person", r"\b(t[uú]|usted|ustedes|te|tuya?|contigo)\b"),
    ("negation", r"\b(no|nunca|jam[aá]s|ni)\b"),
    ("quotation", r"[\"“”'‘’]|\bdijo que\b|\bseg[uú]n\b"),
)


class PolicyCueTransformer:
    """Rasgos auditables derivados del prompt, sin fingir condicionamiento LLM.

    Las expresiones son disparadores contextuales, nunca reglas automáticas de
    clasificación. El clasificador supervisado aprende su peso y combinación.
    """

    def __init__(self, *, version: str = POLICY_FEATURE_VERSION) -> None:
        self.version = version
        self._compiled = tuple(
            (name, re.compile(pattern)) for name, pattern in _PATTERNS
        )

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {"version": self.version}

    def set_params(self, **parameters: Any) -> PolicyCueTransformer:
        for name, value in parameters.items():
            setattr(self, name, value)
        return self

    def fit(self, values: Sequence[str], y: Any = None) -> PolicyCueTransformer:
        return self

    def transform(self, values: Sequence[str]) -> Any:
        from scipy.sparse import csr_matrix

        features: list[list[float]] = []
        for value in values:
            text = _fold(str(value))
            tokens = max(1, len(text.split()))
            row = [
                float(len(pattern.findall(text))) for _name, pattern in self._compiled
            ]
            row.extend(
                (
                    min(1.0, text.count("!") / 3),
                    min(1.0, text.count("?") / 3),
                    min(1.0, len(text) / 500),
                    min(1.0, tokens / 100),
                )
            )
            features.append(row)
        return csr_matrix(np.asarray(features, dtype=np.float32))

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(
            [
                *(name for name, _pattern in _PATTERNS),
                "exclamation",
                "question",
                "characters",
                "tokens",
            ],
            dtype=object,
        )
