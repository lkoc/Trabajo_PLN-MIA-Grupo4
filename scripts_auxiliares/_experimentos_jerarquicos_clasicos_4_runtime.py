"""Compatibilidad mínima para deserializar las cabezas clásicas de 04_201.

Los artefactos fueron serializados cuando esta clase vivía en un módulo
dinámico con este nombre. Mantener una definición pequeña y estable evita
importar el pipeline completo de entrenamiento durante el despliegue.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CalibratedBinaryHead:
    family: str
    estimator: object
    calibrator: object
    calibration_folds: int

    def predict_score(self, features) -> np.ndarray:
        raw = np.asarray(self.estimator.decision_function(features), dtype=float)
        return self.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
