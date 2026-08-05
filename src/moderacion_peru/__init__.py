"""Núcleo reproducible del moderador semiautomático peruano."""

from .schemas import AnnotationRecord, ModelRegistryEntry, ReviewEvent, RunManifest
from .taxonomy import TaxonomyContract, load_taxonomy

__all__ = [
    "AnnotationRecord",
    "ModelRegistryEntry",
    "ReviewEvent",
    "RunManifest",
    "TaxonomyContract",
    "load_taxonomy",
]

__version__ = "2.1.0"
