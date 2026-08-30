"""Recipe RAG modules, imported lazily so offline indexing has no LLM dependency."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "DataPreparationModule",
    "IndexConstructionModule",
    "RetrievalOptimizationModule",
    "GenerationIntegrationModule",
]
__version__ = "0.1.0"

_MODULES = {
    "DataPreparationModule": ".data_preparation",
    "IndexConstructionModule": ".index_construction",
    "RetrievalOptimizationModule": ".retrieval_optimization",
    "GenerationIntegrationModule": ".generation_integration",
}


def __getattr__(name: str) -> Any:
    if name not in _MODULES:
        raise AttributeError(name)
    module = import_module(_MODULES[name], __name__)
    return getattr(module, name)
