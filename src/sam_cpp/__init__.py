from __future__ import annotations

from ._core import __doc__, __version__, SamPredictor
from .utils import download_sam_model_to_cache

__all__ = ["__doc__", "__version__", "SamPredictor"]


def create_sam_predictor() -> SamPredictor:
    model_path = download_sam_model_to_cache("sam_cpp")
    predictor = SamPredictor()
    predictor.load_model(model_path)
    return predictor
