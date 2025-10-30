# recommender/similarity.py
import numpy as np
from abc import ABC, abstractmethod
from typing import List

class SimilarityCalculator(ABC):
    @abstractmethod
    def calculate(self, vec1: List[float], vec2: List[float]) -> float:
        pass


class CosineSimilarityCalculator(SimilarityCalculator):
    """
    Simple cosine similarity calculator for embeddings.
    """

    def calculate(self, vec1: List[float], vec2: List[float]) -> float:
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2)
