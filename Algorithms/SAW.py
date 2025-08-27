import numpy as np
import pandas as pd
from typing import List, Optional
from Algorithms.Entropy import entropy


class SAW:
    """Simple Additive Weighting (SAW) Multi-Criteria Decision Making yöntemi"""

    def __init__(self, decision_matrix, criteria_types, weights=None,
                 use_entropy=True, alternatives=None, criteria=None):
        self.decision_matrix = np.array(decision_matrix, dtype=float)
        self.criteria_types = criteria_types
        self.alternatives = alternatives or [f'A{i + 1}' for i in range(self.decision_matrix.shape[0])]
        self.criteria = criteria or [f'C{i + 1}' for i in range(self.decision_matrix.shape[1])]

        # Ağırlıkları hesapla
        if use_entropy or weights is None:
            self.weights = entropy(self.decision_matrix)
        else:
            self.weights = np.array(weights)

        # Hesaplamaları yap
        self.normalized_matrix = self._normalize_matrix()
        self.scores = self._calculate_scores()
        self.ranking = self._get_ranking()

    def _normalize_matrix(self):
        """Karar matrisini normalize eder"""
        normalized = np.zeros_like(self.decision_matrix, dtype=float)

        for j, criterion_type in enumerate(self.criteria_types):
            column = self.decision_matrix[:, j]

            if criterion_type == 'max':
                max_val = np.max(column)
                normalized[:, j] = column / max_val if max_val != 0 else 0
            else:  # min
                min_val = np.min(column)
                if min_val == 0:
                    max_val = np.max(column)
                    normalized[:, j] = (max_val - column) / max_val
                else:
                    normalized[:, j] = min_val / column

        return normalized

    def _calculate_scores(self):
        """SAW skorlarını hesaplar"""
        weighted_matrix = self.normalized_matrix * self.weights
        return np.sum(weighted_matrix, axis=1)

    def _get_ranking(self):
        """Sıralama yapar"""
        sorted_indices = np.argsort(self.scores)[::-1]
        ranking = []
        for rank, idx in enumerate(sorted_indices, 1):
            ranking.append({
                'rank': rank,
                'alternative': self.alternatives[idx],
                'score': float(self.scores[idx])
            })
        return ranking

    def get_results(self):
        """Sonuçları döndürür"""
        return {
            'weights': {self.criteria[i]: float(self.weights[i]) for i in range(len(self.criteria))},
            'scores': {self.alternatives[i]: float(self.scores[i]) for i in range(len(self.alternatives))},
            'ranking': self.ranking,
            'normalized_matrix': self.normalized_matrix.tolist()
        }