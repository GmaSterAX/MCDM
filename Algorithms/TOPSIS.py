import numpy as np
from Algorithms.Entropy import entropy


class TOPSIS:
    """TOPSIS (Technique for Order Preference by Similarity to Ideal Solution) yöntemi"""

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

        # TOPSIS hesaplamaları
        self.normalized_matrix = self._normalize_matrix()
        self.weighted_matrix = self._weight_matrix()
        self.ideal_best, self.ideal_worst = self._find_ideal_solutions()
        self.distances_best = self._calculate_distances(self.ideal_best)
        self.distances_worst = self._calculate_distances(self.ideal_worst)
        self.closeness_coefficients = self._calculate_closeness_coefficients()
        self.ranking = self._get_ranking()

    def _normalize_matrix(self):
        """Karar matrisini normalize eder (vektör normalizasyonu)"""
        normalized = np.zeros_like(self.decision_matrix, dtype=float)

        for j in range(self.decision_matrix.shape[1]):
            column = self.decision_matrix[:, j]
            norm = np.sqrt(np.sum(column ** 2))
            normalized[:, j] = column / norm if norm != 0 else 0

        return normalized

    def _weight_matrix(self):
        """Normalize matrise ağırlıkları uygular"""
        return self.normalized_matrix * self.weights

    def _find_ideal_solutions(self):
        """İdeal en iyi ve en kötü çözümleri bulur"""
        ideal_best = np.zeros(self.weighted_matrix.shape[1])
        ideal_worst = np.zeros(self.weighted_matrix.shape[1])

        for j, criterion_type in enumerate(self.criteria_types):
            if criterion_type == 'max':
                ideal_best[j] = np.max(self.weighted_matrix[:, j])
                ideal_worst[j] = np.min(self.weighted_matrix[:, j])
            else:  # min
                ideal_best[j] = np.min(self.weighted_matrix[:, j])
                ideal_worst[j] = np.max(self.weighted_matrix[:, j])

        return ideal_best, ideal_worst

    def _calculate_distances(self, ideal_solution):
        """İdeal çözüme olan Öklid mesafelerini hesaplar"""
        distances = np.zeros(self.weighted_matrix.shape[0])

        for i in range(self.weighted_matrix.shape[0]):
            distance = np.sqrt(np.sum((self.weighted_matrix[i, :] - ideal_solution) ** 2))
            distances[i] = distance

        return distances

    def _calculate_closeness_coefficients(self):
        """Yakınlık katsayılarını hesaplar"""
        closeness = np.zeros(self.decision_matrix.shape[0])

        for i in range(self.decision_matrix.shape[0]):
            if self.distances_best[i] + self.distances_worst[i] != 0:
                closeness[i] = self.distances_worst[i] / (self.distances_best[i] + self.distances_worst[i])
            else:
                closeness[i] = 0

        return closeness

    def _get_ranking(self):
        """Sıralama yapar (yakınlık katsayısına göre)"""
        sorted_indices = np.argsort(self.closeness_coefficients)[::-1]  # Büyükten küçüğe
        ranking = []
        for rank, idx in enumerate(sorted_indices, 1):
            ranking.append({
                'rank': rank,
                'alternative': self.alternatives[idx],
                'closeness_coefficient': float(self.closeness_coefficients[idx]),
                'distance_to_best': float(self.distances_best[idx]),
                'distance_to_worst': float(self.distances_worst[idx])
            })
        return ranking

    def get_results(self):
        """Sonuçları döndürür"""
        return {
            'weights': {self.criteria[i]: float(self.weights[i]) for i in range(len(self.criteria))},
            'closeness_coefficients': {self.alternatives[i]: float(self.closeness_coefficients[i]) for i in
                                       range(len(self.alternatives))},
            'ranking': self.ranking,
            'ideal_best': self.ideal_best.tolist(),
            'ideal_worst': self.ideal_worst.tolist()
        }