import numpy as np
from scipy.linalg import eig


class AHP:
    """Analytic Hierarchy Process (AHP)"""

    def __init__(self, pairwise_matrix, decision_matrix, criteria_types, alternatives=None, criteria=None):
        self.pairwise_matrix = np.array(pairwise_matrix, dtype=float)
        self.decision_matrix = np.array(decision_matrix, dtype=float)
        self.criteria_types = criteria_types
        self.alternatives = alternatives or [f'A{i + 1}' for i in range(self.decision_matrix.shape[0])]
        self.criteria = criteria or [f'C{i + 1}' for i in range(self.decision_matrix.shape[1])]

        # Calculation Process
        self.weights = self._calculate_weights()
        self.consistency_ratio = self._calculate_consistency_ratio()
        self.normalized_matrix = self._normalize_matrix()
        self.scores = self._calculate_scores()
        self.ranking = self._get_ranking()

    def _calculate_weights(self):
        """Weight Calculation using Eigenvalue Method"""
        # Eğer pairwise matrix birim matris ise, entropy method kullan
        if np.allclose(self.pairwise_matrix, np.ones_like(self.pairwise_matrix)):
            return self._calculate_weights_entropy()

        eigenvalues, eigenvectors = eig(self.pairwise_matrix)
        max_eigenvalue_index = np.argmax(eigenvalues.real)
        principal_eigenvector = eigenvectors[:, max_eigenvalue_index].real

        # Negatif değerleri pozitif yap
        if np.sum(principal_eigenvector) < 0:
            principal_eigenvector = -principal_eigenvector

        weights = np.abs(principal_eigenvector) / np.sum(np.abs(principal_eigenvector))
        return weights

    def _calculate_weights_entropy(self):
        """Entropy-based weight calculation when pairwise matrix is not available"""
        # Decision matrix'i normalize et
        normalized_matrix = self.decision_matrix.copy()

        # Her kriter için normalizasyon
        for j in range(normalized_matrix.shape[1]):
            column_sum = np.sum(normalized_matrix[:, j])
            if column_sum > 0:
                normalized_matrix[:, j] = normalized_matrix[:, j] / column_sum

        # Entropy hesapla
        m, n = normalized_matrix.shape
        entropy = np.zeros(n)

        for j in range(n):
            entropy_sum = 0
            for i in range(m):
                if normalized_matrix[i, j] > 0:
                    entropy_sum += normalized_matrix[i, j] * np.log(normalized_matrix[i, j])
            entropy[j] = -entropy_sum / np.log(m) if m > 1 else 0

        # Diversity degree
        diversity = 1 - entropy

        # Weights
        diversity_sum = np.sum(diversity)
        weights = diversity / diversity_sum if diversity_sum > 0 else np.ones(n) / n

        return weights

    def _calculate_consistency_ratio(self):
        """Consistency Ratio Calculation"""
        n = self.pairwise_matrix.shape[0]
        if n <= 2 or np.allclose(self.pairwise_matrix, np.ones_like(self.pairwise_matrix)):
            return 0.0

        try:
            # Lambda max
            weighted_sum = np.dot(self.pairwise_matrix, self.weights)
            lambda_max = np.mean(weighted_sum / (self.weights + 1e-10))  # Divide by zero koruması

            # CI (Consistency Index)
            ci = (lambda_max - n) / (n - 1)

            # RI (Random Index)
            ri_values = {3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
            ri = ri_values.get(n, 1.49)

            return ci / ri if ri != 0 else 0.0
        except:
            return 0.0

    def _normalize_matrix(self):
        """Normalizing Decision Matrix"""
        normalized = np.zeros_like(self.decision_matrix, dtype=float)

        for j, criterion_type in enumerate(self.criteria_types):
            column = self.decision_matrix[:, j]

            if criterion_type in ['benefit', 'max']:
                # Benefit criteria - büyük değer iyi
                max_val = np.max(column)
                normalized[:, j] = column / max_val if max_val != 0 else np.zeros_like(column)
            elif criterion_type in ['cost', 'min']:
                # Cost criteria - küçük değer iyi
                min_val = np.min(column)
                if min_val == 0:
                    # Eğer minimum 0 ise, inverse transformation kullan
                    max_val = np.max(column)
                    normalized[:, j] = (max_val - column) / max_val if max_val != 0 else np.zeros_like(column)
                else:
                    # Normal inverse transformation
                    normalized[:, j] = min_val / column
            else:
                # Default olarak benefit kabul et
                max_val = np.max(column)
                normalized[:, j] = column / max_val if max_val != 0 else np.zeros_like(column)

        return normalized

    def _calculate_scores(self):
        """AHP Score Calculation"""
        # Ağırlıklı normalize matris
        weighted_matrix = self.normalized_matrix * self.weights.reshape(1, -1)
        # Her alternatif için toplam skor
        scores = np.sum(weighted_matrix, axis=1)
        return scores

    def _get_ranking(self):
        """Ranking based on scores"""
        sorted_indices = np.argsort(self.scores)[::-1]  # Descending order
        ranking = []
        for rank, idx in enumerate(sorted_indices, 1):
            ranking.append({
                'rank': rank,
                'alternative': self.alternatives[idx],
                'score': float(self.scores[idx])
            })
        return ranking

    def get_results(self):
        """Get all results"""
        # Simple ranking for compatibility
        simple_ranking = [item['alternative'] for item in self.ranking]

        return {
            'weights': {self.criteria[i]: float(self.weights[i]) for i in range(len(self.criteria))},
            'scores': {self.alternatives[i]: float(self.scores[i]) for i in range(len(self.alternatives))},
            'ranking': simple_ranking,
            'detailed_ranking': self.ranking,
            'consistency_ratio': float(self.consistency_ratio),
            'is_consistent': self.consistency_ratio < 0.1,
            'normalized_matrix': self.normalized_matrix.tolist()
        }