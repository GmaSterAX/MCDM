import numpy as np
from Algorithms.Entropy import entropy


class VIKOR:
    """VIKOR (VlseKriterijumska Optimizacija I Kompromisno Resenje) yöntemi"""

    def __init__(self, decision_matrix, criteria_types, weights=None, v=0.5,
                 use_entropy=True, alternatives=None, criteria=None):
        self.decision_matrix = np.array(decision_matrix, dtype=float)
        self.criteria_types = criteria_types
        self.v = v  # Çoğunluk kriteri ağırlığı (0.5 varsayılan)
        self.alternatives = alternatives or [f'A{i + 1}' for i in range(self.decision_matrix.shape[0])]
        self.criteria = criteria or [f'C{i + 1}' for i in range(self.decision_matrix.shape[1])]

        # Ağırlıkları hesapla
        if use_entropy or weights is None:
            self.weights = entropy(self.decision_matrix)
        else:
            self.weights = np.array(weights)

        # VIKOR hesaplamaları
        self.best_values, self.worst_values = self._find_best_worst()
        self.s_values = self._calculate_s_values()
        self.r_values = self._calculate_r_values()
        self.q_values = self._calculate_q_values()
        self.ranking = self._get_ranking()

    def _find_best_worst(self):
        """Her kriter için en iyi ve en kötü değerleri bulur"""
        best_values = np.zeros(self.decision_matrix.shape[1])
        worst_values = np.zeros(self.decision_matrix.shape[1])

        for j, criterion_type in enumerate(self.criteria_types):
            if criterion_type == 'max':
                best_values[j] = np.max(self.decision_matrix[:, j])
                worst_values[j] = np.min(self.decision_matrix[:, j])
            else:  # min
                best_values[j] = np.min(self.decision_matrix[:, j])
                worst_values[j] = np.max(self.decision_matrix[:, j])

        return best_values, worst_values

    def _calculate_s_values(self):
        """S değerlerini hesaplar (grup faydası)"""
        s_values = np.zeros(self.decision_matrix.shape[0])

        for i in range(self.decision_matrix.shape[0]):
            s_sum = 0
            for j in range(self.decision_matrix.shape[1]):
                if self.worst_values[j] != self.best_values[j]:
                    s_sum += self.weights[j] * (self.worst_values[j] - self.decision_matrix[i, j]) / (
                                self.worst_values[j] - self.best_values[j])
            s_values[i] = s_sum

        return s_values

    def _calculate_r_values(self):
        """R değerlerini hesaplar (bireysel pişmanlık)"""
        r_values = np.zeros(self.decision_matrix.shape[0])

        for i in range(self.decision_matrix.shape[0]):
            r_max = 0
            for j in range(self.decision_matrix.shape[1]):
                if self.worst_values[j] != self.best_values[j]:
                    r_val = self.weights[j] * (self.worst_values[j] - self.decision_matrix[i, j]) / (
                                self.worst_values[j] - self.best_values[j])
                    r_max = max(r_max, r_val)
            r_values[i] = r_max

        return r_values

    def _calculate_q_values(self):
        """Q değerlerini hesaplar (VIKOR indeksi)"""
        s_best = np.min(self.s_values)
        s_worst = np.max(self.s_values)
        r_best = np.min(self.r_values)
        r_worst = np.max(self.r_values)

        q_values = np.zeros(self.decision_matrix.shape[0])

        for i in range(self.decision_matrix.shape[0]):
            q1 = (self.s_values[i] - s_best) / (s_worst - s_best) if s_worst != s_best else 0
            q2 = (self.r_values[i] - r_best) / (r_worst - r_best) if r_worst != r_best else 0
            q_values[i] = self.v * q1 + (1 - self.v) * q2

        return q_values

    def _get_ranking(self):
        """Sıralama yapar (Q değerine göre)"""
        sorted_indices = np.argsort(self.q_values)  # Küçükten büyüğe (en iyi en küçük Q)
        ranking = []
        for rank, idx in enumerate(sorted_indices, 1):
            ranking.append({
                'rank': rank,
                'alternative': self.alternatives[idx],
                'q_value': float(self.q_values[idx]),
                's_value': float(self.s_values[idx]),
                'r_value': float(self.r_values[idx])
            })
        return ranking

    def get_results(self):
        """Sonuçları döndürür"""
        return {
            'weights': {self.criteria[i]: float(self.weights[i]) for i in range(len(self.criteria))},
            'q_values': {self.alternatives[i]: float(self.q_values[i]) for i in range(len(self.alternatives))},
            's_values': {self.alternatives[i]: float(self.s_values[i]) for i in range(len(self.alternatives))},
            'r_values': {self.alternatives[i]: float(self.r_values[i]) for i in range(len(self.alternatives))},
            'ranking': self.ranking
        }