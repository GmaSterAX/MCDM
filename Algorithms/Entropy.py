import numpy as np


def col_sum(decision_matrix):
    return matrix.sum(axis=0)


def entropy(decision_matrix):
    """

    :param decision_matrix: 2D matrix
    :return: weigths of criteria

    """
    p_matrix = np.array(decision_matrix, dtype=float)
    m, n = decision_matrix.shape

    col_sum = decision_matrix.sum(axis=0)
    col_sums = np.where(col_sum == 0, 1, col_sum)

    # Normalized values of the matrix
    p_matrix = decision_matrix / col_sums

    # Shannon Entropy values of each criterion
    k = 1 / np.log(m)
    p_log_p = np.zeros_like(p_matrix)
    mask = p_matrix > 0                                             #Apply the log operation just for the values > 0
    p_log_p[mask] = p_matrix[mask] * np.log(p_matrix[mask])
    Ej = -k * np.sum(p_log_p, axis=0)

    # uncertainity value (deviation index)
    d = 1 - Ej

    # Weigth calculation
    wj = d / np.sum(d)

    return wj



