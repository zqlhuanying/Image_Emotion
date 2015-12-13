# encoding: utf-8
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.extmath import safe_sparse_dot

__author__ = 'root'
__date__ = '15-12-13'


class Bayes(MultinomialNB):
    def __init__(self, alpha=0.01):
        super(Bayes, self).__init__(alpha)

    def _joint_log_likelihood(self, X):
        ssd = safe_sparse_dot(X, self.feature_log_prob_.T)
        rows, cols = ssd.shape
        res = np.zeros((rows, cols), dtype=np.float64)

        for i, row in enumerate(ssd):
            s = np.dot(self.class_log_prior_.reshape(-1, 1), row.reshape(1, -1))
            res[i] = np.diag(s)

        return res


if __name__ == "__main__":
    print Bayes.__dict__
    print Bayes.__class__
    print dir(Bayes)
