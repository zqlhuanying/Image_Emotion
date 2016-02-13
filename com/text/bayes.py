# encoding: utf-8
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction import FeatureHasher
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.extmath import safe_sparse_dot

__author__ = 'root'
__date__ = '15-12-13'


class Bayes(MultinomialNB):
    def __init__(self, alpha=0.01):
        super(Bayes, self).__init__(alpha)


if __name__ == "__main__":
    print Bayes.__dict__
    print Bayes.__class__
    print dir(Bayes)
