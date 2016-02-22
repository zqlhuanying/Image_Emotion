# encoding: utf-8
from __future__ import division
import numpy as np
import scipy

__author__ = 'zql'
__date__ = '16-02-15'

ALPHA = 0.1


def f_test(X, Y):
    """
    F Test
    H0: Var(X) == Var(Y) 无显著性差异
    :param X: list or numpy array
    :param Y: list or numpy array
    :return:
    """
    X, Y = _check_array(X, Y)
    # 计算样本标准偏差
    x_std = np.std(X, ddof=1)
    y_std = np.std(Y, ddof=1)

    # Or whatever you want your alpha to be.
    dfx = X.size - 1
    dfy = Y.size - 1
    f = np.power(x_std, 2) / np.power(y_std, 2)
    # when you calculate the cdf as you're doing with Python, that's inherently a one-sided test.
    # The one-sided p-value of a statistic F is either cdf(F) or 1 - cdf(F) depending on what side of the mean F lies.
    # You're trying to measure the probability of the statistic being "more extreme" than what's observed
    # if F is on the left of the mean, "more extreme" means "further to the left", so cdf(F).
    # If F is greater than the mean, then "more extreme" means "further to the right", so 1-cdf(F)
    p_value = scipy.stats.f.cdf(f, dfx, dfy)
    if p_value > 0.5:
        p_value = 1 - p_value
    return _choice(p_value)


def levene_test(X, Y):
    """
    Levene Test
    The null hypothesis that all input samples are from populations with equal variances
    :param X:
    :param Y:
    :return:
    """
    X, Y = _check_array(X, Y)
    statistic, p_value = scipy.stats.levene(X, Y)
    return _choice(p_value)


def pair_test(X, Y):
    """
    Pair-T Test
    :param X:
    :param Y:
    :return:
    """
    X, Y = _check_array(X, Y)
    statistic, p_value = scipy.stats.ttest_rel(X, Y)
    return _choice(p_value)


def _check_array(X, Y):
    X = np.asanyarray(X)
    Y = np.asanyarray(Y)
    return X, Y


def _choice(pval):
    if pval < ALPHA:
        # Reject the null hypothesis that is H0
        return False
    else:
        return True
