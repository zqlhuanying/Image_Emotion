# encoding: utf-8
from sklearn.feature_extraction import FeatureHasher

__author__ = 'zql'
__date__ = '2015/11/10'

# 特征词 Hash 散列器
Feature_Hasher = FeatureHasher(n_features=600000, non_negative=True)
