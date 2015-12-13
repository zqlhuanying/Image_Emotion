# encoding:utf-8
import urllib
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics import precision_score
from sklearn.naive_bayes import MultinomialNB

__author__ = 'root'
__date__ = '15-11-28'

#news = fetch_20newsgroups(subset="all")
#print news.keys()
#print "news.data 的 type : %s" % type(news.data)
#print "news.target 的 type : %s " % type(news.target)
#print "news.data length : %f " % len(news.data)
#print "news.target length : %f " % len(news.target)
#print "news.target 是个 ndarray 数组，也就是训练集中每个文本所对应的类别"


# csr: scipy sparse matrix
# the standard CSR representation
# where the column indices for row i are stored in
# indices[indptr[i]:indptr[i+1]] and their corresponding values are stored in data[indptr[i]:indptr[i+1]].
#docs = [["hello", "world", "hello"], ["goodbye", "cruel", "world"]]
#indptr = [0]
#indices = []
#data = []
#vocabulary = {}
#for d in docs:
#    for term in d:
#        index = vocabulary.setdefault(term, len(vocabulary))
#        indices.append(index)
#        data.append(1)
#    indptr.append(len(indices))
#
#aa = csr_matrix((data, indices, indptr), dtype=int).toarray()
#print indices[indptr[0]:indptr[1]]

# test MultinomialNB in sklearn
# n_features 尽量取大点，以免发生 Hash 碰撞
# 就算取得很大也无妨，因为是稀疏矩阵，还是消耗不了多少内存
train_datas = [{'monkey': 1, 'dog': 1, 'cat': 2, 'elephant': 4}, {'dog': 2, 'run': 5}]
feature_hasher = FeatureHasher(n_features=2 ** 20, non_negative=True)
train_datas = feature_hasher.transform(train_datas).toarray()
"""X = np.array([[1, 2, 4, 1, 1, 1],
 [3, 2, 4, 2, 2, 3],
 [2, 2, 3, 4, 4, 1],
 [2, 0, 3, 2, 3, 1],
 [2, 0, 0, 3, 3, 3],
 [2, 3, 1, 0, 3, 4]])"""
class_label = np.array([1, 2])
# 调整平滑因子
clf = MultinomialNB(alpha=0.01)
train = clf.fit(train_datas, class_label)
test_datas = [{'monkey': 3, 'mouse': 1}]
test_datas = feature_hasher.transform(test_datas).toarray()
test = clf.predict(test_datas)
print train_datas
print test_datas
print train
print test
print clf._joint_log_likelihood(test_datas)
print clf.__dict__


#test metrics in sklearn
y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 1]
print "macro:", precision_score(y_true, y_pred, average='macro')
print "micro:", precision_score(y_true, y_pred, average='micro')
print "none:", precision_score(y_true, y_pred, average=None)
