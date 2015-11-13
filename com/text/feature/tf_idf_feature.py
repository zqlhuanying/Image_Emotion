# encoding: utf-8
import pynlpir as nlpir
from textblob import TextBlob
from com.text.feature.Feature import Feature

__author__ = 'zql'
__date__ = '2015/11/12'


class TFIDFFeature(Feature):
    """
    文本的 TF_IDF 特征
    """
    def __init__(self):
        super(TFIDFFeature, self).__init__()

    def get_key_words(self, s):
        nlpir.open()
        words = nlpir.segment(s)
        print len(words)
        for word in words:
            print ("%s" % word[0])
        for word in nlpir.get_key_words(s, weighted=True):
            print ("(%s, %f)" % (word[0], word[1]))
        nlpir.close()

if __name__ == "__main__":
    ss = "就卡回收的看垃圾食品都就抛弃我kl；sa'j'd拉萨"
    TFIDFFeature().get_key_words(ss)
