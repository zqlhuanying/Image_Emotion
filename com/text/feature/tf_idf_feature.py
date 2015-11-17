# encoding: utf-8
from __future__ import division
from compiler.ast import flatten
import math
from com.text.constant import RESOURCE_BASE_URL
from com.text.feature.feature import Feature
from com.text.load_sample import Load
from com.text.split_words import SplitWords

__author__ = 'zql'
__date__ = '2015/11/12'


class TFIDFFeature(Feature):
    """
    文本的 TF_IDF 特征
    """
    def __init__(self):
        super(TFIDFFeature, self).__init__()

    def get_key_words(self, sentences, based=True):
        """
        以 sentences 为基础， 计算每个 sentence 的关键词
        若 based 为 True 则以训练集中的数据为基础来计算 TFIDF
        """
        sentence_list = list()
        sentence_list.append(sentences)
        if based is True:
            sample_url = RESOURCE_BASE_URL + "weibo_samples.xml"
            training_datas = Load.load_training(sample_url)
            sentence_list.append([data.get("sentence") for data in training_datas])

        splited_words_list = list()
        SplitWords.__init__()
        for sentence in flatten(sentence_list):
            splited_words_list.append(SplitWords.split_words(sentence))
        SplitWords.close()

        for splited_words in splited_words_list:
            print
            scores = {splited_word: self.tfidf(splited_word, splited_words, splited_words_list)
                      for splited_word in set(splited_words)}
            sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            for word, score in sorted_words[:min(10, len(sorted_words))]:
                print("\tWord: %s, TF-IDF: %f" % (word.decode("utf_8"), score))

    def tf(self, word, words):
        return words.count(word) / len(words)

    def idf(self, word, wordslist):
        return math.log(len(wordslist) / (1 + self.__n_contains(word, wordslist)))

    def tfidf(self, word, words, wordslist):
        return self.tf(word, words) * self.idf(word, wordslist)

    def __n_contains(self, word, wordslist):
        return sum(1 for words in wordslist if word in words)

if __name__ == "__main__":
    s1 = r"NLPIR分词系统前身为2000年发布的ICTCLAS词法分析系统，从2009年开始，为了和以前工作进行大的区隔，" \
        r"并NLPIR自然语言处理与信息检索共享平台，调整命名为NLPIR分词系统。"
    s2 = r"NLPIR分词系统前身为2000年发布的NLPIR词法分析系统，从2009年开始，为了和以前工作进行大的区隔，" \
         r"并NLPIR自然语言处理与信息检索共享平台，调整命名为NLPIR分词。"
    s3 = r"NLPIR分词系统前身为2000年发布的词法分析系统，从2009年开始，为了和以前工作进行大的区隔，" \
         r"并推广NLPIR，调整命名为NLPIR分词系统。"
    TFIDFFeature().get_key_words([s1, s2, s3])

#    def count(word, words):
#        c = 0
#        for w in words:
#            if w == word:
#                c += 1
#        return c
#
#    nlpir.open()
#    words = nlpir.segment(s, False)
#    for word in words:
#        p = count(word, words) * 1.0 / len(words)
#        print word, (p * math.log(1/p, 2) + (1 - p) * math.log(1/(1 - p), 2))

    """
    test TextBlob
    """
#    def tf(word, blob):
#        return blob.words.count(word) / len(blob.words)
#
#    def n_containing(word, bloblist):
#        return sum(1 for blob in bloblist if word in blob)
#
#    def idf(word, bloblist):
#        return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))
#
#    def tfidf(word, blob, bloblist):
#        return tf(word, blob) * idf(word, bloblist)
#
#    document1 = TextBlob("""Python is a 2000 made-for-TV horror movie directed by Richard
#                    Clabaugh. The film features several cult favorite actors, including William
#                     The Karate Kid fame, Wil Wheaton, Casper Van Dien, Jenny McCarthy,
#                    ogan, Robert Englund (best known for his role as Freddy Krueger in the
#                    are on Elm Street series of films), Dana Barron, David Bowe, and Sean
#                    The film concerns a genetically engineered snake, a python, that
#                    and unleashes itself on a small town. It includes the classic final
#                    nario evident in films like Friday the 13th. It was filmed in Los Angeles,
#                    nia and Malibu, California. Python was followed by two sequels: Python
#                    2) and Boa vs. Python (2004), both also made-for-TV films.""")
#
#    document2 = TextBlob("""Python, from the Greek word (πύθων/πύθωνας), is a genus of
#                    nonvenomous pythons[2] found in Africa and Asia. Currently, 7 species are
#                    recognised.[2] A member of this genus, P. reticulatus, is among the longest
#                    snakes known.""")
#
#    document3 = TextBlob("""The Colt Python is a .357 Magnum caliber revolver formerly
#                    manufactured by Colt's Manufacturing Company of Hartford, Connecticut.
#                    s sometimes referred to as a "Combat Magnum".[1] It was first introduced
#                    955, the same year as Smith &amp; Wesson's M29 .44 Magnum. The now discontinued
#                     Python targeted the premium revolver market segment. Some firearm
#                    ectors and writers such as Jeff Cooper, Ian V. Hogg, Chuck Hawks, Leroy
#                    pson, Renee Smeets and Martin Dougherty have described the Python as the
#                    st production revolver ever made.""")
#    bloblist = [document1, document2, document3]
#    print document1.words, len(document1.words)
#    for blob in bloblist:
#        tf_score = [tf(word, blob) for word in blob.words if word == "films"]
#        idf_score = [idf(word, bloblist) for word in blob.words if word == "films"]
#        tfidf_score = [tfidf(word, blob, bloblist) for word in blob.words if word == "films"]
#        print blob.words.count("films")
#        for score in tf_score:
#            print score
#
#        for score in idf_score:
#            print score
#        for score in tfidf_score:
#            print score

