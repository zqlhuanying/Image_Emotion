# encoding: utf-8
import os

__author__ = 'zql'
__date__ = '2015/11/4'


RESOURCE_BASE_URL = os.path.dirname(__file__) + "/resource" + "/"
TEST_BASE_URL = os.path.dirname(os.path.dirname(__file__)) + "/test" + "/"

# 情感类别
EMOTION_CLASS = {
    "anger": "愤怒",
    "disgust": "厌恶",
    "fear": "恐惧",
    "happiness": "高兴",
    "like": "喜好",
    "sadness": "悲伤",
    "surprise": "惊讶"
}

# 主客观类别
OBJECTIVE_CLASS = {
    "Y": "主观句",
    "N": "客观句"
}
