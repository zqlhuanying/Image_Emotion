# encoding: utf-8
import os
from com import PROJECTPATH

__author__ = 'zql'
__date__ = '2016/06/12'

# resource path
RESOURCE_BASE_URL = os.path.join(PROJECTPATH, "resource")
TEXT_RESOURCE = os.path.join(RESOURCE_BASE_URL, "text")
IMAGE_RESOURCE = os.path.join(RESOURCE_BASE_URL, "image")
TEST_RESOURCE = os.path.join(RESOURCE_BASE_URL, "test")

# result path
OUT_BASE_URL = os.path.join(PROJECTPATH, "out")
TEXT_OUT = os.path.join(OUT_BASE_URL, "text")
IMAGE_OUT = os.path.join(OUT_BASE_URL, "image")

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
