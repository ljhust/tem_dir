# -*- coding: utf-8 -*-
# @Time    : 2020/6/29 13:38
# @Author  : Lijing
# @File    : bert_test.py

import operator
import os
import sys
import time

from transformers import pipeline

sys.path.append('../..')
from pycorrector.utils.text_utils import is_chinese_string, convert_to_unicode
from pycorrector.utils.logger import logger
from pycorrector.corrector import Corrector
from bert_corrector import BertCorrector

pwd_path = os.path.abspath(os.path.dirname(__file__))


def get_data():
    f1_lines = open("toLJ/open.txt", encoding='utf8').readlines()
    f2_lines = open("toLJ/public.txt", encoding='utf8').readlines()
    f3_lines = open("toLJ/yyzz.txt", encoding='utf8').readlines()

    address = []
    entities = []

    for i, line in enumerate(f1_lines):
        if line.strip() == "存款人名称":
            if f1_lines[i + 1].strip() != "None":
                entities.append(f1_lines[i + 1].strip())
        if line.strip() == "地址":
            if f1_lines[i + 1].strip() != "None":
                address.append(f1_lines[i + 1].strip())

    for i, line in enumerate(f2_lines):
        if line.strip() == "企业名称":
            if f1_lines[i + 1].strip() != "None":
                entities.append(f1_lines[i + 1].strip())
        if line.strip() == "住所":
            if f1_lines[i + 1].strip() != "None":
                address.append(f1_lines[i + 1].strip())

    for i, line in enumerate(f3_lines):
        if line.strip() == "名称":
            if f1_lines[i + 1].strip() != "None":
                entities.append(f1_lines[i + 1].strip())
        if line.strip() == "地址":
            if f1_lines[i + 1].strip() != "None":
                address.append(f1_lines[i + 1].strip())

    return entities, address

entities, address = get_data()

d = BertCorrector()
for sent in entities:
    corrected_sent, err = d.bert_correct(sent)
    print("original sentence:{} => {}, err:{}".format(sent, corrected_sent, err))