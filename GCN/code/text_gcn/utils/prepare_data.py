#!/usr/bin/python
# -*-coding:utf-8-*-
import pandas as pd
import os
"""构建自定义数据集"""
DATA_PATH = os.path.join(os.path.abspath('..'), 'data')


def prepare_data(dataset_name, sentences, labels, train_or_test_list):
    """
    将数据集处理为如下格式输入：
    args:
    dataset_name = 'own'
    sentences = ['Would you like a plain sweater or something else?​','...',..]
    labels = ['Yes', 'No', ...]
    train_or_test_list = ['train', 'test',...]
    """

    meta_data_list = []

    for i in range(len(sentences)):
        meta = str(i) + '\t' + train_or_test_list[i] + '\t' + labels[i]
        meta_data_list.append(meta)

    meta_data_str = '\n'.join(meta_data_list)

    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    with open(os.path.join(DATA_PATH, '{}.txt'.format(dataset_name)), 'w', encoding='utf-8') as f:
        f.write(meta_data_str)

    corpus_str = '\n'.join(sentences)

    if not os.path.exists(os.path.join(DATA_PATH, 'corpus')):
        os.makedirs(os.path.join(DATA_PATH, 'corpus'))
    with open(os.path.join(DATA_PATH, 'corpus/{}.txt'.format(dataset_name)), 'w', encoding='utf-8') as f:
        f.write(corpus_str)


def load_data(train_path, test_path):
    """加载数据"""
    sentences = []
    labels = []
    train_or_test_list = []
    for line in open(train_path, 'r', encoding='utf-8'):
        line = line.strip()
        text, label = line.split("\t")
        sentences.append(text)
        labels.append(label)
        train_or_test_list.append("train")
    for line in open(test_path, 'r', encoding='utf-8'):
        line = line.strip()
        text, label = line.split("\t")
        sentences.append(text)
        labels.append(label)
        train_or_test_list.append("test")

    return sentences, labels, train_or_test_list


if __name__ == '__main__':
    # Weibo
    # Train_path = "../data/Weibo/train.txt"
    # Test_path = "../data/Weibo/test.txt"
    # sentences, labels, train_or_test_list = load_data(Train_path, Test_path)
    # dataset_name = 'Weibo'
    # prepare_data(dataset_name, sentences, labels, train_or_test_list)

    # R8
    Train_path = "../data/R8/train.txt"
    Test_path = "../data/R8/test.txt"
    sentences, labels, train_or_test_list = load_data(Train_path, Test_path)
    dataset_name = 'R8'
    prepare_data(dataset_name, sentences, labels, train_or_test_list)