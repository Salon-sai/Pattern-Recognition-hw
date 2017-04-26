# -*- coding: utf-8 -*-

from os import listdir, path
import numpy as np
import codecs
import json

def parse_data(in_file):
    """
    获取文件中的数据并转变成一位数组
    :param in_file: 文件名
    :return: 一位数组
    """
    with codecs.open(in_file, 'rb') as fi:
        fi.readline()
        data = [float(line.decode('utf8')) for line in fi if len(line) > 0]
    return data

def get_data_set(set_type="train", category="pos"):
    """
    根据类型和训练集获取数据集
    :param set_type: 数据集类别：训练和测试
    :param category: 便签类别：pos和neg
    :return: 数据集合
    """
    data_set = []
    dir_name = set_type + "/" + category
    for name in listdir(dir_name):
        data_set.append(parse_data(dir_name + "/" + name))

    return data_set

if path.isfile('train_pos_data.json'):
    with codecs.open('train_pos_data.json', 'rb') as fi:
        train_pos_data = json.load(fi)
else:
    train_pos_data = get_data_set()
    with codecs.open('train_pos_data.json', 'w') as fi:
        json.dump(train_pos_data, fi)

if path.isfile('train_neg_data.json'):
    with codecs.open('train_neg_data.json', 'rb') as fi:
        train_neg_data = json.load(fi)
else:
    train_neg_data = get_data_set(category="neg")
    with codecs.open('train_neg_data.json', 'w') as fi:
        json.dump(train_neg_data, fi)

train_pos_labels = np.ones((len(train_pos_data), 1), dtype=int)
train_neg_labels = -1 * np.ones((len(train_neg_data), 1), dtype=int)

if path.isfile('test_pos_data.json'):
    with codecs.open('test_pos_data.json', 'rb') as fi:
        test_pos_data = json.load(fi)
else:
    test_pos_data = get_data_set(set_type="test")
    with codecs.open('test_pos_data.json', 'w') as fi:
        json.dump(test_pos_data, fi)

if path.isfile('test_neg_data.json'):
    with codecs.open('test_neg_data.json', 'rb') as fi:
        test_neg_data = json.load(fi)
else:
    test_neg_data = get_data_set("test", "neg")
    with codecs.open('test_neg_data.json', 'w') as fi:
        json.dump(test_neg_data, fi)

test_pos_labels = np.ones((len(test_pos_data), 1), dtype=int)
test_neg_labels = -1 * np.ones((len(test_neg_data), 1), dtype=int)

all_train_data = np.matrix(np.concatenate((train_pos_data, train_neg_data)))
all_train_label = np.concatenate((train_pos_labels, train_neg_labels))

all_test_data = np.concatenate((test_pos_data, test_neg_data))
all_test_label = np.concatenate((test_pos_labels, test_neg_labels))
