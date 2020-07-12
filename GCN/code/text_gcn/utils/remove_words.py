from utils import remove_words
from utils import read_stopwords
import sys
import jieba
import os

DATASETS = ['R8', 'Weibo']  # 添加自己的数据集
"""自定义文本预处理"""

DATA_PATH = os.path.join(os.path.abspath('..'), 'data')


# 命令行参数读取
if len(sys.argv) != 2:
    sys.exit("Use: python remove_words.py <dataset>")



dataset = sys.argv[1]

if dataset not in DATASETS:
    sys.exit("wrong dataset name, DATASETS添加自定义数据集")

if dataset == 'R8':
    # 读取停用词
    stop_words = read_stopwords(os.path.join(DATA_PATH, 'stopwords.txt'))

    # 读取文档内容
    doc_content_list = []

    with open(os.path.join(DATA_PATH, 'corpus/{}.txt'.format(dataset)), 'rb') as f:
        for line in f.readlines():
            doc_content_list.append(line.strip().decode('latin1'))

    clean_corpus_str = remove_words(doc_content_list, lan='en', stop_words=stop_words)

    with open(os.path.join(DATA_PATH, 'corpus/{}.clean.txt'.format(dataset)), 'w', encoding='utf-8') as f:
        f.write(clean_corpus_str)

elif dataset == "Weibo":
    # 读取文档内容
    doc_content_list = []
    with open(os.path.join(DATA_PATH, 'corpus/{}.txt'.format(dataset)), 'r', encoding='utf-8') as f:
        for line in f.readlines():
            doc_content_list.append(line.strip())

    # 分词
    doc_words_list = []
    for line in doc_content_list:
        line_list = [w for w in list(jieba.cut(line)) if w != ' ']
        line_str = ' '.join(line_list)
        doc_words_list.append(line_str)

    clean_corpus_str = remove_words(doc_words_list, lan='ch')

    with open(os.path.join(DATA_PATH, 'corpus/{}.clean.txt'.format(dataset)), 'w', encoding='utf-8') as f:
        f.write(clean_corpus_str)

else:
    sys.exit("dataset has no pre_processing")


