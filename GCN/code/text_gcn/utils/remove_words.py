import nltk
from utils import remove_words
from utils import read_stopwords
import sys
import jieba

"""自定义文本预处理"""

# 命令行参数读取
if len(sys.argv) != 2:
    sys.exit("Use: python remove_words.py <dataset>")

datasets = ['R8', 'Weibo']  # 添加自己的数据集
dataset = sys.argv[1]

if dataset not in datasets:
    sys.exit("wrong dataset name")

if dataset == 'R8':
    # 读取停用词
    stop_words = read_stopwords('../data/stopwords.txt')
    # print(stop_words)

    # 读取文档内容
    doc_content_list = []
    f = open('../data/corpus/' + dataset + '.txt', 'rb')
    for line in f.readlines():
        doc_content_list.append(line.strip().decode('latin1'))
    f.close()

    clean_corpus_str = remove_words(doc_content_list, lan='en', stop_words=stop_words)

    f = open('../data/corpus/' + dataset + '.clean.txt', 'w', encoding='utf-8')
    f.write(clean_corpus_str)
    f.close()
elif dataset == "Weibo":
    # 读取文档内容
    doc_content_list = []
    f = open('../data/corpus/' + dataset + '.txt', 'r', encoding='utf-8')
    for line in f.readlines():
        doc_content_list.append(line.strip())
    f.close()
    # 分词
    doc_words_list = []
    for line in doc_content_list:
        line_list = [w for w in list(jieba.cut(line)) if w != ' ']
        line_str = ' '.join(line_list)
        doc_words_list.append(line_str)

    clean_corpus_str = remove_words(doc_words_list, lan='ch')

    f = open('../data/corpus/' + dataset + '.clean.txt', 'w', encoding='utf-8')
    f.write(clean_corpus_str)
    f.close()
else:
    sys.exit("dataset has no pre_processing")






# # 得到文档的max_len,min_len,aver_len
#
# min_len = 10000
# aver_len = 0
# max_len = 0
#
# f = open('data/corpus/' + dataset + '.clean.txt', 'r')
#
# lines = f.readlines()
# for line in lines:
#     line = line.strip()
#     temp = line.split()
#     aver_len = aver_len + len(temp)
#     if len(temp) < min_len:
#         min_len = len(temp)
#     if len(temp) > max_len:
#         max_len = len(temp)
# f.close()
# aver_len = 1.0 * aver_len / len(lines)
# print('min_len : ' + str(min_len))
# print('max_len : ' + str(max_len))
# print('average_len : ' + str(aver_len))
