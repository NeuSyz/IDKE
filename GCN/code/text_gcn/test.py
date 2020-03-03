import tensorflow as tf
import os
from utils.utils import *
from sklearn import metrics
from gcn_model.models import GCN, MLP


if len(sys.argv) != 2:
    sys.exit("Use: python test.py <dataset>")

datasets = ['R8', 'Weibo']  # 当前数据集
dataset = sys.argv[1]

if dataset not in datasets:
    sys.exit("wrong dataset name")

# Settings
os.environ["CUDA_VISIBLE_DEVICES"] = ""

flags = tf.app.flags
FLAGS = flags.FLAGS
# 'R8'
flags.DEFINE_string('dataset', dataset, 'Dataset string.')
# 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_string('model', 'gcn', 'Model string.')
flags.DEFINE_float('learning_rate', 0.02, 'Initial learning rate.')
flags.DEFINE_integer('hidden1', 200, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0, 'Weight for L2 loss on embedding matrix.')  # 5e-4
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(FLAGS.dataset)

features = sp.identity(features.shape[0])  # featureless
features = preprocess_features(features)
if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)
}

model = model_func(placeholders, input_dim=features[2][1], logging=True)
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint("ckpt/{}/".format(dataset)))
    feed_dict_test = construct_feed_dict(features, support, y_test, test_mask, placeholders)

    pred, labels, embedding = sess.run([model.pred, model.labels, model.layers[0].embedding], feed_dict=feed_dict_test)


test_pred = []
test_labels = []
print(len(test_mask))
for i in range(len(test_mask)):
    if test_mask[i]:
        test_pred.append(pred[i])
        test_labels.append(labels[i])

print("Test Precision, Recall and F1-Score...")
print(metrics.classification_report(test_labels, test_pred, digits=4))
print("Macro average Test Precision, Recall and F1-Score...")
print(metrics.precision_recall_fscore_support(test_labels, test_pred, average='macro'))
print("Micro average Test Precision, Recall and F1-Score...")
print(metrics.precision_recall_fscore_support(test_labels, test_pred, average='micro'))

# -----------------------------生成词向量，文档向量---------------------------------------------
# # doc and word embeddings
# # print('embeddings:')
#
# word_embeddings = embedding[train_size: adj.shape[0] - test_size]
# train_doc_embeddings = embedding[:train_size]  # include val docs
# test_doc_embeddings = embedding[adj.shape[0] - test_size:]
#
# print("embedding shape:", len(word_embeddings), len(train_doc_embeddings),
#       len(test_doc_embeddings))
# # print(word_embeddings)
#
# f = open('data/corpus/' + dataset + '_vocab.txt', 'r', encoding='utf-8')
# words = f.readlines()
# f.close()
#
# vocab_size = len(words)
# word_vectors = []
# for i in range(vocab_size):
#     word = words[i].strip()
#     word_vector = word_embeddings[i]
#     word_vector_str = ' '.join([str(x) for x in word_vector])
#     word_vectors.append(word + ' ' + word_vector_str)
#
# word_embeddings_str = '\n'.join(word_vectors)
# f = open('data/' + dataset + '_word_vectors.txt', 'w', encoding='utf-8')
# f.write(word_embeddings_str)
# f.close()
#
# doc_vectors = []
# doc_id = 0
# for i in range(train_size):
#     doc_vector = train_doc_embeddings[i]
#     doc_vector_str = ' '.join([str(x) for x in doc_vector])
#     doc_vectors.append('doc_' + str(doc_id) + ' ' + doc_vector_str)
#     doc_id += 1
#
# for i in range(test_size):
#     doc_vector = test_doc_embeddings[i]
#     doc_vector_str = ' '.join([str(x) for x in doc_vector])
#     doc_vectors.append('doc_' + str(doc_id) + ' ' + doc_vector_str)
#     doc_id += 1
#
# doc_embeddings_str = '\n'.join(doc_vectors)
# f = open('data/' + dataset + '_doc_vectors.txt', 'w', encoding='utf-8')
# f.write(doc_embeddings_str)
# f.close()
