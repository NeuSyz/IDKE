# text_gcn

基于GCN实现文本分类任务


论文：Liang Yao, Chengsheng Mao, Yuan Luo. "Graph Convolutional Networks for Text Classification." In 33rd AAAI Conference on Artificial Intelligence (AAAI-19), 7370-7377


## 环境

Python 3.6
Tensorflow = 1.12

## Run

1. Run `cd utils/`  `python remove_words.py <dataset>`

2. Run `cd utils/`  `python build_graph.py <dataset>`

3. Run `python train.py <dataset>`

4. Run `python test.py <dataset>`

注：使用自己数据集，请先运行prepare_data.py
## 项目结构

* ckpt 模型保存路径
* data
  * corpus 处理后数据
  * graph 图结构输入数据
  * R8 :raw  dataset
  * Weibo :raw dataset
* gcn_model 
  * inits.py 初始化的公用函数
  * layers.py gcn层定义
  * metrics.py 评测指标计算
  * models.py 模型定义
* utils 
  * build_graph.py 构建图结构输入数据
  * prepare_data.py 自定义数据集处理 
  * remove_words.py 数据清洗
  * utils.py 工具函数
* train.py
* test.py

## 样例数据
R8:英文

Weibo:中文



