
## GCN 的实现

在正式开始前，导入所需的包或模块


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
```

### 1.数据获取

实验采用Zachary 空手道俱乐部图网络，Zachary 空手道俱乐部是一个被广泛使用的社交网络，其中的节点代表空手道俱乐部的成员，边代表成员之间的相互关系。当年，Zachary 在研究空手道俱乐部的时候，管理员和教员发生了冲突，导致俱乐部一分为二。可使用networks包获取网络，并将其转化为邻接矩阵表示．


```python
import networkx as nx

G = nx.karate_club_graph()
A = nx.adjacency_matrix(G).todense()
A = torch.FloatTensor(A)
```

接下来为Ａ矩阵增加自环得到Ａ_hat，可将Ａ与单位矩阵I相加实现，并计算Ａ_hat的度矩阵D_hat.


```python
I = torch.eye(A.size(0))
A_hat = A + I
D_hat = torch.sum(A_hat, axis=0)
D_hat = torch.diag(D_hat)
D_hat
```




    tensor([[17.,  0.,  0.,  ...,  0.,  0.,  0.],
            [ 0., 10.,  0.,  ...,  0.,  0.,  0.],
            [ 0.,  0., 11.,  ...,  0.,  0.,  0.],
            ...,
            [ 0.,  0.,  0.,  ...,  7.,  0.,  0.],
            [ 0.,  0.,  0.,  ...,  0., 13.,  0.],
            [ 0.,  0.,  0.,  ...,  0.,  0., 18.]])



### 2.标签及特征设置

查看社交网络的节点特征：


```python
G.nodes.data()
```




    NodeDataView({0: {'club': 'Mr. Hi'}, 1: {'club': 'Mr. Hi'}, 2: {'club': 'Mr. Hi'}, 3: {'club': 'Mr. Hi'}, 4: {'club': 'Mr. Hi'}, 5: {'club': 'Mr. Hi'}, 6: {'club': 'Mr. Hi'}, 7: {'club': 'Mr. Hi'}, 8: {'club': 'Mr. Hi'}, 9: {'club': 'Officer'}, 10: {'club': 'Mr. Hi'}, 11: {'club': 'Mr. Hi'}, 12: {'club': 'Mr. Hi'}, 13: {'club': 'Mr. Hi'}, 14: {'club': 'Officer'}, 15: {'club': 'Officer'}, 16: {'club': 'Mr. Hi'}, 17: {'club': 'Mr. Hi'}, 18: {'club': 'Officer'}, 19: {'club': 'Mr. Hi'}, 20: {'club': 'Officer'}, 21: {'club': 'Mr. Hi'}, 22: {'club': 'Officer'}, 23: {'club': 'Officer'}, 24: {'club': 'Officer'}, 25: {'club': 'Officer'}, 26: {'club': 'Officer'}, 27: {'club': 'Officer'}, 28: {'club': 'Officer'}, 29: {'club': 'Officer'}, 30: {'club': 'Officer'}, 31: {'club': 'Officer'}, 32: {'club': 'Officer'}, 33: {'club': 'Officer'}})



根据以上节点分类数据，标记节点的标签('Mr.Hi' 为0, 'Officer'为１)，Real只用与网络的评价，不参与训练．


```python
Real = torch.zeros(34 , dtype=torch.long)
for i in [1,2,3,4,5,6,7,8,11,12,13,14,17,18,20,22]:
    Real[i-1] = 0
for i in [9,10,15,16,19,21,23,24,25,26,27,28,29,30,31,32,33,34]:
    Real[i-1] = 1
```

构造类别标签Y，直接参与网络的训练，一个类别仅给出一个样本的标记．构造Ｙ_mask记录已标记的样本，用于损失函数计算．


```python
N = len(A)
Y = torch.zeros(N,1).long()
Y[0][0]=0
Y[N-1][0]=1

Y_mask = torch.zeros(N,1,dtype=torch.bool)
Y_mask[0][0]=1
Y_mask[N-1][0]=1
```

### ３．模型构建

模型以节点的特征矩阵为输入，利用传播公式f = D_hat**-1 * A_hat * X * W实现卷积操作．由于Zachary 空手道俱乐部图网络没有节点的特征，所以我们只使用单位矩阵作为特征表征，即每个节点被表示为一个 one-hot 编码的类别变量，即将单位矩阵Ｉ作为网络的输入．


```python
class Gcn_layer(nn.Module):
    
    def __init__(self, dim_in, dim_out):
        super(Gcn_layer,self).__init__()
        self.W = nn.Parameter(torch.tensor(
            np.random.normal(loc=0, scale=0.1, size=(dim_in, dim_out)),dtype=torch.float))
    
    def forward(self, A_hat, D_hat, X):
        return F.relu((D_hat.inverse()).matmul(A_hat).matmul(X).matmul(self.W))
                              

class GCN(nn.Module):
    
    def __init__(self, dim_in, dim_out):
        super(GCN, self).__init__()
        self.layer1 = Gcn_layer(dim_in, dim_in)
        self.layer2 = Gcn_layer(dim_in, dim_in//2)
        self.layer3 = Gcn_layer(dim_in//2, dim_out)
    
    def forward(self, A_hat, D_hat, X):
        H1 = F.relu(self.layer1(A_hat, D_hat, X))
        H2 = F.relu(self.layer2(A_hat, D_hat, H1))
        output = F.softmax(self.layer3(A_hat, D_hat, H2),dim=1)
        return output
    
```

### 4.模型训练


```python
gcn = GCN(N, 2)
gd = torch.optim.Adam(gcn.parameters())

for i in range(300):
    y_pred = gcn(A_hat, D_hat, I)
    loss = (-y_pred.log().gather(1,Y.view(-1,1)))
    loss = loss.masked_select(Y_mask).mean()
    gd.zero_grad()
    loss.backward()
    gd.step()
    if i%30==0:
        _,mi = y_pred.max(1)
        print((mi == Real).float().mean().item())
```

    0.529411792755127
    0.9117646813392639
    0.970588207244873
    0.970588207244873
    0.970588207244873
    1.0
    1.0
    1.0
    1.0
    1.0


由此可见，即便在网络已知标签很少的情况下，GCN通过结点间的连接关系，也能取得很好的分类效果．
