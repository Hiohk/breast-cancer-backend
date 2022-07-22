import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, nclass, dropout):
        super(GCN, self).__init__()  #super()._init_()利用父类里的对象构造函数

        self.gc1 = GraphConvolution(nfeat, nhid1)#第一层的输入输出，输出维度：1433，16
        self.gc2 = GraphConvolution(nhid1, nhid2)#第二层的输入输出，输出维度：16，7
        self.gc3 = GraphConvolution(nhid2, nclass)  # 第二层的输入输出，输出维度：16，7
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(x, adj)
        return F.log_softmax(x, dim=1), F.softmax(x, dim=1)  #
