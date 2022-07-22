import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
import scipy.spatial
import math
from collections import Counter
from itertools import combinations
import itertools


def load_data(path="../data/breast/", dataset="breast"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))#从文本文件加载数据2708*（1433+2）
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)#属性矩阵，三列表示
    labels = idx_features_labels[:, -1]

    # calculate similarity
    # features = normalize(features)  # 归一化特征矩阵,D^(-1)X
    featuresm = features.todense()
    # featuresm = features.todense().A
    simi = np.zeros((featuresm.shape[0], featuresm.shape[0]))
    for i in range(0, featuresm.shape[0]):
        for j in range(i+1, featuresm.shape[0]):
            # X = torch.Tensor(featuresm[i])
            # X = X.tolist()
            # Y = torch.Tensor(featuresm[j])
            # Y = Y.tolist()
            # XY = list(zip(X, Y))
            # simi[i][j] = Entropy(X) + Entropy(Y) - Entropy(XY)
            # simi[i][j] = scipy.spatial.distance.correlation(np.array(featuresm[i]), np.array(featuresm[j]))
            ss = torch.Tensor(featuresm[i, :])
            dd = torch.Tensor(featuresm[j, :])
            simi[i][j] = F.cosine_similarity(torch.Tensor(featuresm[i, :]), torch.Tensor(featuresm[j, :]))
    simi = simi + simi.T

    # build graph
    # para = np.mean(simi)
    para = 0.94  # mean_simi
    simi[simi > para] = 1
    simi[simi <= para] = 0
    adj = simi
    adj = normalize(adj + sp.eye(adj.shape[0]))#归一化邻接矩阵D^(-1)(A+IN)

    idx_train = range(616)
    idx_test = range(616, 683)

    labels = labels.astype(float)
    labels = torch.LongTensor(labels)
    adj = torch.Tensor(adj)
    simi = torch.Tensor(simi)
    features = torch.FloatTensor(np.array(features.todense()))

    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_test)

    return simi, adj, features, labels, idx_train, idx_test

def load_data1(path="../data/breast/", dataset="breast_BMI"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))#从文本文件加载数据2708*（1433+2）
    features = sp.csr_matrix(idx_features_labels[:, :-1], dtype=np.float32)#属性矩阵，三列表示
    labels = idx_features_labels[:, -1]

    # calculate similarity
    features = normalize(features)  # 归一化特征矩阵,D^(-1)X
    featuresm = features.todense()
    # featuresm = features.todense().A
    simi = np.zeros((featuresm.shape[0], featuresm.shape[0]))
    for i in range(0, featuresm.shape[0]):
        for j in range(i+1, featuresm.shape[0]):
            # X = torch.Tensor(featuresm[i])
            # X = X.tolist()
            # Y = torch.Tensor(featuresm[j])
            # Y = Y.tolist()
            # XY = list(zip(X, Y))
            # simi[i][j] = Entropy(X) + Entropy(Y) - Entropy(XY)
            # simi[i][j] = scipy.spatial.distance.correlation(np.array(featuresm[i]), np.array(featuresm[j]))
            ss = torch.Tensor(featuresm[i, :])
            dd = torch.Tensor(featuresm[j, :])
            # if labels[i] =='1' and labels[j]=='1':
            #     simi[i][j]=1
            # else:
            #     simi[i][j]=0
            simi[i][j] = F.cosine_similarity(torch.Tensor(featuresm[i, :]), torch.Tensor(featuresm[j, :]))
    simi = simi + simi.T

    # build graph
    # para = np.mean(simi)
    para = 0.99872  # mean_simi
    simi[simi > para] = 1
    simi[simi <= para] = 0
    adj = simi
    adj = normalize(adj + sp.eye(adj.shape[0]))#归一化邻接矩阵D^(-1)(A+IN)
    print(adj)
    idx_train = range(92)
    idx_test = range(92, 116)

    labels = labels.astype(float)
    labels = torch.LongTensor(labels)
    adj = torch.Tensor(adj)
    simi = torch.Tensor(simi)
    features = torch.FloatTensor(np.array(features.todense()))

    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_test)

    return simi, adj, features, labels, idx_train, idx_test

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    #mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx): # 把一个系数矩阵转为torch稀疏张量
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def Entropy(DataList):
    counts = len(DataList)  # 总数量
    counter = Counter(DataList)  # 每个变量出现的次数
    prob = {i[0]: i[1] / counts for i in counter.items()}  # 计算每个变量的 p*log(p)
    H = - sum([i[1] * math.log2(i[1]) for i in prob.items()])  # 计算熵
    return H