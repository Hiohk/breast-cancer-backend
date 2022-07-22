from __future__ import division
from __future__ import print_function
import time
import argparse
import numpy as np
import torch
import scipy.sparse as sp
import torch.nn.functional as F
import torch.optim as optim
from utils import load_data, accuracy,load_data1
from models import GCN
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=400,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,#5
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden1', type=int, default=8,#16
                    help='Number of hidden units.')
parser.add_argument('--hidden2', type=int, default=4,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.2,#0.3
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
simi, adj, features, labels, idx_train, idx_test = load_data1()

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid1=args.hidden1,
            nhid2=args.hidden2,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_test = idx_test.cuda()

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output, prob_train = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    precision_train, recall_train, f1_train = precision_recall(output[idx_train], labels[idx_train])
    AUC_train, AUC_PR_train = AUC(prob_train[idx_train], labels[idx_train])#
    loss_train.backward()
    optimizer.step()
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'precision_train: {:.4f}'.format(precision_train.item()),
          'recall_train: {:.4f}'.format(recall_train.item()),
          'f1_train: {:.4f}'.format(f1_train.item()),
          'auc_train: {:.4f}'.format(AUC_train.item()),
          'aucPR_train: {:.4f}'.format(AUC_PR_train.item()),
          'time: {:.4f}s'.format(time.time() - t))

def test():
    model.eval()
    output_SGCN, prob_test = model(features, adj)
    loss_test = F.nll_loss(output_SGCN[idx_test], labels[idx_test])
    acc_SGCN = accuracy(output_SGCN[idx_test], labels[idx_test])
    precision_SGCN, recall_SGCN, f1_SGCN = precision_recall(output_SGCN[idx_test], labels[idx_test])
    AUC_SGCN, AUC_PR_SGCN = AUC(prob_test[idx_test], labels[idx_test])
    print(prob_test)
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "SGCN_accuracy= {:.4f}".format(acc_SGCN.item()),
          "SGCN_precision= {:.4f}".format(precision_SGCN.item()),
          "SGCN_recall= {:.4f}".format(recall_SGCN.item()),
          "SGCN_f1= {:.4f}".format(f1_SGCN.item()),
          'SGCN_auc: {:.4f}'.format(AUC_SGCN.item()),
          'SGCN_aucPR: {:.4f}'.format(AUC_PR_SGCN.item()),
          )

def AUC(output, labels):
    preds = output.max(1)[1].type_as(labels)
    precision, recall, thresholds = precision_recall_curve(labels, preds)
    AUC_PR = auc(recall, precision)
    return roc_auc_score(labels, preds), AUC_PR#

def precision_recall(output, labels):
    preds = output.max(1)[1].type_as(labels)
    precision_train = precision_score(labels, preds, average='binary')
    recall_train = recall_score(labels, preds, average='binary')
    f1_train = f1_score(labels, preds, average='binary')
    return precision_train, recall_train, f1_train

def new_prediction(new_sample):
    # new_sample = [5, 1, 1, 1, 2, 1, 3, 1, 1]#阴性例子
    # new_sample = [4, 8, 6, 3, 4, 10, 7, 1, 1]#阳性例子
    new_sample = torch.Tensor(new_sample).resize(1, 9)
    new_simi_vector = torch.zeros(1, features.shape[0])
    for i in range(0, features.shape[0]):
        ss = features[i, :]
        new_simi = F.cosine_similarity(new_sample, ss.resize(1, 9))
        if new_simi > 0.99872:
            new_simi_vector[0, i] = 1
        else:
            new_simi_vector[0, i] = 0
    jiao = torch.zeros(1).resize(1, 1)
    first_raw = torch.cat([jiao, new_simi_vector], dim=1)
    new_adj1 = torch.cat([new_simi_vector.T, simi], dim=1)
    new_adj = torch.cat([first_raw, new_adj1], dim=0)
    new_adj = normalize(new_adj + torch.eye(new_adj.shape[0]))
    new_features = torch.cat([new_sample, features], dim=0)
    new_adj = torch.Tensor(new_adj)
    new_features = torch.Tensor(new_features)
    new_output, new_prob = model(new_features, new_adj)
    final_prob = new_prob[0, 1]
    # print('该患者患乳腺癌的概率为{:.2%},请进一步结合专家诊断得出最终结果！'.format(final_prob))
    return final_prob.item()

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
# test()
#
#
# new_prediction()
