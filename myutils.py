from scipy.io import loadmat
import scipy.sparse as sp
from collections import defaultdict
import numpy as np
import copy as cp
from torch_geometric.data import HeteroData
from torch_geometric.utils import from_scipy_sparse_matrix
from sklearn.model_selection import train_test_split
from numpy import random
import torch
from typing import List


def load_data(dataset_name: str) -> tuple:
    """
    Returns:
    list: 三个关系的adjacent matrix
    feature data: shape=(|N|,feature_dim)
    labels: shape=(|N|,)
    """
    if(dataset_name == 'Amazon'):
        amazon_data = loadmat('./input/Amazon.mat')
        # users reviewing at least one same product
        network_upu_data = amazon_data['net_upu']
        # users having at least one same star rating within one week
        network_usu_data = amazon_data['net_usu']
        # users with top 5% mutual review text similarities
        network_uvu_data = amazon_data['net_uvu']
        network_homo_data = amazon_data['homo']

        features_data = amazon_data['features'].todense().A
        labels = amazon_data['label'].flatten()

        return [network_homo_data, network_upu_data, network_usu_data, network_uvu_data], features_data, labels


def constructPyGHeteroData(dataset_name: str = 'Amazon', train_ratio: float = 0.6, test_ratio: float = 0.67) -> HeteroData:
    """
    将mat形式的数据集转化为HeteroData数据

    Returns:
    data: PyG的HeteroData数据
    """
    # TODO 考虑每次把mat转化后的文件储存一下

    network_data_list, features_data, labels = load_data(dataset_name)
    network_homo_data, network_r1_data, network_r2_data, network_r3_data = network_data_list
    # r1 -> upu
    # r2 -> usu
    # r3 -> uvu

    # 前3304的数据unlabelled
    index = list(range(3305, len(labels)))
    train_mask, rest_mask, y_train, y_rest = train_test_split(index, labels[3305:], stratify=labels[3305:],
                                                              train_size=train_ratio, random_state=2, shuffle=True)
    valid_mask, test_mask, y_valid, y_test = train_test_split(rest_mask, y_rest, stratify=y_rest,
                                                              test_size=test_ratio, random_state=2, shuffle=True)
    # print(network_homo_data.shape)
    data = HeteroData()
    # nodes info
    data['review'].x = features_data
    data['review'].y = labels
    data['review'].train_mask = train_mask
    data['review'].valid_mask = valid_mask
    data['review'].test_mask = test_mask

    # adj mat
    # shape -> (|N|,|N|)
    # data['review', 'r1', 'review'].adj = torch.Tensor(
    #     network_r1_data.todense().A)
    # data['review', 'r2', 'review'].adj = torch.Tensor(
    #     network_r2_data.todense().A)
    # data['review', 'r3', 'review'].adj = torch.Tensor(
    #     network_r3_data.todense().A)
    # data['review', 'r1', 'review'].adj = network_r1_data.todense().A
    # data['review', 'r2', 'review'].adj = network_r2_data.todense().A
    # data['review', 'r3', 'review'].adj = network_r3_data.todense().A


    # adj list
    data['review', 'r1', 'review'].adj_list = [sparse_to_adjlist(network_r1_data)]
    data['review', 'r2', 'review'].adj_list = [sparse_to_adjlist(network_r2_data)]
    data['review', 'r3', 'review'].adj_list = [sparse_to_adjlist(network_r3_data)]

    # homo adj list
    # 总感觉有点别扭
    data['homo_adj_list'] = [sparse_to_adjlist(network_homo_data)]

    # relation info -> shape: (2, num_edges)
    # from_scipy_smatrix 返回的是一个元组，第二项是edge_weight——有什么用吗？
    # data['review', 'r1', 'review'].edge_index = from_scipy_sparse_matrix(network_r1_data)[0]
    # data['review', 'r2', 'review'].edge_index = from_scipy_sparse_matrix(network_r2_data)[0]
    # data['review', 'r3', 'review'].edge_index = from_scipy_sparse_matrix(network_r3_data)[0]
    # no edge attr

    # split pos and neg
    data['review'].train_pos_mask,data['review'].train_neg_mask = split_pos_neg_nodes(train_mask,labels[train_mask])

    return data


def sparse_to_adjlist(sp_matrix) -> List[dict]:
    """
    Transfer sparse matrix to adjacency list

    :param sp_matrix: scipy的稀疏矩阵
    :return adj_list: 稀疏矩阵转为[{},{},...,{}]的格式
    """
    # add self loop
    homo_adj = sp_matrix + sp.eye(sp_matrix.shape[0])
    # create adj_list
    # 注意defaultdict的作用->查找不存在key时返回工厂函数
    adj_lists = defaultdict(set)
    edges = homo_adj.nonzero()
    for index, node in enumerate(edges[0]):
        adj_lists[node].add(edges[1][index])
        adj_lists[edges[1][index]].add(node)
    
    return (adj_lists)
    


def LabelBalancedSampler(batch_mask, y_train, adj_list, size):
    """ 
    用于对一个batch的train_mask 根据label信息进行过滤 -> pick操作

    其实就是个under sample...

    Params:
    batch_mask: 
    y_train: 
    adj_list: [{},{},...,{}]
    """
    degree_train = [len(adj_list[node]) for node in batch_mask]
    # 这就是label frequency呀！
    lf_train = (y_train.sum()-len(y_train))*y_train + len(y_train)
    smp_prob = np.array(degree_train) / lf_train
    return random.choices(batch_mask, weights=smp_prob, k=size)


def split_pos_neg_nodes(nodes,labels) -> tuple:
    """ 
    :return 元组 分别装class为0和1的nodes
    """
    pos_nodes = []
    neg_nodes = cp.deepcopy(nodes)
    aux_nodes = cp.deepcopy(nodes)
    for idx, label in enumerate(labels):
        if label == 1:
            pos_nodes.append(aux_nodes[idx])
            neg_nodes.remove(aux_nodes[idx])
    

    return pos_nodes, neg_nodes
