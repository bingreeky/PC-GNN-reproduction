from scipy.io import loadmat
import pickle
import scipy.sparse as sp
import copy as cp
from collections import defaultdict
"""
	Read data and save the adjacency matrices to adjacency lists
"""


def sparse_to_adjlist(sp_matrix, filename):
    """
    Transfer sparse matrix to adjacency list
    :param sp_matrix: the sparse matrix
    :param filename: the filename of adjlist
    """
    # add self loop
    homo_adj = sp_matrix + sp.eye(sp_matrix.shape[0])
    # create adj_list
    adj_lists = defaultdict(set)
    edges = homo_adj.nonzero()
    for index, node in enumerate(edges[0]):
        adj_lists[node].add(edges[1][index])
        adj_lists[edges[1][index]].add(node)
    with open(filename, 'wb') as file:
        pickle.dump(adj_lists, file)
    file.close()


if __name__ == "__main__":

    prefix = 'data/'

    yelp = loadmat('data/YelpChi.mat')
    net_rur = yelp['net_rur']
    net_rtr = yelp['net_rtr']
    net_rsr = yelp['net_rsr']
    yelp_homo = yelp['homo']

    sparse_to_adjlist(net_rur, prefix + 'yelp_rur_adjlists.pickle')
    sparse_to_adjlist(net_rtr, prefix + 'yelp_rtr_adjlists.pickle')
    sparse_to_adjlist(net_rsr, prefix + 'yelp_rsr_adjlists.pickle')
    sparse_to_adjlist(yelp_homo, prefix + 'yelp_homo_adjlists.pickle')

    amz = loadmat('data/Amazon.mat')
    net_upu = amz['net_upu']
    net_usu = amz['net_usu']
    net_uvu = amz['net_uvu']
    amz_homo = amz['homo']

    sparse_to_adjlist(net_upu, prefix + 'amz_upu_adjlists.pickle')
    sparse_to_adjlist(net_usu, prefix + 'amz_usu_adjlists.pickle')
    sparse_to_adjlist(net_uvu, prefix + 'amz_uvu_adjlists.pickle')
    sparse_to_adjlist(amz_homo, prefix + 'amz_homo_adjlists.pickle')
