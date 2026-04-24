
#Some codes are modified from https://github.com/Shengyu-Feng/ARIEL, https://github.com/xuebaliang/scziDesk

from __future__ import print_function

from typing import Tuple
import networkx as nx
import torch
from torch import Tensor
import h5py
import scanpy as sc
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder
from torch_scatter import scatter_add
from torch_geometric.utils import  add_self_loops
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


def GraphConstruction(data, features, num_clusters):
    cell_num = features.shape[0]
    average_num = cell_num // num_clusters
    neighbor_num = average_num // 10
    neighbor_num = min(neighbor_num, 15)
    neighbor_num = max(neighbor_num, 5)
 
    # Calculate Pearson distance matrix
    dis_matrix = squareform(pdist(features, metric='correlation'))


    # 这里我是想在原来函数的基础上增加边索引计算方式：高斯核相似度，后续还有余弦相似度

    # from sklearn.metrics.pairwise import euclidean_distances
    # # 计算欧氏距离矩阵
    # dist_matrix = euclidean_distances(features)
    # # 计算高斯核相似度矩阵  sigma =1.0
    # sim_matrix = np.exp(-dist_matrix**2 / (2 * 1.0**2))

    # normalize == "row"
    # # --- 正则化 ---
    # if normalize == "row":  # 行归一化
    #     sim_matrix = sim_matrix / (sim_matrix.sum(axis=1, keepdims=True) + 1e-10)
    # elif normalize == "symmetric":  # 对称归一化
    #     d = np.sum(sim_matrix, axis=1)
    #     D_inv_sqrt = np.diag(1.0 / np.sqrt(d + 1e-10))
    #     sim_matrix = D_inv_sqrt @ sim_matrix @ D_inv_sqrt
    # elif normalize == "max":
    #     sim_matrix = sim_matrix / (np.max(sim_matrix) + 1e-10)


    # print(sim_matrix)
    # print(sim_matrix.shape)
    # print('+'*100)

    # # print(dis_matrix)
    # # print(dis_matrix.shape)
    # # print('-'*100)
    # # assert 1==0

    # dis_matrix = (dis_matrix + sim_matrix)/2




    # Build kNN graph
    nbrs = NearestNeighbors(n_neighbors=neighbor_num, metric='precomputed').fit(dis_matrix)
    _, indices = nbrs.kneighbors(dis_matrix)  # Get only indices
 
    # Create adjacency matrix
    n_samples = features.shape[0]
    adj_matrix = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in indices[i]:
             adj_matrix[i, j] = 1  

      
    #Create a list of egdes
    edge_list = torch.empty((2,0), dtype=torch.int64)
    for i in range( adj_matrix.shape[0]):
     for j in range(i + 1,   adj_matrix.shape[1]):
        if  adj_matrix[i, j] == 1:
            col = torch.tensor([i, j], dtype=torch.int64)
            edge_list = torch.cat((edge_list, col.unsqueeze(1)), dim=1)
    data.edge_index=edge_list
          
    #generate a graph
    G = nx.from_numpy_array(adj_matrix)
    print("Building a " + str(G))
    print("===================================================")

    return G

#------------------------------------------------------------------
def normalize_adj_tensor(adj):
    """Symmetrically normalize adjacency tensor."""
    rowsum = torch.sum(adj,1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[d_inv_sqrt == float("Inf")] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    return torch.mm(torch.mm(adj,d_mat_inv_sqrt).transpose(0,1),d_mat_inv_sqrt)

def normalize_adj_tensor_sp(adj):
    """Symmetrically normalize sparse adjacency tensor."""
    device = adj.device
    adj = adj.to("cpu")
    rowsum = torch.spmm(adj, torch.ones((adj.size(0),1))).reshape(-1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[d_inv_sqrt == float("Inf")] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    adj = torch.mm(torch.smm(adj.transpose(0,1),d_mat_inv_sqrt.transpose(0,1)),d_mat_inv_sqrt)
    return adj.to(device)

def edge2adj(x, edge_index):
    """Convert edge index to adjacency matrix"""
    num_nodes = x.shape[0]
    tmp, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    edge_weight = torch.ones(tmp.size(1), dtype=None,
                                     device=edge_index.device)
    row, col = tmp[0], tmp[1]
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    #return torch.sparse.FloatTensor(tmp, edge_weight,torch.Size((num_nodes, num_nodes)))
    return torch.sparse_coo_tensor(tmp, edge_weight, torch.Size((num_nodes, num_nodes)))

def normalization(features_):
    features = features_.copy()
    for i in range(len(features)):
        features[i] = features[i] / sum(features[i]) * 100000
    features = np.log2(features + 1)
    return features

def dominateset(aff_matrix, NR_OF_KNN):
    thres = np.sort(aff_matrix)[:, -NR_OF_KNN]
    aff_matrix.T[aff_matrix.T < thres] = 0
    aff_matrix = (aff_matrix + aff_matrix.T) / 2
    return aff_matrix

import os
def load_h5_data_attack(data_path, num_nb_genes=2000):
    print("Reading data!")
    count_file = os.path.join("data/attack", "count_{}.csv".format(data_path))
    label_file = os.path.join("data/attack", "label_{}.csv".format(data_path))

    X = np.array(pd.read_csv(count_file))
    Y_raw = np.array(pd.read_csv(label_file))[:,1].tolist()
    

    # 提取类别名称
    class_names = np.unique(Y_raw)

    # 标签编码为数字（如果需要进行训练）
    encoder = LabelEncoder()
    numeric_labels = encoder.fit_transform(Y_raw)

    # ===== 预处理 =====
    X = preprocess(X, nb_genes=num_nb_genes)

    return X, numeric_labels, class_names


def preprocess(X, gene_names_all, nb_genes=2000):
    """
    Scanpy-based preprocessing. Returns filtered X and selected gene names.
    
    Args:
        X: (cells x genes) numpy array
        gene_names_all: list of all original gene names (length = X.shape[1])
        nb_genes: how many most variable genes to keep

    Returns:
        X: preprocessed (cells x nb_genes)
        selected_gene_names: list of gene names after filtering
    """
    print("There are " + str(X.shape[0]) + " cells, " + str(X.shape[1]) + " genes")
    print("Data Preprocessing!")

    adata = sc.AnnData(X)
    adata.var_names = gene_names_all  # 关键设置！！确保变量名带进去

    adata = normalize(
        adata,
        copy=True,
        highly_genes=nb_genes,
        size_factors=True,
        normalize_input=True,
        logtrans_input=True
    )

    X = adata.X.astype('float32')
    selected_gene_names = adata.var_names.to_list()
    
    print(f"Keeping {nb_genes} genes")
    return X, selected_gene_names




def normalize(adata, copy=True, highly_genes = None, filter_min_counts=True, 
              size_factors=True, normalize_input=True, logtrans_input=True):
    """
    Normalizes input data and retains only most variable genes 
    (indicated by highly_genes parameter)

    Args:
        adata ([type]): [description]
        copy (bool, optional): [description]. Defaults to True.
        highly_genes ([type], optional): [description]. Defaults to None.
        filter_min_counts (bool, optional): [description]. Defaults to True.
        size_factors (bool, optional): [description]. Defaults to True.
        normalize_input (bool, optional): [description]. Defaults to True.
        logtrans_input (bool, optional): [description]. Defaults to True.

    Raises:
        NotImplementedError: [description]

    Returns:
        [type]: [description]
    """
    if isinstance(adata, sc.AnnData):
        if copy:
            adata = adata.copy()
    elif isinstance(adata, str):
        adata = sc.read(adata)
    else:
        raise NotImplementedError
    norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
    assert 'n_count' not in adata.obs, norm_error
    

    if filter_min_counts:
        sc.pp.filter_genes(adata, min_cells=1)
        #sc.pp.filter_cells(adata, min_genes=1)
    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata
    if size_factors:
        sc.pp.normalize_total(adata, exclude_highly_expressed=True)

    if logtrans_input:
        sc.pp.log1p(adata)
    if highly_genes != None:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes = highly_genes, subset=True)

    return adata
#---------------Reading files .h file 1----------------------------------------#

"""
Load scRNA-seq data set from .h5 file and perfrom preprocessing
"""
def load_h5_data_11(data_path,num_nb_genes=2000):  # 
    print("Reading data!")
    inputFile = h5py.File(data_path, 'r')
   
    X = np.array(inputFile['X']).astype('float32')
    Y = np.array(inputFile['Y'])


    if Y.dtype != "int64":
        encoder_x = LabelEncoder()
        Y = encoder_x.fit_transform(Y)
        
        class_names = encoder_x.classes_
    inputFile.close()
    
    X=preprocess(X, nb_genes=num_nb_genes)

    return X, Y,class_names


def load_h5_data1(data_path, num_nb_genes=2000):
    print("Reading data!")
    inputFile = h5py.File(data_path, 'r')

    X = np.array(inputFile['X']).astype('float32')      # (301, 21721)
    Y_raw = np.array(inputFile['Y'])                    # (301,)
    inputFile.close()

    # print(X.shape)
    # print(Y_raw.shape)
    # print(X[0])         #   [  9.08   0.     0.   ...  48.22   0.   111.55]
    # print(Y_raw[0])     # b'2338'
    # print('-'*100)      #

    # ===== 统一编码标签 =====
    if Y_raw.dtype.kind in {"U", "S", "O"}:
        encoder = LabelEncoder()
        Y = encoder.fit_transform(Y_raw)
        n_classes = len(encoder.classes_)
    else:
        Y = Y_raw.astype(int)
        n_classes = len(np.unique(Y))

    # ===== 构造统一的 class_names =====
    class_names = np.array([f"Cell type {i+1}" for i in range(n_classes)])

    # ===== 预处理 =====
    # X = preprocess(X, nb_genes=num_nb_genes)
    # return X, Y, class_names
    # 获取基因类型~~~
    X, gene_names = preprocess(X, nb_genes=num_nb_genes)  # 接收 gene_names

    # print(gene_names)
    # print('='*100)

    return X, Y, class_names, gene_names

#---------------Reading files .h file 2----------------------------------------#
def load_h5_data2(data_path, is_NE=False, n_clusters=20, K=None):
    mat, obs, var, uns = read_data(data_path, sparsify=False, skip_exprs=False)

    # print(mat)
    # print('-1'*100)
    # print(mat)
    # print('-2'*100)
    # print(var)
    # print('-3'*100)expressed_genes
    # print(uns)
    # print('-4'*100)'seurat_genes

    if isinstance(mat, np.ndarray):
        X = np.array(mat)
    else:
        X = np.array(mat.toarray())
    cell_name = np.array(obs["cell_type1"]) # 他会显示每一个样本的细胞类型！而不是一次性说明全部的6类细胞类型！

    class_names, Y = np.unique(cell_name, return_inverse=True)
    
    X =X.astype('float32')
    # X=preprocess(X, nb_genes=2000)
    # return X, Y , class_names
    # 获取基因类型~~~
    gene_names_all = var.index.to_list()
    X, gene_names = preprocess(X, gene_names_all, nb_genes=2000)

    # X, gene_names = preprocess(X, nb_genes=2000)  # 接收 gene_names

    # print(gene_names)
    # print('='*100)
    return X, Y, class_names, gene_names

def empty_safe(fn, dtype):
    def _fn(x):
        if x.size:
            return fn(x)
        return x.astype(dtype)
    return _fn

decode = empty_safe(np.vectorize(lambda _x: _x.decode("utf-8")), str)


def read_data(filename, sparsify = False, skip_exprs = False):
    with h5py.File(filename, "r") as f:
        obs = pd.DataFrame(dict_from_group(f["obs"]), index = decode(f["obs_names"][...]))
        var = pd.DataFrame(dict_from_group(f["var"]), index = decode(f["var_names"][...]))
        uns = dict_from_group(f["uns"])
        if not skip_exprs:
            exprs_handle = f["exprs"]
            if isinstance(exprs_handle, h5py.Group):
                mat = sp.csr_matrix((exprs_handle["data"][...], exprs_handle["indices"][...],
                                               exprs_handle["indptr"][...]), shape = exprs_handle["shape"][...])
            else:
                mat = exprs_handle[...].astype(np.float32)
                if sparsify:
                    mat = sp.sparse.csr_matrix(mat)
        else:
            mat = sp.csr_matrix((obs.shape[0], var.shape[0]))
    return mat, obs, var, uns

class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


decode = empty_safe(np.vectorize(lambda _x: _x.decode("utf-8")), str)
def read_clean(data):
    assert isinstance(data, np.ndarray)
    if data.dtype.type is np.bytes_:
        data = decode(data)
    if data.size == 1:
        data = data.flat[0]
    return data


def dict_from_group(group):
    assert isinstance(group, h5py.Group)
    d = dotdict()
    for key in group:
        if isinstance(group[key], h5py.Group):
            value = dict_from_group(group[key])
        else:
            value = read_clean(group[key][...])
        d[key] = value
    return d

#-----------------------------------------------------------------------------
#Graph augmentation

def EdgeDropping(edge_index: Tensor, p: float = 0.5,
                 force_undirected: bool = False,
                 training: bool = True) -> Tuple[Tensor, Tensor]:
    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')

    if not training or p == 0.0:
        edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
        return edge_index, edge_mask

    row, col = edge_index

    edge_mask = torch.bernoulli(torch.ones(row.size(0), device=edge_index.device) * (1 - p)).type(torch.bool)

    if force_undirected:
        edge_mask[row > col] = False

    edge_index = edge_index[:, edge_mask]

    if force_undirected:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_mask = edge_mask.nonzero().repeat((2, 1)).squeeze()

    return edge_index, edge_mask


def GeneDropping(x, drop_prob):
    drop_mask = torch.bernoulli(torch.ones(x.size(1), device=x.device) * (1 - drop_prob)).type(torch.bool) 
    x = x.clone()
    x[:, drop_mask] = 0
    return x


#-----------------------------------------------------------------------------
class GCNConv(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GCNConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
    
#-----------------------------------------------------------------------------
def GraphAdversarialAttack(model, edge_index_1, edge_index_2, x_1, x_2, iters, node_ratio, alpha, beta):
    """ PGD attack on both features and edges"""

    for param in  model.parameters():
        param.requires_grad = False
    model.eval()
    device = x_1.device
    total_edges = edge_index_2.shape[1]         
    n_node = x_2.shape[0]
    eps = total_edges * node_ratio/2
    xi = 1e-3
    
    A_ = torch.sparse_coo_tensor(edge_index_2, torch.ones(total_edges,device=device), torch.Size((n_node, n_node))).to_dense() 
    C_ = torch.ones_like(A_) - 2 * A_ - torch.eye(A_.shape[0],device=device)
    S_ = torch.zeros_like(A_, requires_grad= True)
    mask = torch.ones_like(A_)
    mask = mask - torch.tril(mask)
    delta = torch.zeros_like(x_2, device=device, requires_grad=True)
    adj_1 = edge2adj(x_1, edge_index_1)
    model.to(device)

    for itr in range(iters):
        S = (S_ * mask)
        S = S + S.T
        A_prime = A_ + (S * C_)
        adj_hat = normalize_adj_tensor(A_prime + torch.eye(n_node,device=device))
        z1 = model(x_1, adj_1)
        z2 = model(x_2 + delta, adj_hat) 
        Attackloss = model.loss(z1, z2, batch_size=0) 
        Attackloss.backward()

        S_.data = (S_.data + alpha/np.sqrt(itr+1)*S_.grad.detach()) # annealing
        S_.data = bisection(S_.data, eps, xi) # clip S
        S_.grad.zero_()
        
        delta.data = (delta.data + beta*delta.grad.detach().sign()).clamp(-0.04,0.04)        
        delta.grad.zero_()

        randm = torch.rand(n_node, n_node,device=device)
        discretized_S = torch.where(S_.detach() > randm, torch.ones(n_node, n_node,device=device), torch.zeros(n_node, n_node, device=device))
        discretized_S = discretized_S + discretized_S.T
        A_hat = A_ + discretized_S * C_ + torch.eye(n_node,device=device)
        
    for param in model.parameters():
        param.requires_grad = True
    model.train()
    x_hat = x_2 + delta.data.to(device)
    assert torch.equal(A_hat, A_hat.transpose(0,1))
    return normalize_adj_tensor(A_hat), x_hat

def bisection(a,eps,xi,ub=1):
    pa = torch.clamp(a, 0, ub)
    if torch.sum(pa) <= eps:
        upper_S_update = pa
    else:
        mu_l = torch.min(a-1)
        mu_u = torch.max(a)
        mu_a = (mu_u + mu_l)/2
        while torch.abs(mu_u - mu_l)>xi:
            mu_a = (mu_u + mu_l)/2
            gu = torch.sum(torch.clamp(a-mu_a, 0, ub)) - eps
            gu_l = torch.sum(torch.clamp(a-mu_l, 0, ub)) - eps
            if gu == 0:
                break
            if torch.sign(gu) == torch.sign(gu_l):
                mu_l = mu_a
            else:
                mu_u = mu_a
        upper_S_update = torch.clamp(a-mu_a, 0, ub)
    return upper_S_update
