# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("Agg")  # headless server 必备
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import logging
import time
import pandas as pd
import argparse
from multiprocessing import Pool
import random
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import metrics as M

from utils import load_h5_data1 , load_h5_data2
from readFile import selRepsample
from client import create_clients
from torch_geometric.nn import GCNConv

from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.utils import negative_sampling

import torch
import numpy as np
from math import sqrt, erf, exp
import math
import os
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

import matplotlib.pyplot as plt
import umap

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling

import metrics as M
from utils import load_h5_data1, load_h5_data2, load_h5_data_attack
from readFile import selRepsample

from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix

from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
def align_cluster_labels(y_true, y_pred):
    """
    使用 Hungarian algorithm 对齐聚类标签，使 pred 的编号尽量匹配 y_true
    """
    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    aligned_pred = np.array([mapping[p] for p in y_pred])
    return aligned_pred


root = os.path.dirname(__file__)



# =========================================================
# UMAP 可视化
# =========================================================
import numpy as np
import umap
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def get_custom_cmap():
    custom_colors = [
        "#9B002C",  # 暗红（主类）
        "#C62E2E",  # 红
        "#E1672D",  # 橙
        "#F2B455",  # 暖黄
        "#F6E79A",  # 淡黄
        "#E5F5C5",  # 淡黄绿（过渡）
        "#C7EFCF",  # 浅绿（中性）
        "#A0E5D8",  # 青绿（过渡）
        "#6CBFD4",  # 浅冷色（极少类）
        "#3A79C5",  # 冷蓝（极少类）
    ]
    return LinearSegmentedColormap.from_list("custom_grad", custom_colors, N=256)

def build_fixed_color_map(num_classes):
    """
    为所有真实类别构建固定颜色映射
    保证不同实验 / 不同图 / 不同聚类结果颜色一致
    """
    cmap = get_custom_cmap()
    color_map = {}

    for c in range(num_classes):
        t = c / max(num_classes - 1, 1)
        t = t ** 0.5   # 保留你原来的非线性渐进
        color_map[c] = cmap(t)

    return color_map

def plot_tsne(
    emb_input,
    labels,
    title="t-SNE",
    save_path=None,
    point_size=3,
    random_state=0,
    perplexity=30,
    class_names=None,
    legend_fontsize=14,
    color_map=None      # ⭐ 新增：固定颜色映射
):
    """
    t-SNE 可视化
    - 颜色由真实类别编号决定
    - 即使只出现部分类别，颜色也保持全局一致
    """

    emb_input = np.asarray(emb_input)
    labels = np.asarray(labels)

    if perplexity >= emb_input.shape[0]:
        perplexity = max(5, emb_input.shape[0] // 3)

    reducer = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        init="pca",
        learning_rate="auto"
    )

    emb_2d = reducer.fit_transform(emb_input)

    unique_labels = np.unique(labels)

    fig, ax = plt.subplots(figsize=(7, 6))

    for lab in unique_labels:
        mask = (labels == lab)

        # === 核心：颜色只由类别编号决定 ===
        if color_map is not None and lab in color_map:
            color = color_map[lab]
        else:
            color = "gray"   # 理论上不会触发

        name = (
            class_names[lab]
            if (class_names is not None and lab < len(class_names))
            else str(lab)
        )

        ax.scatter(
            emb_2d[mask, 0],
            emb_2d[mask, 1],
            s=point_size,
            color=color,
            label=name
        )

    ax.set_title(title, fontsize=20)
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)

    legend = fig.legend(
        bbox_to_anchor=(1.02, 0.5),
        loc="center left",
        fontsize=legend_fontsize,
        frameon=True
    )
    legend.get_frame().set_linewidth(1.0)

    plt.tight_layout(rect=[0, 0, 0.85, 1])

    if save_path is not None:
        save_path = os.path.splitext(save_path)[0] + ".tiff"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=400, format="tiff", bbox_inches="tight")

    plt.show()



def build_cluster_count_table(y_true, y_pred_aligned, class_names):
    """
    统计每个类别的细胞数量，生成论文表格格式

    y_true: 真实标签
    y_pred_aligned: 对齐后的预测标签
    class_names: 类别名称列表
    """

    unique_classes = np.unique(y_true)
    
    rows = []
    for c in unique_classes:
        class_name = class_names[c] if c < len(class_names) else str(c)
        
        gt_count = np.sum(y_true == c)
        pred_count = np.sum(y_pred_aligned == c)

        rows.append([class_name, gt_count, pred_count])

    df = pd.DataFrame(rows, columns=[
        "Sub-cluster", 
        "Ground Truth Count", 
        "Predicted Count"
    ])

    return df






# ···················································································差分隐私代码························································
# ···················································································差分隐私代码························································
def flatten(grad):
    return grad.data.view(-1)

# 差分隐私核心函数
def calibrateAnalyticGaussianMechanism(epsilon, delta, GS=1, tol=1.e-12):

    """ Calibrate a Gaussian perturbation for differential privacy using the analytic Gaussian mechanism of [Balle and Wang, ICML'18]
        Arguments:
        epsilon : target epsilon (epsilon > 0)
        delta : target delta (0 < delta < 1)
        GS : upper bound on L2 global sensitivity (GS >= 0)
        tol : error tolerance for binary search (tol > 0)
        Output:
        sigma : standard deviation of Gaussian noise needed to achieve (epsilon,delta)-DP under global sensitivity GS
    """

    def Phi(t):
        return 0.5 * (1.0 + erf(float(t) / sqrt(2.0)))

    def caseA(epsilon, s):
        return Phi(sqrt(epsilon * s)) - exp(epsilon) * Phi(-sqrt(epsilon * (s + 2.0)))

    def caseB(epsilon, s):
        return Phi(-sqrt(epsilon * s)) - exp(epsilon) * Phi(-sqrt(epsilon * (s + 2.0)))

    def doubling_trick(predicate_stop, s_inf, s_sup):
        while (not predicate_stop(s_sup)):
            s_inf = s_sup
            s_sup = 2.0 * s_inf
        return s_inf, s_sup

    def binary_search(predicate_stop, predicate_left, s_inf, s_sup):
        s_mid = s_inf + (s_sup - s_inf) / 2.0
        while (not predicate_stop(s_mid)):
            if (predicate_left(s_mid)):
                s_sup = s_mid
            else:
                s_inf = s_mid
            s_mid = s_inf + (s_sup - s_inf) / 2.0
        return s_mid

    delta_thr = caseA(epsilon, 0.0)

    if (delta == delta_thr):
        alpha = 1.0

    else:
        if (delta > delta_thr):
            predicate_stop_DT = lambda s: caseA(epsilon, s) >= delta
            function_s_to_delta = lambda s: caseA(epsilon, s)
            predicate_left_BS = lambda s: function_s_to_delta(s) > delta
            function_s_to_alpha = lambda s: sqrt(1.0 + s / 2.0) - sqrt(s / 2.0)

        else:
            predicate_stop_DT = lambda s: caseB(epsilon, s) <= delta
            function_s_to_delta = lambda s: caseB(epsilon, s)
            predicate_left_BS = lambda s: function_s_to_delta(s) < delta
            function_s_to_alpha = lambda s: sqrt(1.0 + s / 2.0) + sqrt(s / 2.0)

        predicate_stop_BS = lambda s: abs(function_s_to_delta(s) - delta) <= tol

        s_inf, s_sup = doubling_trick(predicate_stop_DT, 0.0, 1.0)
        s_final = binary_search(predicate_stop_BS, predicate_left_BS, s_inf, s_sup)
        alpha = function_s_to_alpha(s_final)

    sigma = alpha * GS / sqrt(2.0 * epsilon)
    return sigma


def dp(grad, device,epsilon=8, delta=1e-3, l2_norm_clip=5):
    sigma = calibrateAnalyticGaussianMechanism(epsilon=epsilon, delta=delta, GS=l2_norm_clip)
    #sigma = 0.0001 #TODO
    result = torch.tensor(np.random.normal(0, sigma, grad.shape), device=device) + grad
    return result


# ···················································································基础模型代码代码························································
# ···················································································基础模型代码代码························································
# 统一种子（在任何随机操作和模型初始化之前调用一次）
def set_global_seed(seed: int = 0, deterministic: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_edge_index_by_knn(features, num_clusters=10, method="pearson", combine_with=None, alpha=0.5):
    """
    features: (n_samples, n_features)
    method: 主方法 ("pearson" | "cosine" | "gaussian")
    combine_with: None 或 "pearson"/"gaussian"/"cosine"，表示要融合的第二种方法
    alpha: 融合权重, 0~1之间
           dis_matrix = alpha * dis_matrix_main + (1-alpha) * dis_matrix_other
    """

    # --- 计算主方法的距离矩阵 ---
    if method == "pearson": # 45
        dis_matrix = pairwise_distances(features, metric="correlation")
    elif method == "cosine":    # 47
        dis_matrix = pairwise_distances(features, metric="cosine")
    elif method == "gaussian":  # 48
        dist_matrix = pairwise_distances(features, metric="euclidean")
        sigma = 1.0
        sim_matrix = np.exp(-dist_matrix ** 2 / (2 * sigma ** 2))
        dis_matrix = 1 - sim_matrix
    elif method == "hamming":   # 48
        dis_matrix = pairwise_distances(features, metric="hamming")
    else:
        raise ValueError("Unknown similarity method!")


    n_samples = features.shape[0]
    average_num = max(1, n_samples // max(num_clusters, 1))
    neighbor_num = min(max(average_num // 10, 5), 15)
 
    # --- KNN 建图 ---
    nbrs = NearestNeighbors(n_neighbors=neighbor_num, metric="precomputed").fit(dis_matrix)
    _, indices = nbrs.kneighbors(dis_matrix)
    
    edges_undirected = set()
    for i in range(n_samples):
        for j in indices[i]:
            if i != j:
                edges_undirected.add((i, j))
                edges_undirected.add((j, i))
    edge_index = torch.tensor(list(edges_undirected), dtype=torch.long).t().contiguous()


    adj_label = torch.zeros((n_samples, n_samples), dtype=torch.float32)
    for i, j in edges_undirected:
        adj_label[i, j] = 1.0

    return edge_index, adj_label



# ===========================
# GAE 模型（替换 VGAE）
# ===========================
class GAE_GCN_3L(nn.Module):
    """
    GAE with 3 GCN layers: gc1 -> gc2 -> gc_z（无 μ / logvar）
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout=0.2):
        super(GAE_GCN_3L, self).__init__()
        self.gc1 = GCNConv(input_dim, hidden_dim)
        self.gc2 = GCNConv(hidden_dim, hidden_dim)
        self.gc_z = GCNConv(hidden_dim, latent_dim)   # <--- 用一个分支直接输出 z
        self.dropout = dropout

    def encode(self, x, edge_index):
        h = F.relu(self.gc1(x, edge_index))
        if self.dropout > 0:
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.gc2(h, edge_index))
        if self.dropout > 0:
            h = F.dropout(h, p=self.dropout, training=self.training)
        z = self.gc_z(h, edge_index)
        return z

    def decode(self, z):
        return torch.sigmoid(torch.matmul(z, z.t()))

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        adj_recon = self.decode(z)
        return z, adj_recon


# GAE 的重构 loss（没有 KL）
def gae_loss(adj_recon, adj_label):
    return F.binary_cross_entropy(adj_recon, adj_label)


### ###  对比学习 loss
def edge_contrastive_loss(z, edge_index, num_neg=5, temperature=0.2):
    # 归一化
    z = F.normalize(z, dim=1)
    pos_i, pos_j = edge_index

    # 负采样若干倍
    neg_edge_index = negative_sampling(
        edge_index=edge_index, 
        num_nodes=z.size(0), 
        num_neg_samples=pos_i.numel() * num_neg, 
        method='sparse'
    )
    neg_i, neg_j = neg_edge_index
    # logits（点积/温度缩放）
    pos_logit = (z[pos_i] * z[pos_j]).sum(dim=1) / temperature
    neg_logit = (z[neg_i] * z[neg_j]).sum(dim=1) / temperature
    # BCE with logits
    pos_loss = F.binary_cross_entropy_with_logits(pos_logit, torch.ones_like(pos_logit))
    neg_loss = F.binary_cross_entropy_with_logits(neg_logit, torch.zeros_like(neg_logit))
    return (pos_loss + neg_loss) * 0.5




# ···················································································联邦学习代码························································
# ···················································································联邦学习代码························································
class RunningAverage():
    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)

def log_creater(output_dir,expname):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    log_name='{}.log'.format(expname)
    #log_name = mode+'_{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
    final_log_file = os.path.join(output_dir,log_name)

    # creat a log
    log = logging.getLogger('train_log')
    log.setLevel(logging.DEBUG)

    # FileHandler
    file = logging.FileHandler(final_log_file,'w')
    file.setLevel(logging.DEBUG)

    # StreamHandler
    stream = logging.StreamHandler()
    stream.setLevel(logging.DEBUG)

    # Formatter
    formatter = logging.Formatter(
        '[%(asctime)s][line: %(lineno)d] ==> [INFO] %(message)s')

    # setFormatter
    file.setFormatter(formatter)
    stream.setFormatter(formatter)

    # addHandler
    log.addHandler(file)
    log.addHandler(stream)

    log.info('creating {}'.format(final_log_file))
    return log


def save_checkpoint(model, checkpoint,log, expname):
    #model_name = os.path.join(checkpoint, 'mode_{}_'.format(mode)+'epoch_' + str(epoch) + '_model.pth.tar')
    model_name=os.path.join(checkpoint, '{}'.format(expname) + '_modelbest.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    log.info("Saving checkpoint... ")
    log.info("Saved to {}".format(model_name))
    with open(model_name, 'wb') as f:
        torch.save(model.state_dict(), f)



def initialClientModel(serverModel,device,input_dim=2000,hidden_dim=1024,latent_dim=64,dropout=0.1):
    clientModel = GAE_GCN_3L(input_dim=input_dim,
                        hidden_dim=hidden_dim, latent_dim=latent_dim,
                        dropout=dropout).to(device)
    clientModel=clientModel.to(device=device)
    # clientModel = torch.nn.DataParallel(clientModel).to(device)# .cuda()
    clientModel.load_state_dict(serverModel.state_dict())
    return clientModel

def compute_grad_update(old_model, new_model, lr, device):
    # maybe later to implement on selected layers/parameters
    if device:
        old_model, new_model = old_model.to(device), new_model.to(device)
    return [(new_param.data - old_param.data)/(-lr) for old_param, new_param in zip(old_model.parameters(), new_model.parameters())]


def add_update_to_model(model, update, weight=1.0, device=None):
    if not update: return model
    if device:
        model = model.to(device)
        update = [param.to(device) for param in update]

    for param_model, param_update in zip(model.parameters(), update):
        param_model.data += weight * param_update.data
    return model



# 差分隐私
# 这是联邦服务器端模型聚合的 核心函数: 它将多个客户端训练的模型差异（或梯度更新）聚合起来，然后更新服务器模型。
def updateServerModel(server_model, grad_updates, lr, device=None, epsilon=8, delta=1e-3,l2_norm_clip=5):
    aggregated=[]

    if True:# 差分隐私 加噪
        for i in range(len(grad_updates)):
            aggregated.append(dp(grad_updates[i], device=device ,epsilon=epsilon, delta=delta,l2_norm_clip=l2_norm_clip))
    else:   # 正常
        aggregated=grad_updates

    add_update_to_model(server_model, aggregated, weight=-1.0 * lr)
    return server_model

# 没用差分隐私
# 这是联邦服务器端模型聚合的 核心函数: 它将多个客户端训练的模型差异（或梯度更新）聚合起来，然后更新服务器模型。
def updateServerModel_111(server_model, grad_updates, device=None):
    aggregated=[]
    aggregated=grad_updates
    add_update_to_model(server_model, aggregated, weight=-1.0 * lr)
    return server_model




def Train(logger,
            dataset_name,
            device,

            clients,
            serverModel,
            numberClients=1,

            com_time=100,
            num_epochs=5, 
            lr=0.001,
            gnn_hidden=1024, 
            gnn_latent=64,
            num_neg=10, 
            temperature=0.2,
            graph_method="pearson",
            n_clusters=10, 
            random_state=0,
            epsilon_l=0.02,    # 0.2   8   20
            delta_l=1e-5, # 1e-3 
            l2_clip=2.0,
   
            enable_umap=True,
            umap_outdir="./umap_figs",
            umap_random_state=0,
            umap_point_size=3
          ):

    eval_metric = {"ARI": [], "NMI": [], "CA": [], "JI": []}
    best_ari, best_epoch, best_metrics, best_state = -1.0, -1, None, None
    


    # 客户端 KNN 构图 生成中······
    client_names = list(clients.keys())
    client_graphs = []
    for client_idx in range(numberClients):
        client_data = clients[client_names[client_idx]]
        # 拆分特征和标签
        genes = np.array([feat for feat, label in client_data], dtype=np.float32)   # (n, 2000)

        edge_index, adj_label  = build_edge_index_by_knn(genes, num_clusters=n_clusters, method=graph_method) 
        client_graphs.append((edge_index.to(device), adj_label.to(device)))



    for time in range(com_time):
        loss_avg = RunningAverage()

        logger.info('Epoch {}/{}'.format(time+1, com_time))

        client_Model_list = []
        client_names = list(clients.keys())

        # 客户端训练
        for client_idx in range(numberClients):
            logger.info('client id {}'.format(client_idx))

            client_data = clients[client_names[client_idx]]

            # logger.info(f"client {client_idx}: {len(client_data)} samples")   # 输出样本数量
            # print("---"*100)

            # 拆分特征和标签
            genes = np.array([feat for feat, label in client_data], dtype=np.float32)   # (n, 2000)
            label = np.array([label for feat, label in client_data], dtype=np.int64)     # (n,)
            # 数据 to tensor
            genes = torch.from_numpy(genes).to(device)
            label = torch.from_numpy(label).to(device)

            edge_index, adj_label = client_graphs[client_idx]   

            # initial clientModel
            clientModel = initialClientModel(serverModel, device=device,input_dim=2000,hidden_dim=gnn_hidden,latent_dim=gnn_latent)
            clientModel.train()
            optimizer = torch.optim.Adam(clientModel.parameters(), lr=lr)

            
            # 本地多 epoch 训练
            for _ in range(num_epochs):
                optimizer.zero_grad()


                ##### # VGAE  重建loss
                # z_vgae_h, adj_recon, mu, logvar = clientModel(genes, edge_index)
                # loss_vgae_rec = vgae_loss(adj_recon, adj_label, mu, logvar)

                
                # ===== 测试 GAE ====
                z_vgae_h, adj_recon = clientModel(genes, edge_index)
                loss_vgae_rec = gae_loss(adj_recon, adj_label)


                #### Pair 对比损失
                loss_pair_vgae = edge_contrastive_loss(z_vgae_h, edge_index, num_neg=num_neg, temperature=temperature)

                total_loss = (loss_vgae_rec+loss_pair_vgae)*0.5
                total_loss.backward()


                optimizer.step()
                loss_avg.update(total_loss.item())

            client_Model_list.append(clientModel)
            logger.info("Length of model list: {}".format(len(client_Model_list)))


        # 梯度聚合
        aggregated_grads = []
        for clientModel in client_Model_list:
            grads = compute_grad_update(serverModel, clientModel, lr, device)
            try:        # update to serverModel
                aggregated_grads = [aggregated_grads[i] + grads[i] for i in range(len(grads))]
            except:
                print('** can not sum up, if this alarm continuously shows more than once, check!')
                aggregated_grads = grads


        # print(aggregated_grads)
        print(type(aggregated_grads))

        aggregated_grads = [x / numberClients for x in aggregated_grads]


        # 这样设置就不会应用差分隐私策略！！！！！！！！！！！！！！！！！
        serverModel = updateServerModel_111(serverModel, aggregated_grads,  device=device)


        # 聚类计算
        features = []
        labels = []
        serverModel.eval()
        print("=> Merging Features ...")
        for client_idx in range(numberClients):
            client_data = clients[client_names[client_idx]]
            # 拆分特征和标签
            genes = np.array([feat for feat, label in client_data], dtype=np.float32)   # (n, 2000)
            label = np.array([label for feat, label in client_data], dtype=np.int64)     # (n,)
            genes = torch.from_numpy(genes).to(device)
            label = torch.from_numpy(label).to(device)

            edge_index, adj_label = client_graphs[client_idx] 
            with torch.no_grad():
                z_vgae_h, adj_recon = clientModel(genes, edge_index)

            features += list(z_vgae_h.detach().cpu().numpy())
            labels += list(label.detach().cpu().numpy())



        # 检查features，如果有空值和无限大的值，就替换掉
        # Check for NaN or infinity values in features
        if np.any(np.isnan(features)):
            print("Features contain NaN values")
            # assert 1==0
        if np.any(np.isinf(features)):
            print("Features contain infinity values")
            # assert 1==0
            # Replace NaN or infinity values with a large finite number, like 0
            features = np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)



        # print(np.array(features).shape,'feas+'*80)
        print("=> K Means Clustering ...")
        kmeans = KMeans(n_init="auto",  random_state=random_state).fit(np.array(features))   # k=7  n_clusters       random_state    0
        pred = kmeans.labels_


        labels=np.array(labels)
        print(pred.shape,'+'*80)
        # print(pred)
        print(labels.shape,'-'*80)
        # print(labels)

        metrics = M.compute_metrics(labels, pred)

        
        # # 日志
        log = f"[Ep {time:03d}] VGAE_rec={loss_vgae_rec.item():.4f} pair(VGAE)={loss_pair_vgae.item():.4f} " \
              f"| total={total_loss.item():.4f}" f"| ARI={metrics['ARI']:.4f} NMI={metrics['NMI']:.4f} CA={metrics['CA']:.4f} " \
                f"JI={metrics['JI']:.4f} CS={metrics['CS']:.4f} HS={metrics['HS']:.4f} VMS={metrics['VMS']:.4f}"
        print(log)

        # 记录与 best
        for k in ("ARI","NMI","CA","JI"):
            eval_metric[k].append(float('%.4f' % metrics[k]))

        if metrics["ARI"] > best_ari:
            best_ari = metrics["ARI"]; best_time = time + 1
            best_metrics = metrics.copy()

        
            best_emb = features.copy()
            best_pred = pred.copy()

            pred_aligned = align_cluster_labels(labels, best_pred)  # 用labels，比用Y 多出了几个， 从3909 变成了3935 个样本，是因为联邦学习分样本的原因！


        print(f"\nBest model saved from Epoch {best_time}")
        print("Best metrics:", best_metrics)    


    print(f"\nBest model saved from Epoch {best_time}")
    print("Best metrics:", best_metrics)    
    


    # 训练结束后：对 best epoch 做 UMAP 可视化（按 pred 和按真值）
    if enable_umap and best_emb is not None and best_pred is not None:
        os.makedirs(umap_outdir, exist_ok=True)
        umap_outdir=umap_outdir + f"/{dataset_name}"

        pred_path = os.path.join(umap_outdir, f"{dataset_name}_DL_SC_best_pred_epoch{best_time}.png")
        gt_path = os.path.join(umap_outdir, f"{dataset_name}_DL_SC_best_gt_epoch{best_time}.png")


               # 计算类别颜色
        GLOBAL_COLOR_MAP = build_fixed_color_map(37)    # 该数据集一共37类,也就是有37个颜色.

        # best_emb 是你在 best_epoch 得到的嵌入（比如 Z 或 embeddings）
        # 图 1：按聚类结果上色（但标签已经对齐真实类别）
        plot_tsne(
            emb_input=best_emb,
            labels=pred_aligned,
            title="PFDL_SC",
            save_path=pred_path,
            point_size=3,
            random_state=0,
            perplexity=80,
            class_names=class_names,
            color_map=GLOBAL_COLOR_MAP
        )

        plot_tsne(
            emb_input=best_emb,
            labels=labels,
            title="PFDL_SC (Ground Truth)",
            save_path=gt_path,
            point_size=3,
            random_state=0,
            perplexity=80,
            class_names=class_names,
            color_map=GLOBAL_COLOR_MAP
        )

        print(f"UMAP figures saved to: {umap_outdir}")


    

    #########       生成csv表格，用于几类不同聚类的数目，用以后续构建 类别频次热图！！！
    # === 构建数量统计表 ===
    count_table = build_cluster_count_table(Y, pred_aligned, class_names)
    

    umap_outdir=umap_outdir +'/'+dataset_name
    # 保存为 CSV（论文用）
    table_save_path = os.path.join(
        umap_outdir +  f"_FL_SC_cluster_count_table.csv"
    )

    count_table.to_csv(table_save_path, index=False)

    print("\nCluster count table:")
    print(count_table.head())
    print(f"Saved to: {table_save_path}")


    return serverModel, eval_metric









if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser(description='FL-VGAE pipeline for scRNA-seq"')
    parser.add_argument('--device', default='cuda:0', help='default: cuda:0')
    parser.add_argument('--epochs', type=int, default=5, help='default: 100')#   每一轮内部实验的 epoch  5
    parser.add_argument('--lr', type=float, default=0.001, help='default: 0.01')
    parser.add_argument('--dataset', default='yan', help='dataset_name')
    # VGAE模型
    parser.add_argument("--gnn-hidden", type=int, default=1024) # GNN层隐藏层维度
    parser.add_argument("--gnn-latent", type=int, default=128)   # GNN层特征编码维度
    parser.add_argument("--graph-method", type=str, default="pearson", )    # choices=["pearson","cosine","gaussian"]
    parser.add_argument("--num-neg", type=int, default=10)  # 负样本倍数
    parser.add_argument("--temperature", type=float, default=0.2)   # 温度系数，用于负样本生成
    # FL 联邦学习
    parser.add_argument('--com-time', type=int, default=10, help='default: 3')# 通信轮询次数    30
    parser.add_argument('--client', type=int, default=3, help='default: 3')     # 客户端数目    3
    parser.add_argument('--expname', default="FL-GAE", help='experiment name')    # 实验名称
    # DP 差分隐私
    parser.add_argument('--epsilon', type=int, default=100, help='default: 20')   #   用于差分隐私，
    parser.add_argument('--delta', type=float, default=1e-5, help='default: 10e-5')
    parser.add_argument('--l2_clip', type=int, default=5, help='default: 5')    # 高斯加噪参数： L2 全局敏感度的上限 (GS >= 0)



    args = parser.parse_args()

    random_state = 0    # 12345
    # 设置随机种子---
    set_global_seed(random_state)

    
    dataset_name = args.dataset
    EPOCHS = args.epochs
    lr = args.lr
    if dataset_name in ["Shekhar", "Tosches", "Bach", "hrvatin"]:      # ,"Pollen"
        device = "cpu"  # 在较大的数据集上，使用cpu，使其得以运行····
    else:
        device = "cuda:0"  

    # device = "cpu"


    expname = args.expname

    com_time=args.com_time
    numberClients = args.client

    epsilon = args.epsilon
    delta = args.delta

    dp_com_time=args.com_time# -20
    # 将全局隐私预算拆分为每一轮每个客户端的预算（重要！）
    epsilon_l = args.epsilon/(2*np.sqrt(2*dp_com_time*np.log(2/args.delta)))
    delta_l = args.delta/(2*dp_com_time)

    # print(epsilon_l,delta_l)
    epsilon_l = epsilon_l * np.sqrt(args.client) / (np.sqrt(np.log(2/delta_l)))
    delta_l = delta_l/args.client
    # print(epsilon_l,delta_l)
    # assert 1==0



    logger = log_creater(output_dir=os.path.join(root, "output","{}_epsilon_{}".format(expname,epsilon_l)),
                            expname=expname+"_"+expname +"_{}".format(expname))
    logger.info("epsilon_l: {}".format(epsilon_l))
    logger.info("delta_l: {}".format(delta_l))
    logger.info("communication_time: {}".format(com_time))
    logger.info("Epochs: {}".format(EPOCHS))
    logger.info("lr: {}".format(lr))


    # 针对隐私攻击章节内容 的 代码
    X, Y , class_names = load_h5_data_attack(dataset_name, 2000)

    X_train, y_train = shuffle(X, Y, random_state=random_state)
    X_samples, y_samples = selRepsample(X_train,y_train)        # 采样代表性数据 即 每一类都要抽取出一个样本来，确保每个客户端都有 每一类样本存在！！！

    ### 计算聚类的类别数量！！！！！！          获取唯一的类别    并通过len（）计算类别数目   
    n_clusters = len(np.unique(y_train))
    # # 设置客户端，划分样本
    clients = create_clients(X_samples, y_samples,X_train, y_train, num_clients=numberClients, initial='client') 

    # === 从 model_base 构建模型 ===
    serverModel = GAE_GCN_3L(input_dim=2000, hidden_dim=args.gnn_hidden, latent_dim=args.gnn_latent, dropout=0.1).to(device)
    serverModel = serverModel.to(device)

    # start train
    trained_model , eval_metric = Train(logger,
                                        dataset_name,
                                        device,

                                        clients,
                                        serverModel,
                                        numberClients=numberClients,

                                        com_time=com_time,
                                        num_epochs=EPOCHS, 
                                        lr=lr,
                                        gnn_hidden=args.gnn_hidden, 
                                        gnn_latent=args.gnn_latent,
                                        num_neg=args.num_neg, 
                                        temperature=args.temperature,
                                        graph_method=args.graph_method,
                                        n_clusters=n_clusters, 
                                        random_state=random_state,

                                        epsilon_l= epsilon_l, 
                                        delta_l=delta_l, 
                                        l2_clip=args.l2_clip
                                        )

    # df = pd.DataFrame(eval_metric)
    # df.to_csv(os.path.join(root,
    #                         "output",
    #                         "{}_epsilon_{}".format(expname,epsilon),
    #                         "{}_{}_e_{}_train_statistics.csv".format(expname, dataset_name,epsilon)), index=False)









