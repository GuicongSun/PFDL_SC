# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("Agg")  # headless server 必备
import matplotlib.pyplot as plt

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
from utils import load_h5_data1, load_h5_data2

from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix



def selRepsample(X,y):
    X_samples,y_samples = [],[]
    y_set = set(list(y))
    for y_label in y_set:
        for x_sample,y_sample in zip(X,y):
            if y_sample == y_label:
                X_samples.append(x_sample)
                y_samples.append(y_sample)
                break
    return X_samples,y_samples




############# 实现 “生物学分析”功能的完整 Python 脚本模块
#################################################################################################################
# Let's implement the DEG analysis and plotting part based on the description.
# This script assumes you have the clustering predictions (pred), true labels (Y),
# and expression matrix (X) after running the model.
'''
1. compute_top_deg(...)
功能：使用 Wilcoxon 秩和检验，计算每个聚类簇的前50个差异表达基因（DEG）
2. compute_deg_overlap_heatmap(...)
功能：生成热图，对比模型预测簇和真实细胞类型之间的 DEG 重叠比例
3. find_marker_genes(...)
功能：提取每个簇中表达量最高的前2个基因，作为候选标记基因
4. plot_marker_gene_expression(...)
功能：画出 marker 基因在不同簇中的平均表达热图
'''

import numpy as np
import pandas as pd
from scipy.stats import ranksums
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def compute_top_deg(X, labels, top_k=50):
    """
    Compute top-K DEGs for each cluster using Wilcoxon rank-sum test.
    Args:
        X (ndarray): Gene expression matrix (cells x genes)
        labels (ndarray): Cluster labels for each cell
        top_k (int): Number of DEGs to keep per cluster
    Returns:
        dict: cluster_id -> list of top K gene indices
    """
    n_clusters = len(np.unique(labels))
    n_genes = X.shape[1]
    cluster_deg = {}

    for c in range(n_clusters):
        in_cluster = X[labels == c]
        out_cluster = X[labels != c]
        p_values = []
        for g in range(n_genes):
            try:
                stat, p = ranksums(in_cluster[:, g], out_cluster[:, g])
            except ValueError:
                p = 1.0
            p_values.append(p)
        p_values = np.array(p_values)
        top_genes = np.argsort(p_values)[:top_k]
        cluster_deg[c] = top_genes.tolist()
    
    return cluster_deg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import linear_sum_assignment

from matplotlib import colors

from matplotlib.colors import LinearSegmentedColormap

import textwrap

def auto_wrap(label, width=10):
    return "\n".join(textwrap.wrap(label, width=width))

def compute_deg_overlap_heatmap(
    cluster_degs,
    true_degs,
    cluster_labels,
    true_labels,
    class_names,
    out_file="deg_overlap_heatmap.tif"
):

    # ===== 1. 计算 overlap matrix =====
    overlap_matrix = np.zeros((len(cluster_labels), len(true_labels)))

    for i, c in enumerate(cluster_labels):
        pred_genes = set(cluster_degs[c])
        for j, t in enumerate(true_labels):
            true_genes = set(true_degs[t])
            overlap = len(pred_genes & true_genes) / len(pred_genes)
            overlap_matrix[i, j] = overlap * 100

    # ===== 2. 只根据最大匹配重排 cluster（不动 cell type 顺序）=====
    best_match = np.argmax(overlap_matrix, axis=1)
    sorted_cluster_indices = np.argsort(best_match)

    aligned_matrix = overlap_matrix[sorted_cluster_indices, :]
    aligned_clusters = [f"Cluster {cluster_labels[i]}" for i in sorted_cluster_indices]
    aligned_celltypes = [class_names[j] for j in true_labels]


    df = pd.DataFrame(
        aligned_matrix,
        index=aligned_clusters,
        columns=aligned_celltypes
    )

    # ===== 3. 画热图（期刊风格）=====
    plt.figure(figsize=(7.5, 6))
    # 定义深蓝色
    white_to_darkblue = LinearSegmentedColormap.from_list("white_to_darkblue", ["#f7fbff", "#c3cfda", "#7ea8c0", "#2c6394", "#051d41"])
    ax = sns.heatmap(
        df,
        # annot=False,    
        annot=True,    
        fmt=".0f",          # ← 保留两位小数
        cmap = white_to_darkblue,

        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "DEG overlap (%)"},
        vmin=0,
        vmax=100
    )


    # 2️⃣ 纵坐标：只保留 cluster 数字，不要 "Cluster"
    ax.set_yticklabels(
        [label.replace("Cluster ", "") for label in df.index],
        rotation=0,
        fontsize=14
    )

    wrapped_labels = [
        auto_wrap(lbl.get_text(), width=12)
        for lbl in ax.get_xticklabels()
    ]
    # 自己手动设置！！！
    wrapped_labels=[
        "B cell",
        "T cell",
        "endothelial cell",
        "macrophage",
        "mesenchymal\nstem cell",
        "skeletal muscle\nsatellite cell"  # 👈 人为控制只换一次行
    ]

    ax.set_xticklabels(
        wrapped_labels,
        rotation=45,
        ha="center",
        fontsize=12
    )


    ax.set_ylabel("Predicted clusters", fontsize=16)
    ax.set_title("base-SC",fontsize=16)               


    # ===== 放大 colorbar 字体 =====
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("DEG overlap (%)", fontsize=16)


    plt.tight_layout()
    plt.savefig(out_file, dpi=400, format="tiff")
    plt.close()


    return df



def find_marker_genes(X, labels, top_n=2):
    """
    For each cluster, return top N genes with highest average expression.
    """
    marker_dict = {}
    for c in np.unique(labels):
        cluster_expr = X[labels == c]
        avg_expr = cluster_expr.mean(axis=0)
        top_genes = np.argsort(-avg_expr)[:top_n]
        marker_dict[c] = top_genes.tolist()
    return marker_dict


def plot_marker_gene_expression(X, labels, marker_genes, gene_names, cluster_name_map=None, out_file="output/base-SC_marker_gene_expression.tif"):
    """
    Plot heatmap of marker gene average expression across clusters.

    Args:
        X: ndarray, shape (n_samples, n_genes)
        labels: ndarray, shape (n_samples,), predicted cluster labels
        marker_genes: dict, {cluster_id: [gene_index1, gene_index2, ...]}
        gene_names: list[str], length = n_genes
        cluster_name_map: dict[int, str], optional, maps cluster id to name (e.g., {0: "B cell", 1: "T cell", ...})
        out_file: str, path to save the heatmap
    """

    n_clusters = len(np.unique(labels))
    marker_gene_list = sorted(set(g for genes in marker_genes.values() for g in genes))
    
    data = []
    for c in range(n_clusters):
        avg_expr = X[labels == c].mean(axis=0)
        row = [avg_expr[g] for g in marker_gene_list]
        data.append(row)

    # 构造 DataFrame，横轴为基因名
    gene_labels = [gene_names[i] for i in marker_gene_list]
    df = pd.DataFrame(data, columns=gene_labels)

    # 重新设置纵坐标顺序，按指定的新顺序
    new_cluster_order = [3, 1, 0, 4, 2, 5]  # 你希望的顺序
    df_reordered = df.iloc[new_cluster_order]

    # 设置纵轴为细胞名称（如果有提供映射）
    if cluster_name_map:
        row_labels = [cluster_name_map.get(i, f"Cluster {i}") for i in new_cluster_order]
    else:
        row_labels = [f"Cluster {i}" for i in new_cluster_order]
    
    # 更新行标签
    df_reordered.index = row_labels

    # 绘制热图
    plt.figure(figsize=(12, 6))

    white_to_darkblue = LinearSegmentedColormap.from_list("white_to_darkblue", ["#f7fbff", "#c3cfda", "#7ea8c0", "#2c6394", "#051d41"])

    ax = sns.heatmap(
        df_reordered,
        annot=False,
        cmap=white_to_darkblue,
        cbar_kws={'label': 'Average Expression'},
        linewidths=0.5,
        linecolor="white",
    )

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=14)

    # 设置纵轴标签
    wrapped_labels = [
        "B cell",
        "T cell",
        "endothelial cell",
        "macrophage",
        "mesenchymal\nstem cell",
        "skeletal muscle\nsatellite cell"
    ]
    ax.set_yticklabels(wrapped_labels, rotation=0, fontsize=14)

    # 调整 colorbar 字体大小
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label("Mean Expression in Group", fontsize=16)

    plt.tight_layout()
    plt.savefig(out_file, dpi=400, format="tiff")
    plt.close()

    return df_reordered

#################################################################################################################
















def align_cluster_labels(y_true, y_pred):
    """
    使用 Hungarian algorithm 对齐聚类标签，使 pred 的编号尽量匹配 y_true
    """
    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    aligned_pred = np.array([mapping[p] for p in y_pred])
    return aligned_pred


# 统一种子（在任何随机操作和模型初始化之前调用一次）
def set_global_seed(seed: int = 12345, deterministic: bool = True):
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


#   对比学习 loss
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


def Train_VGAE(
    dataset_name,
    X, Y,
    num_epochs=100,
    lr=1e-3,
    gnn_hidden=512, gnn_latent=128,
    num_neg=10, 
    temperature=0.2,
    graph_method="pearson", # gaussian
    device="cpu",
   
):
    n_nodes, input_dim = X.shape
    x = torch.from_numpy(X).float().to(device)

    # Step-1: KNN 构图
    num_clusters = len(np.unique(Y))
    edge_index, adj_label  = build_edge_index_by_knn(X, num_clusters=num_clusters, method=graph_method, combine_with=None, alpha=0.5) # combine_with="gaussian"   None
    edge_index = edge_index.to(device)
    adj_label = adj_label.to(device)

    vgae = GAE_GCN_3L(input_dim=input_dim, hidden_dim=gnn_hidden, latent_dim=gnn_latent, dropout=0.1).to(device)

    optimizer = torch.optim.Adam(list(vgae.parameters()), lr=lr)

    eval_metric = {"ARI": [], "NMI": [], "CA": [], "JI": []}
    best_ari, best_epoch, best_metrics, best_state = -1.0, -1, None, None

    best_emb = None
    best_pred = None

    # 使用 ReduceLROnPlateau 来降低学习率
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=10, factor=0.5, min_lr=1e-6, verbose=True)

    for epoch in range(1, num_epochs + 1):
        vgae.train()
        optimizer.zero_grad()

        ### ===== GAE 前向 =====
        z_vgae_h, adj_recon = vgae(x, edge_index)
        loss_vgae_rec = gae_loss(adj_recon, adj_label)
        
        #### Pair 对比损失
        loss_pair_vgae = edge_contrastive_loss(z_vgae_h, edge_index, num_neg=num_neg, temperature=temperature)

        loss = (loss_vgae_rec+loss_pair_vgae)*0.5


        loss.backward()
        optimizer.step()


        z_vgae_h = z_vgae_h.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=num_clusters,n_init="auto", random_state=0).fit(z_vgae_h)
        pred = kmeans.labels_
        metrics = M.compute_metrics(Y, pred)


        # # 日志
        log = f"[Ep {epoch:03d}] VGAE_rec={loss_vgae_rec.item():.4f} pair(VGAE)={loss_pair_vgae.item():.4f} " \
              f"| total={loss.item():.4f}" f"| ARI={metrics['ARI']:.4f} NMI={metrics['NMI']:.4f} CA={metrics['CA']:.4f} " \
                f"JI={metrics['JI']:.4f} CS={metrics['CS']:.4f} HS={metrics['HS']:.4f} VMS={metrics['VMS']:.4f}"
        print(log)


        for k in eval_metric.keys():
            eval_metric[k].append(float('%.3f' % metrics[k]))

        # # 使用学习率调度器
        # scheduler.step(metrics["ARI"])

        if metrics["ARI"] > best_ari:
            best_ari = metrics["ARI"]
            best_epoch = epoch
            best_metrics = metrics.copy()

            best_emb = z_vgae_h.copy()
            best_pred = pred.copy()

            print(Y.shape)
            print(best_pred.shape)
            print('-'*100)

            pred_aligned = align_cluster_labels(Y, best_pred)

    print(f"\nBest model saved from Epoch {best_epoch}")
    print("Best metrics:", best_metrics)



    # Step 1: 预测簇 & 真实类型，分别计算前50个 DEG
    pred_deg = compute_top_deg(X, best_pred, top_k=50)
    true_deg = compute_top_deg(X, Y, top_k=50)

    compute_deg_overlap_heatmap(
        cluster_degs=pred_deg,
        true_degs=true_deg,
        cluster_labels=list(pred_deg.keys()) ,
        true_labels=list(true_deg.keys()),
        class_names=class_names,
        out_file="output/base-SC-deg_overlap_heatmap.tif"
    )


    # Step 3: 找每个簇中表达量最高的 2 个基因（marker）
    marker_genes = find_marker_genes(X, best_pred, top_n=3)

    # Step 4: 显示 marker gene 的表达热图（保存为 marker_gene_expression.png）
    # gene_names: 是你加载数据时 2000 个基因的名称（必须有）
    plot_marker_gene_expression(X, best_pred, marker_genes, gene_names)


    return vgae, eval_metric




# ===============================
# 主程序
# ===============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VGAE pipeline for scRNA-seq")
    parser.add_argument("--dataset", type=str, default="yan", help="dataset name (without .h5)")
    parser.add_argument("--epochs", type=int, default=100, help="number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")

    parser.add_argument("--gnn-hidden", type=int, default=1024)
    parser.add_argument("--gnn-latent", type=int, default=128)
    parser.add_argument("--graph-method", type=str, default="pearson", )    # choices=["pearson","cosine","gaussian"]

    parser.add_argument("--num-neg", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.2)
  

    args = parser.parse_args()

    # 设置随机种子---
    set_global_seed(0)

    dataset_name = args.dataset
    EPOCHS = args.epochs
    lr = args.lr

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if dataset_name in ["Shekhar", "Tosches", "Bach", "hrvatin"]:  # hrvatin数据太大，需要用cpu跑！！！
        device = "cpu"
    else:
        device = "cuda:0"  # 或者其他默认设备

    # device = "cpu"


    # 加载数据（X: n x 2000, Y: labels）
    data_path = "./data/" + dataset_name + ".h5"
    # 不同的文件类型，需要的是不同的
    if dataset_name in ["Muraro", "Quake_10x_Bladder", "Quake_10x_Limb_Muscle","Young","Quake_Smart-seq2_Diaphragm"]:   
        #                       2122                2500                   3909     5685                            870            
        X, Y, class_names,gene_names  = load_h5_data2(data_path, 2000)
    else:
        X, Y , class_names,gene_names = load_h5_data1(data_path, 2000)

    


    # 打乱 & 可选重采样（保持你原有流程）
    X_train, y_train = shuffle(X, Y, random_state=0)
    X_samples, y_samples = selRepsample(X_train, y_train)  # 若要替换，可将 X_train = X_samples

    # 训练
    vgae, eval_metric = Train_VGAE(
        dataset_name=dataset_name,
        X=X_train, Y=y_train,
        num_epochs=EPOCHS, lr=lr,
        gnn_hidden=args.gnn_hidden, gnn_latent=args.gnn_latent,
        
        num_neg=args.num_neg, 
        temperature=args.temperature,
        graph_method=args.graph_method,
        device=device
    )

    print("Training finished!")



