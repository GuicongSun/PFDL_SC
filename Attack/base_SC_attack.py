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
from utils import load_h5_data1, load_h5_data2 , load_h5_data_attack
from readFile import selRepsample

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

import pandas as pd
from sklearn.metrics import confusion_matrix


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
   
    enable_umap=True,
    umap_outdir="./umap_figs",
    umap_random_state=0,
    umap_point_size=3
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
        # kmeans = KMeans(n_clusters=num_clusters,n_init="auto", random_state=0).fit(z_vgae_h)
        kmeans = KMeans(n_init="auto", random_state=0).fit(z_vgae_h)

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


    # 训练结束后：对 best epoch 做 UMAP 可视化（按 pred 和按真值）
    if enable_umap and best_emb is not None and best_pred is not None:
        os.makedirs(umap_outdir + f"/{dataset_name}", exist_ok=True)
        umap_outdir=umap_outdir + f"/{dataset_name}"
        pred_path = os.path.join(umap_outdir, f"{dataset_name}_base_SC_best_pred_epoch{best_epoch}.png")
        gt_path = os.path.join(umap_outdir, f"{dataset_name}_base_SC_best_gt_epoch{best_epoch}.png")


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
        umap_outdir +  f"_base_SC_cluster_count_table.csv"
    )

    count_table.to_csv(table_save_path, index=False)

    print("\nCluster count table:")
    print(count_table.head())
    print(f"Saved to: {table_save_path}")


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


    # 针对隐私攻击章节内容 的 代码
    X, Y , class_names = load_h5_data_attack(dataset_name, 2000)


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



