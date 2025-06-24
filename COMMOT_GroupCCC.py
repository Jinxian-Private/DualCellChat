import os
import gc
import ot
import pickle
import anndata
import scanpy as sc
import pandas as pd
import numpy as np

from scipy import sparse
from scipy.stats import spearmanr, pearsonr
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt

import commot as ct

from matplotlib import cm
from matplotlib.lines import Line2D
from typing import Optional, Union
import plotly
import networkx as nx
from networkx.drawing.nx_agraph import to_agraph

def get_cmap_qualitative(cmap_name):
    if cmap_name == "Plotly":
        cmap = plotly.colors.qualitative.Plotly
    elif cmap_name == "Alphabet":
        cmap = plotly.colors.qualitative.Alphabet
    elif cmap_name == "Light24":
        cmap = plotly.colors.qualitative.Light24
    elif cmap_name == "Dark24":
        cmap = plotly.colors.qualitative.Dark24
    # Safe and Vivid are strings of form "rbg(...)"
    # Handle this later.
    elif cmap_name == "Safe":
        cmap = plotly.colors.qualitative.Safe
    elif cmap_name == "Vivid":
        cmap = plotly.colors.qualitative.Vivid
    return cmap

def summarize_cluster(X, clusterid, clusternames, n_permutations=500):
    # Input a sparse matrix of cell signaling and output a pandas dataframe
    # for cluster-cluster signaling
    n = len(clusternames)
    X_cluster = np.empty([n,n], float)
    p_cluster = np.zeros([n,n], float)
    for i in range(n):
        tmp_idx_i = np.where(clusterid==clusternames[i])[0]
        for j in range(n):
            tmp_idx_j = np.where(clusterid==clusternames[j])[0]
            X_cluster[i,j] = X[tmp_idx_i,:][:,tmp_idx_j].mean()
    for i in range(n_permutations):
        clusterid_perm = np.random.permutation(clusterid)
        X_cluster_perm = np.empty([n,n], float)
        for j in range(n):
            tmp_idx_j = np.where(clusterid_perm==clusternames[j])[0]
            for k in range(n):
                tmp_idx_k = np.where(clusterid_perm==clusternames[k])[0]
                X_cluster_perm[j,k] = X[tmp_idx_j,:][:,tmp_idx_k].mean()
        p_cluster[X_cluster_perm >= X_cluster] += 1.0
    p_cluster = p_cluster / n_permutations
    df_cluster = pd.DataFrame(data=X_cluster, index=clusternames, columns=clusternames)
    df_p_value = pd.DataFrame(data=p_cluster, index=clusternames, columns=clusternames)
    return df_cluster, df_p_value

def cluster_communication(
    database_name: str = None,
    pathway_name: str = None,
    lr_pair = None,
    clustering: str = None,
    n_permutations: int = 100,
    random_seed: int = 1,
    copy: bool = False
):
    #########
    # input:
    # 1.total class types: celltypes = list( adata.obs[clustering].unique() )
    # 2.class label for each cell: clusterid = np.array(adata.obs[clustering], str)
    # 3.communication matrix: S = adata.obsp['commot-'+obsp_names[i]] pathway/lr_pair
    # output: communication_matrix(tmp_df), communication_pvalue(tmp_p_value)
    ##########
    np.random.seed(random_seed)

    # assert database_name is not None, "Please at least specify database_name."

    # celltypes = list( adata.obs[clustering].unique() )
    # celltypes.sort()
    # for i in range(len(celltypes)):
    #     celltypes[i] = str(celltypes[i])
    # clusterid = np.array(adata.obs[clustering], str)
    # obsp_names = []
    # if not lr_pair is None:
    #     obsp_names.append(database_name+'-'+lr_pair[0]+'-'+lr_pair[1])
    # elif not pathway_name is None:
    #     obsp_names.append(database_name+'-'+pathway_name)
    # else:
    #     obsp_names.append(database_name+'-total-total')
    # # name_mat = adata.uns['commot-'+pathway_name+'-info']['df_ligrec'].values
    # # name_mat = np.concatenate((name_mat, np.array([['total','total']],str)), axis=0)
    # for i in range(len(obsp_names)):
    # df_class = np.load('generated_data/V1_Breast_Cancer_Block_A_Section_1/cell_types.npy')

    # cell_type_indeces = np.load('generated_data/V1_Breast_Cancer_Block_A_Section_1/cell_types.npy')
    # cell_type_dict = {0:0,1:0,2:1,3:1,4:1,5:1,6:1,7:1,8:2,9:2,10:2,11:2,12:2,13:3,14:3,15:3,16:3,17:3,18:3,19:3}
    # cell_type = [] 
    # for cell_i in cell_type_indeces:
    #     cell_type.append(cell_type_dict[cell_i])
    # clusterid = np.array(cell_type)
    clusterid = np.load('generated_data/V1_Breast_Cancer_Block_A_Section_1/cell_types.npy')
    celltypes = list( set(clusterid) )
    celltypes.sort()
    print(celltypes)

    S = np.load('results/matrix/adj_pred_4.npy')
    # S = adata.obsp['commot-'+obsp_names[i]]
    tmp_df, tmp_p_value = summarize_cluster(S,
        clusterid, celltypes, n_permutations=n_permutations)
    cluster_CCI = {'communication_matrix': tmp_df, 'communication_pvalue': tmp_p_value}

    # adata.uns['commot_cluster-'+clustering+'-'+obsp_names[i]] = {'communication_matrix': tmp_df, 'communication_pvalue': tmp_p_value}
    
    # return adata if copy else None
    return cluster_CCI

def linear_clamp_value(x, lower_bound, upper_bound, out_min, out_max):
    if x <= lower_bound:
        y = out_min
    elif x >= upper_bound:
        y = out_max
    else:
        y = out_min + (x - lower_bound)/(upper_bound-lower_bound) * (out_max-out_min)
    return y

def plot_cluster_signaling_network(S,
    labels = None,
    node_size = 0.2,
    node_colormap = "Plotly",
    node_cluster_colormap = None,
    node_pos = None,
    edge_width_lb_quantile = 0.05,
    edge_width_ub_quantile = 0.95,
    edge_width_min = 1,
    edge_width_max = 4,
    edge_color = None, # expect to range from 0 to 1
    edge_colormap = None,
    background_pos = None,
    background_ndcolor = "lavender",
    background_ndsize = 1,
    filename = "network_plot.pdf",
):
    if labels is None:
        labels = [str(i) for i in range(S.shape[0])]
    node_cmap = get_cmap_qualitative(node_colormap)
    G = nx.MultiDiGraph()

    edge_width_lb = np.quantile(S.reshape(-1), edge_width_lb_quantile)
    edge_width_ub = np.quantile(S.reshape(-1), edge_width_ub_quantile)


    # Draw the background geometry
    if not background_pos is None:
        for i in range(background_pos.shape[0]):
            G.add_node("cell_"+str(i), shape='point', color=background_ndcolor, fillcolor=background_ndcolor, width=background_ndsize)
            G.nodes["cell_"+str(i)]["pos"] = "%f,%f!" %(background_pos[i,0],background_pos[i,1])

    # Draw the nodes (cluster)
    for i in range(len(labels)):
        if node_cluster_colormap is None:
            G.add_node(labels[i], shape="point", fillcolor=node_cmap[i], color=node_cmap[i])
        elif not node_cluster_colormap is None:
            G.add_node(labels[i], shape="point", fillcolor=node_cluster_colormap[labels[i]], color=node_cmap[i])
        if not node_pos is None:
            G.nodes[labels[i]]["pos"] = "%f,%f!" % (node_pos[i,0],node_pos[i,1])
        G.nodes[labels[i]]["width"] = str(node_size)

    # Draw the edges
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            if S[i,j] > 0:
                G.add_edge(labels[i], labels[j], splines="curved")
                G[labels[i]][labels[j]][0]["penwidth"] = str(linear_clamp_value(S[i,j],edge_width_lb,edge_width_ub,edge_width_min,edge_width_max))
                if edge_color == "node":
                    G[labels[i]][labels[j]][0]['color'] = node_cmap[i]
                elif isinstance(edge_color, np.ndarray):
                    G[labels[i]][labels[j]][0]['color'] = mpl.colors.to_hex( edge_colormap(edge_color[i,j]) )
                else:
                    G[labels[i]][labels[j]][0]['color'] = edge_color
    
    # Draw the network
    A = to_agraph(G)
    if node_pos is None:
        A.layout("dot")
    else:
        A.layout()
    A.draw(filename)

def plot_cluster_communication_network(
    cluster_CCI,
    uns_names: list = None,
    clustering: str = None,
    quantile_cutoff: float = 0.99,
    p_value_cutoff: float = 0.05,
    self_communication_off: bool = False,
    filename: str = None,
    nx_node_size: float = 0.2,
    nx_node_cmap: str = "Plotly",
    nx_node_cluster_cmap: dict = None,
    nx_pos_idx: np.ndarray = np.array([0,1],int),
    nx_node_pos: str = "cluster",
    nx_edge_width_lb_quantile: float = 0.05,
    nx_edge_width_ub_quantile: float = 0.95,
    nx_edge_width_min: float = 1,
    nx_edge_width_max: float = 4,
    nx_edge_color: Union[str, np.ndarray] = "node",
    nx_edge_colormap = cm.Greys,
    nx_bg_pos: bool = True,
    nx_bg_color: str = "lavender",
    nx_bg_ndsize: float = 0.05,
):
    """
    Plot cluster-cluster communication as network.
    *input:
    1. communication_matrix,communication_pvalue from cluster_communication function
    2. labels, from communication_matrix colunms
    *output: png plot
    *external input: 
    1. node_pos
    2. background_pos
    """
    
    # X_tmp = adata.uns[uns_names[0]]['communication_matrix'].copy()
    X_tmp = cluster_CCI['communication_matrix'].copy()
    labels = list( X_tmp.columns.values )
    X = np.zeros_like(X_tmp.values, float)
    for i in range(len(uns_names)):
        X_tmp = cluster_CCI['communication_matrix'].values.copy()
        p_values_tmp = cluster_CCI['communication_pvalue'].values.copy()
        if not quantile_cutoff is None:
            cutoff = np.quantile(X_tmp.reshape(-1), quantile_cutoff)
        else:
            cutoff = np.inf
        tmp_mask = ( X_tmp < cutoff ) * ( p_values_tmp > p_value_cutoff )
        X_tmp[tmp_mask] = 0
        X = X + X_tmp
    X = X / len(uns_names)
    if self_communication_off:
        for i in range(X.shape[0]):
            X[i,i] = 0

    if nx_node_pos == "cluster":
        node_pos = [adata.uns["cluster_pos-"+clustering][labels[i]] for i in range(len(labels)) ]
        node_pos = np.array(node_pos)
        node_pos = node_pos[:, nx_pos_idx]
        lx = np.max(node_pos[:,0])-np.min(node_pos[:,0])
        ly = np.max(node_pos[:,1])-np.min(node_pos[:,1])
        pos_scale = max(lx, ly)
        node_pos = node_pos / pos_scale * 8.0
    else:
        node_pos = None
    if nx_bg_pos:
        background_pos = adata.obsm["spatial"][:,nx_pos_idx]
        background_pos = background_pos / pos_scale * 8.0
    else:
        background_pos = None
    print(X)
    print(labels)
    print(nx_node_size) #0.2
    print(nx_node_cmap) #Light24
    print(nx_node_cluster_cmap) #None
    print('node_pos')
    print(node_pos) #None
    print(nx_edge_width_lb_quantile) #0.05
    print(nx_edge_width_ub_quantile)#0.95
    print(nx_edge_width_min) #1 
    print(nx_edge_width_max) #4
    print(nx_edge_color) #node
    print(nx_edge_colormap)
    print(stop)
    plot_cluster_signaling_network(X,
        labels = labels,
        filename = filename,
        node_size = nx_node_size,
        node_colormap = nx_node_cmap,
        node_cluster_colormap = nx_node_cluster_cmap,
        node_pos = node_pos,
        edge_width_lb_quantile = nx_edge_width_lb_quantile,
        edge_width_ub_quantile = nx_edge_width_ub_quantile,
        edge_width_min = nx_edge_width_min,
        edge_width_max = nx_edge_width_max,
        edge_color = nx_edge_color,
        edge_colormap = nx_edge_colormap,
        background_pos = background_pos,
        background_ndcolor = nx_bg_color,
        background_ndsize = nx_bg_ndsize
    )


    legend_elements = []
    if nx_node_cluster_cmap is None:
        cluster_cmap = get_cmap_qualitative(nx_node_cmap)
        for i in range(len(labels)):
            legend_elements.append(Line2D([0],[0], marker='o',color='w', markerfacecolor=cluster_cmap[i], label=labels[i], markersize=10))
    elif not nx_node_cluster_cmap is None:
        for i in range(len(labels)):
            legend_elements.append(Line2D([0],[0], marker='o',color='w', markerfacecolor=nx_node_cluster_cmap[labels[i]], label=labels[i], markersize=10))
    
    fig, ax = plt.subplots()
    tmp_filename,tmp_type = filename.split('.')
    ax.legend(handles=legend_elements, loc='center')
    ax.axis('off')
    fig.savefig(tmp_filename+"_cluster_legend."+tmp_type, bbox_inches='tight')


# # ############################# Downstream analysis: identify DEG
# adata = sc.datasets.visium_sge(sample_id='V1_Mouse_Brain_Sagittal_Posterior')

# adata.var_names_make_unique()
# adata.raw = adata
# sc.pp.normalize_total(adata, inplace=True)
# sc.pp.log1p(adata)

# sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
# adata = adata[:, adata.var.highly_variable]
# sc.tl.pca(adata, svd_solver='arpack')
# sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
# # sc.tl.umap(adata)
# sc.tl.leiden(adata, resolution=0.4)
# sc.pl.spatial(adata, color='leiden')

# adata_dis500 = sc.read_h5ad("./adata.h5ad") #3355 × 32285
# adata_dis500.obs['leiden'] = adata.obs['leiden']

# 1. cluster cell communication matrix into cell type communication matrix
cluster_CCI = cluster_communication(database_name='cellchat', pathway_name='PSAP', clustering='leiden',
    n_permutations=100)
print(cluster_CCI)

# 2.plot cluster communication
plot_cluster_communication_network(cluster_CCI, uns_names=['commot_cluster-leiden-cellchat-PSAP'],
     nx_node_pos=None, nx_bg_pos=False, p_value_cutoff = 5e-2, filename='PSAP_cluster.pdf', nx_node_cmap='Light24')


