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

def plot_cluster_communication_network(X,
    labels,
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

    node_pos = None
    background_pos = None

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


if __name__ == '__main__':
    dataname = 'HDST_cancer' #'HDST_ob','HDST_cancer','MERFISH','seqFISH'

    df_communication = pd.read_csv('results/enrichment/'+dataname+'/all_edges_enrichment_evaluation_iter4.csv')
    print(df_communication)

    df_celltype = pd.read_csv('data/'+dataname+'/cell_type.csv')
    cell_type = df_celltype['Cell_class_name'].values.tolist()
    cell_type = list(set(cell_type))
    cell_type = sorted(cell_type)
    if dataname == 'HDST_ob' or dataname == 'HDST_cancer':
        cell_type.remove('ambiguous')
    print(cell_type)

    cell_type_map = {}
    for i in range(len(cell_type)):
        cell_type_map[cell_type[i]] = i

    significance_matrix = np.zeros((len(cell_type),len(cell_type)))
    connectivity_matrix = np.zeros((len(cell_type),len(cell_type)))
    for i in range(len(df_communication)):
        cell_type_A =  df_communication.iloc[i]['cell type A']
        cell_type_B =  df_communication.iloc[i]['cell type B']
        connectivity = df_communication.iloc[i]['average_connectivity']
        significance =  df_communication.iloc[i]['significance']
        print(cell_type_A,cell_type_B,connectivity,significance)
        if cell_type_A in cell_type and cell_type_B in cell_type:
            cell_type_index_A = cell_type_map[cell_type_A]
            cell_type_index_B = cell_type_map[cell_type_B]
            print(cell_type_index_A,cell_type_index_B)
            significance_matrix[cell_type_index_A,cell_type_index_B] = significance
            connectivity_matrix[cell_type_index_A,cell_type_index_B] = connectivity
    print(significance_matrix)
    print(connectivity_matrix)

    file_path = 'plot/'+dataname+'.pdf'
    plot_cluster_communication_network(X=significance_matrix,labels=cell_type,nx_node_pos=None, nx_bg_pos=False, p_value_cutoff = 5e-2,
                                       filename=file_path,nx_node_cmap='Light24')
