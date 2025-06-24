#!/usr/bin/env python
"""
TODO:
# Author: 
# Created Time : 

# File Name: 
# Description: 

"""


import numpy as np
import scipy.sparse as sp
import random
import inspect
# try:
#     import tensorflow as tf
# except ImportError:
#     raise ImportError('DeepLinc requires TensorFlow. Please follow instructions'
#                       ' at https://www.tensorflow.org/install/ to install'
#                       ' it.')


# =============== Data processing ===============
# ===============================================

import os
os.environ['PYTHONHASHSEED'] = '0'

import matplotlib
havedisplay = "DISPLAY" in os.environ
if havedisplay:  #if you have a display use a plotting backend
    matplotlib.use('TkAgg')
else:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import seaborn as sns
import networkx as nx

import pandas as pd 

import numpy as np
import scipy.sparse as sp
import random
import math

def sparse2tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def plot_histogram(data, xlabel, ylabel, filename, ifhist=True, ifrug=False, ifkde=False, ifxlog=False, ifylog=False, figsize=(15,10), color="cornflowerblue"):
    figure, ax = plt.subplots(figsize=figsize, dpi=100)
    sns.distplot(data, hist=ifhist, rug=ifrug, kde=ifkde, color=color)
    if ifxlog:
        plt.xscale("log")
    if ifylog:
        plt.yscale("log")  #plt.yscale("log",basey=10), where basex or basey are the bases of log

    plt.tick_params(labelsize=30)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]

    font1 = {'family':'Arial','weight':'normal','size':30,}
    plt.xlabel(xlabel, font1)
    plt.ylabel(ylabel, font1)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    plt.savefig(filename+'.png')
    plt.close()


def write_csv_matrix(matrix, filename, ifindex=False, ifheader=True, rownames=None, colnames=None, transpose=False):
    if transpose:
        matrix = matrix.T
        rownames, colnames = colnames, rownames
        ifindex, ifheader = ifheader, ifindex

    pd.DataFrame(matrix, index=rownames, columns=colnames).to_csv(filename+'.csv', index=ifindex, header=ifheader)

def connection_number_between_groups(adj, cell_type_label):
    single_cell_VS_cell_type = np.zeros((cell_type_label.shape[0], len(np.unique(cell_type_label))), dtype=int)
    for i1 in range(0, adj.shape[0]):
        single_cell_adjacency = adj[i1, :]
        single_cell_adjacency_index = np.where(single_cell_adjacency == 1)
        single_cell_adjacency_cell_type = cell_type_label[single_cell_adjacency_index[0]]
        single_cell_adjacency_cell_type_unique = np.sort(np.unique(single_cell_adjacency_cell_type))
        for i2 in single_cell_adjacency_cell_type_unique:
            single_cell_VS_cell_type[i1,i2] = list(single_cell_adjacency_cell_type).count(i2)

    cell_type_VS_cell_type = np.zeros((len(np.unique(cell_type_label)), len(np.unique(cell_type_label))), dtype=float)
    single_cell_VS_cell_type_usedforplot = {}
    for i3 in range(0,len(np.unique(cell_type_label))):
        cell_type_VS_cell_type[i3,:] = single_cell_VS_cell_type[np.where(cell_type_label == i3)[0],:].sum(axis=0)
        single_cell_VS_cell_type_usedforplot[i3] = single_cell_VS_cell_type[np.where(cell_type_label == i3)[0],:] #8个key-value，key是细胞类型ID，value是某种类型细胞的single cell_VS_cell type的邻接数矩阵

    return cell_type_VS_cell_type, single_cell_VS_cell_type_usedforplot

def generate_adj_new_long_edges(dist_matrix, new_edges, all_new_edges_dist, cutoff_distance):
    selected_new_long_edges = new_edges[all_new_edges_dist >= cutoff_distance]
    mask = np.ones(selected_new_long_edges.shape[0])
    adj_new_long_edges = sp.csr_matrix((mask, (selected_new_long_edges[:, 0], selected_new_long_edges[:, 1])), shape=dist_matrix.shape)
    adj_new_long_edges = adj_new_long_edges + adj_new_long_edges.T
    return adj_new_long_edges


def edges_enrichment_evaluation(adj, cell_type_label, cell_type_name, N=2000, edge_type='all edges', **kwargs):
    if kwargs:
        dist_matrix = kwargs['dist_matrix']
        cutoff_distance = kwargs['cutoff_distance']

    cell_type_ID_name = {}
    cell_type_ID_number = {}
    for i in range(0, len(np.unique(cell_type_label))):
        cell_type_ID_name[str(i)] = str(i)
        cell_type_ID_number[i] = np.where(cell_type_label==i)[0].shape[0]

    cell_type_VS_cell_type_ID_number = np.zeros((len(np.unique(cell_type_label)),len(np.unique(cell_type_label))))
    for i in range(0, len(np.unique(cell_type_label))):
        for j in range(0, len(np.unique(cell_type_label))):
            cell_type_VS_cell_type_ID_number[i,j] = cell_type_ID_number[i] * cell_type_ID_number[j] #几何平均

    cell_type_VS_cell_type_shuffle_alltimes_1 = {}
    cell_type_VS_cell_type_shuffle_alltimes_2 = {}
    for i in cell_type_ID_name:
        for j in cell_type_ID_name:
            cell_type_VS_cell_type_shuffle_alltimes_1[i + '-' + j] = []
            cell_type_VS_cell_type_shuffle_alltimes_2[i + '-' + j] = []

    cell_type_VS_cell_type_true, _ = connection_number_between_groups(adj, cell_type_label)
    if edge_type == 'all edges':
        cell_type_VS_cell_type_true = cell_type_VS_cell_type_true/cell_type_VS_cell_type_ID_number

    def merge(test_onetime, test_alltime):
        for i in range(0, test_onetime.shape[0]):
            for j in range(0, test_onetime.shape[0]):
                thelist = test_alltime['%s-%s'%(i,j)]
                thelist.append(test_onetime[i,j])

    if edge_type == 'all edges':
        cell_type_shuffle = cell_type_label
    if edge_type == 'all edges':
        print('shuffle N:',N)
        for num in range(0, N):
            random.shuffle(cell_type_shuffle)
            cell_type_VS_cell_type_shuffle_onetime, _ = connection_number_between_groups(adj, cell_type_shuffle)
            cell_type_VS_cell_type_shuffle_onetime = cell_type_VS_cell_type_shuffle_onetime/cell_type_VS_cell_type_ID_number
            if num+1 <= N/2:
                merge(cell_type_VS_cell_type_shuffle_onetime, cell_type_VS_cell_type_shuffle_alltimes_1)
            else:
                merge(cell_type_VS_cell_type_shuffle_onetime, cell_type_VS_cell_type_shuffle_alltimes_2)
            if num+1%100 == 0:
                print('%s times of permutations have completed calculating ...'%num+1)

    elif edge_type == 'long edges':
        for num in range(0, N):
            adj_shuffle = randAdj_long_edges(dist_matrix, cutoff_distance, int(cell_type_VS_cell_type_true.sum()/2))
            cell_type_VS_cell_type_shuffle_onetime, _ = connection_number_between_groups(adj_shuffle, cell_type_label)
            if num+1 <= N/2:
                merge(cell_type_VS_cell_type_shuffle_onetime, cell_type_VS_cell_type_shuffle_alltimes_1)
            else:
                merge(cell_type_VS_cell_type_shuffle_onetime, cell_type_VS_cell_type_shuffle_alltimes_2)
            if num+1%100 == 0:
                print('%s times of permutations have completed calculating ...'%num+1)

    #计算右侧P value，即假定两类细胞之间是enrichment/interaction的，计算P value
    cell_type_VS_cell_type_enrichment_P = np.zeros((len(np.unique(cell_type_label)), len(np.unique(cell_type_label))), dtype=float)
    for i in range(0,len(np.unique(cell_type_label))):
        for j in range(0,len(np.unique(cell_type_label))):
            P_tmp = len(np.where(np.array(cell_type_VS_cell_type_shuffle_alltimes_1['%s-%s'%(i,j)]) >= cell_type_VS_cell_type_true[i,j])[0]) / (N/2)
            cell_type_VS_cell_type_enrichment_P[i,j] =  P_tmp

    #计算左侧P value，即假定两类细胞之间是depletion/avoidance的，计算P value（用负值与enrichment区分）
    cell_type_VS_cell_type_depletion_P = np.zeros((len(np.unique(cell_type_label)), len(np.unique(cell_type_label))), dtype=float)
    for i in range(0,len(np.unique(cell_type_label))):
        for j in range(0,len(np.unique(cell_type_label))):
            P_tmp = len(np.where(np.array(cell_type_VS_cell_type_shuffle_alltimes_2['%s-%s'%(i,j)]) <= cell_type_VS_cell_type_true[i,j])[0]) / (N/2)
            cell_type_VS_cell_type_depletion_P[i,j] = P_tmp

    #合并在一起，用于画热度图
    cell_type_VS_cell_type_merge_P = np.zeros((len(np.unique(cell_type_label)), len(np.unique(cell_type_label))), dtype=float)
    P_enrichment = []
    P_depletion = []
    for i in range(0,len(np.unique(cell_type_label))):
        for j in range(0,len(np.unique(cell_type_label))):
            if cell_type_VS_cell_type_enrichment_P[i,j] == 0.5:
                if cell_type_VS_cell_type_depletion_P[i,j] == 0:
                    cell_type_VS_cell_type_merge_P[i,j] = -3
                elif cell_type_VS_cell_type_depletion_P[i,j] == 1:
                    cell_type_VS_cell_type_merge_P[i,j] = +3
                elif cell_type_VS_cell_type_depletion_P[i,j] <= 0.5 and cell_type_VS_cell_type_depletion_P[i,j] > 0:
                    cell_type_VS_cell_type_merge_P[i,j] = - (-math.log10(cell_type_VS_cell_type_depletion_P[i,j])) #负值表示depletion/avoidance
                    P_depletion.append(- (-math.log10(cell_type_VS_cell_type_depletion_P[i,j])))
                elif cell_type_VS_cell_type_depletion_P[i,j] > 0.5 and cell_type_VS_cell_type_depletion_P[i,j] < 1:
                    cell_type_VS_cell_type_merge_P[i,j] = -math.log10(1-cell_type_VS_cell_type_depletion_P[i,j]) #正值表示enrichment/interaction
                    P_enrichment.append(-math.log10(1-cell_type_VS_cell_type_depletion_P[i,j]))
            elif cell_type_VS_cell_type_enrichment_P[i,j] == 0:
                cell_type_VS_cell_type_merge_P[i,j] = +3
            elif cell_type_VS_cell_type_enrichment_P[i,j] == 1:
                cell_type_VS_cell_type_merge_P[i,j] = -3
            elif cell_type_VS_cell_type_enrichment_P[i,j] < 0.5 and cell_type_VS_cell_type_enrichment_P[i,j] > 0:
                if cell_type_VS_cell_type_depletion_P[i,j] == 1:
                    cell_type_VS_cell_type_merge_P[i,j] = +3
                elif cell_type_VS_cell_type_depletion_P[i,j] > 0.5 and cell_type_VS_cell_type_depletion_P[i,j] < 1:
                    cell_type_VS_cell_type_merge_P[i,j] = -math.log10(cell_type_VS_cell_type_enrichment_P[i,j])
                    P_enrichment.append(-math.log10(cell_type_VS_cell_type_enrichment_P[i,j]))
                elif cell_type_VS_cell_type_enrichment_P[i,j] <= cell_type_VS_cell_type_depletion_P[i,j] and cell_type_VS_cell_type_depletion_P[i,j] < 0.5:
                    cell_type_VS_cell_type_merge_P[i,j] = -math.log10(cell_type_VS_cell_type_enrichment_P[i,j])
                    P_enrichment.append(-math.log10(cell_type_VS_cell_type_enrichment_P[i,j]))
                elif cell_type_VS_cell_type_depletion_P[i,j] < cell_type_VS_cell_type_enrichment_P[i,j] and cell_type_VS_cell_type_depletion_P[i,j] > 0:
                    cell_type_VS_cell_type_merge_P[i,j] = - (-math.log10(cell_type_VS_cell_type_depletion_P[i,j]))
                    P_depletion.append(- (-math.log10(cell_type_VS_cell_type_depletion_P[i,j])))
                elif cell_type_VS_cell_type_depletion_P[i,j] == 0:
                    cell_type_VS_cell_type_merge_P[i,j] = -3
            elif cell_type_VS_cell_type_enrichment_P[i,j] > 0.5 and cell_type_VS_cell_type_enrichment_P[i,j] < 1:
                if cell_type_VS_cell_type_depletion_P[i,j] == 0:
                    cell_type_VS_cell_type_merge_P[i,j] = -3
                elif cell_type_VS_cell_type_depletion_P[i,j] < 0.5 and cell_type_VS_cell_type_depletion_P[i,j] > 0:
                    cell_type_VS_cell_type_merge_P[i,j] = - (-math.log10(cell_type_VS_cell_type_depletion_P[i,j]))
                    P_depletion.append(- (-math.log10(cell_type_VS_cell_type_depletion_P[i,j])))
                elif cell_type_VS_cell_type_depletion_P[i,j] <= cell_type_VS_cell_type_enrichment_P[i,j] and cell_type_VS_cell_type_depletion_P[i,j] > 0.5:
                    cell_type_VS_cell_type_merge_P[i,j] = - (-math.log10(1-cell_type_VS_cell_type_enrichment_P[i,j]))
                    P_depletion.append(- (-math.log10(1-cell_type_VS_cell_type_enrichment_P[i,j])))
                elif cell_type_VS_cell_type_depletion_P[i,j] > cell_type_VS_cell_type_enrichment_P[i,j] and cell_type_VS_cell_type_depletion_P[i,j] < 1:
                    cell_type_VS_cell_type_merge_P[i,j] = -math.log10(1-cell_type_VS_cell_type_depletion_P[i,j])
                    P_enrichment.append(-math.log10(1-cell_type_VS_cell_type_depletion_P[i,j]))
                elif cell_type_VS_cell_type_depletion_P[i,j] == 1:
                    cell_type_VS_cell_type_merge_P[i,j] = +3

    tmp1 = [x1 for y1 in cell_type_VS_cell_type_true for x1 in y1]
    tmp2 = [x2 for y2 in cell_type_VS_cell_type_merge_P for x2 in y2]
    tmp3 = [x3 for x3 in cell_type_name for y3 in range(len(cell_type_name))]
    tmp4 = [x4 for y4 in range(len(cell_type_name)) for x4 in cell_type_name]
    test_result = np.array([tmp3,tmp4,tmp1,tmp2]).T

    return test_result, cell_type_VS_cell_type_merge_P, cell_type_VS_cell_type_enrichment_P, cell_type_VS_cell_type_depletion_P


def randAdj_long_edges(dist_matrix, cutoff_distance, edge_number):  #edge_number指最终生成的随机邻接矩阵中所有1值的数目
    adj_all_long_edges = np.int64(dist_matrix >= cutoff_distance)
    adj_all_long_edges = sp.csr_matrix(adj_all_long_edges)
    # edges = sparse_to_tuple(sp.triu(adj_all_long_edges))[0]
    edges = sparse2tuple(sp.triu(adj_all_long_edges))[0]
    selected_long_edges = edges[random.sample(range(0, edges.shape[0]), edge_number),:]
    data = np.ones(selected_long_edges.shape[0])
    adj_selected = sp.csr_matrix((data, (selected_long_edges[:, 0], selected_long_edges[:, 1])), shape=adj_all_long_edges.shape)
    adj_selected = adj_selected + adj_selected.T
    return adj_selected.toarray()

def ranked_partial(adj_orig, adj_rec, coord, size):  #size是list，[3,5]代表把总图切成宽3份(x)、高5份(y)的子图
    x_gap = (coord[:,0].max()-coord[:,0].min())/size[0]
    y_gap = (coord[:,1].max()-coord[:,1].min())/size[1]
    x_point = np.arange(coord[:,0].min(), coord[:,0].max(), x_gap).tolist()
    if coord[:,0].max() not in x_point:
        x_point += [coord[:,0].max()]
    y_point = np.arange(coord[:,1].min(), coord[:,1].max(), y_gap).tolist()
    if coord[:,1].max() not in y_point:
        y_point += [coord[:,1].max()]

    x_interval = [[x_point[i],x_point[i+1]] for i in range(len(x_point)) if i!=len(x_point)-1]
    y_interval = [[y_point[i],y_point[i+1]] for i in range(len(y_point)) if i!=len(y_point)-1]

    id_part = {}
    subregion_mark = []
    for i in x_interval:
        for j in y_interval:
            id_list = np.where((coord[:,0]>=i[0]) & (coord[:,0]<i[1]) & (coord[:,1]>=j[0]) & (coord[:,1]<j[1]))[0].tolist()  #左开右闭，上开下闭
            adj_orig_tmp = adj_orig[id_list,:][:,id_list]
            adj_rec_tmp = adj_rec[id_list,:][:,id_list]
            if adj_orig_tmp.shape[0]*adj_orig_tmp.shape[1] == 0:
                break
            else:
                diff = np.where((adj_orig_tmp-adj_rec_tmp)!=0)[0].shape[0] / (adj_orig_tmp.shape[0]*adj_orig_tmp.shape[1])
                id_part[diff] = id_list
                subregion_mark.append([i,j])

    return sorted(id_part.items(), key=lambda item:item[0], reverse=True), subregion_mark

def adjacency_visualization(cell_type, coord, adj, filename):
    colors = ['beige', 'royalblue', 'maroon', 'olive', 'tomato', 'mediumpurple', 'paleturquoise', 'brown', 
              'firebrick', 'mediumturquoise', 'lightsalmon', 'orchid', 'dimgray', 'dodgerblue', 'mistyrose', 
              'sienna', 'tan', 'teal', 'chartreuse']

    X = np.hstack((cell_type, coord))

    #获取节点信息
    class_cellid = []
    for num in range(55):
        class_cellid.append(list(X[X[:,1]==num, 0].astype('int')-1))

    #获取坐标信息
    class_coord = []
    for num in range(55):
        class_coord.append(X[X[:,1]==num, 2:].tolist())

    #获取节点与坐标之间的映射关系，分cell type存储用于画节点
    pos_usedfor_nodes = []
    for num in range(55):
        pos_usedfor_nodes.append(dict(zip(class_cellid[num],class_coord[num])))

    #获取节点与坐标之间的映射关系，不分cell type存储用于画边
    pos_usedfor_edges = {}
    pos_usedfor_edges = pos_usedfor_nodes[0].copy()
    for num in range(1,55):
        pos_usedfor_edges.update(pos_usedfor_nodes[num])

    edges_tmp1 = np.where(adj == 1)
    edges1 = []
    edges_cluster = [edges1]
    edges_tmp_cluster = [edges_tmp1]

    for z2,z3 in enumerate(edges_tmp_cluster):
        edges_num = z3[0].shape[0]
        for z4 in range(0, edges_num):
            edges_cluster[z2].append((z3[0][z4],z3[1][z4]))

    # 循环画三种方式计算的连接图
    for z5 in edges_cluster:
        ax = plt.axes([0.042, 0.055, 0.9, 0.9])#[xmin,ymin,xmax,ymax]
        ax.set_xlim(min(X[:,2])-15,max(X[:,2])+15)
        ax.set_ylim(min(X[:,3])-15,max(X[:,3])+15)
        ax.xaxis.set_major_locator(plt.MultipleLocator(400.0))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(200.0))#设置x从坐标间隔
        ax.yaxis.set_major_locator(plt.MultipleLocator(400.0))#设置y主坐标间隔
        ax.yaxis.set_minor_locator(plt.MultipleLocator(200.0))#设置y从坐标间隔
        ax.grid(which='major', axis='x', linewidth=0.3, linestyle='-', color='0.3')#由每个x主坐标出发对x主坐标画垂直于x轴的线段 
        ax.grid(which='minor', axis='x', linewidth=0.1, linestyle='-', color='0.1')
        ax.grid(which='major', axis='y', linewidth=0.3, linestyle='-', color='0.3') 
        ax.grid(which='minor', axis='y', linewidth=0.1, linestyle='-', color='0.1') 
        ax.set_xticklabels([i for i in range(int(min(X[:,2])-15),int(max(X[:,2])+15),400)])
        ax.set_yticklabels([i for i in range(int(min(X[:,3])-15),int(max(X[:,3])+15),400)])
        G = nx.Graph()
        for i2 in pos_usedfor_nodes:
            cellid_oneclass = list(i2.keys())
            nx.draw_networkx_nodes(G, i2, cellid_oneclass, node_size=150, node_color=colors[pos_usedfor_nodes.index(i2)])  #HDST_cancer和seqFISH的node_size是150，MERFISH是145
        nx.draw_networkx_edges(G, pos_usedfor_edges, z5, width=0.8)
        nx.draw(G)

        plt.savefig(filename + '.png')
