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

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from utils._clustering import leiden_clustering
import seaborn as sns
import plotly
from typing import Optional, Union

def communication_deg_detection(
    adata: anndata.AnnData,
    n_var_genes: int = None,
    var_genes = None,
    database_name: str = None,
    pathway_name: str = None,
    summary: str = 'receiver',
    lr_pair: tuple = ('total','total'),
    nknots: int = 6,
    n_deg_genes: int = None,
    n_points: int = 50,
    deg_pvalue_cutoff: float = 0.05,
):
    # prepare input adata for R
    adata_deg = anndata.AnnData(
        X = adata.X,
        var = pd.DataFrame(index=list(adata.var_names)),
        obs = pd.DataFrame(index=list(adata.obs_names)))
    adata_deg_var = adata_deg.copy()
    sc.pp.filter_genes(adata_deg_var, min_cells=3)
    sc.pp.filter_genes(adata_deg, min_cells=3)
    sc.pp.normalize_total(adata_deg_var, target_sum=1e4)
    sc.pp.log1p(adata_deg_var)
    if n_var_genes is None:
        sc.pp.highly_variable_genes(adata_deg_var, min_mean=0.0125, max_mean=3, min_disp=0.5)
    elif not n_var_genes is None:
        sc.pp.highly_variable_genes(adata_deg_var, n_top_genes=n_var_genes)

    if var_genes is None:
        adata_deg = adata_deg[:, adata_deg_var.var.highly_variable]
    else:
        adata_deg = adata_deg[:, var_genes]
    del adata_deg_var
    # print(adata_deg) # n_obs × n_vars = 3355 × 2676, 预处理，并计算high variable的基因
    # print(adata_deg.X)


    df_LR_activity = pd.read_csv('results/HBC_LR_delta_roc.csv')
    LR_rank = df_LR_activity['LR'].values
    summary_name = LR_rank[0]
    df_LR_activity = pd.read_csv('results/matrix/receiver_iter_4.csv')
    comm_sum = df_LR_activity[summary_name].values.reshape(-1,1)
    cell_weight = np.ones_like(comm_sum).reshape(-1,1)

    # "pseudoTime" = comm_sum
    # "cellWeight" = cell_weight
    # "counts" = X,adata_deg.X

    return adata_deg, comm_sum, cell_weight

def communication_deg_clustering(
    df_deg: pd.DataFrame,
    df_yhat: pd.DataFrame,
    deg_clustering_npc: int = 10,
    deg_clustering_knn: int = 5,
    deg_clustering_res: float = 1.0,
    n_deg_genes: int = 200,
    p_value_cutoff: float = 0.05
):
    df_deg = df_deg[df_deg['pvalue'] <= p_value_cutoff]
    n_deg_genes = min(n_deg_genes, df_deg.shape[0])
    idx = np.argsort(-df_deg['waldStat'])
    df_deg = df_deg.iloc[idx[:n_deg_genes]]
    yhat_scaled = df_yhat.loc[df_deg.index]
    x_pca = PCA(n_components=deg_clustering_npc, svd_solver='full').fit_transform(yhat_scaled.values)
    cluster_labels = leiden_clustering(x_pca, k=deg_clustering_knn, resolution=deg_clustering_res, input='embedding')

    data_tmp = np.concatenate((df_deg.values, cluster_labels.reshape(-1,1)),axis=1)
    df_metadata = pd.DataFrame(data=data_tmp, index=df_deg.index,
        columns=['waldStat','df','pvalue','cluster'] )
    return df_metadata, yhat_scaled

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

def plot_communication_dependent_genes(
    df_deg: pd.DataFrame,
    df_yhat: pd.DataFrame,
    show_gene_names: bool = True,
    top_ngene_per_cluster: int = -1,
    colormap: str = 'magma',
    cluster_colormap: str = 'Plotly',
    font_scale: float = 1.4,
    filename = None,
    return_genes = False
):
    cmap = get_cmap_qualitative(cluster_colormap)
    wald_stats = df_deg['waldStat'].values
    pvalue = df_deg['pvalue'].values
    labels = np.array( df_deg['cluster'].values, int)
    nlabel = np.max(labels)+1
    yhat_mat = df_yhat.values
    peak_locs = []
    for i in range(nlabel):
        tmp_idx = np.where(labels==i)[0]
        tmp_y = yhat_mat[tmp_idx,:]
        peak_locs.append(np.mean(np.argmax(tmp_y, axis=1)))
    cluster_order = np.argsort(peak_locs)
    idx = np.array([])
    row_colors = []
    for i in cluster_order:
        tmp_idx = np.where(labels==i)[0]
        tmp_order = np.argsort(-wald_stats[tmp_idx])
        if top_ngene_per_cluster >= 0:
            top_ngene = min(len(tmp_idx), top_ngene_per_cluster)
        else:
            top_ngene = len(tmp_idx)
        idx = np.concatenate((idx, tmp_idx[tmp_order][:top_ngene]))
        for j in range(top_ngene):
            row_colors.append(cmap[i % len(cmap)])

    sns.set(font_scale=font_scale)
    g = sns.clustermap(df_yhat.iloc[idx], 
        row_cluster=False, 
        col_cluster=False, 
        row_colors=row_colors,
        cmap = colormap,
        xticklabels = False,
        yticklabels = show_gene_names,
        linewidths=0)
    g.cax.set_position([.1, .2, .03, .45])
    plt.savefig(filename, dpi=300)

    if return_genes:
        return list( df_deg.iloc[idx].index )

def treebased_score_multifeature(
    X,
    y,
    cov,
    method='rf',
    n_trees=100,
    n_repeat=10,
    max_depth=5,
    max_features='sqrt',
    learning_rate=0.1,
    subsample=1.0
):
    random_seeds = np.random.randint(100000, size=n_repeat)
    X_train = np.concatenate((X, cov), axis=1)
    ranks = np.empty([X.shape[1], n_repeat], float)
    for i in range(n_repeat):
        if method == 'rf':
            model = RandomForestRegressor(n_estimators=n_trees, 
                max_depth=max_depth,
                max_features=max_features,
                random_state=random_seeds[i],
                n_jobs=-1
            )
        elif method == 'gbt':
            model = GradientBoostingRegressor(n_estimators=n_trees,
                max_depth=max_depth,
                max_features=max_features,
                learning_rate=learning_rate,
                subsample=subsample,
                random_state=random_seeds[i]
            )
        model.fit(X_train, y)
        importance = model.feature_importances_
        sorted_idx = np.argsort(-importance)
        for j in range(X.shape[1]):
            rank = np.where(sorted_idx==j)[0][0]
            ranks[j,i] = float(X_train.shape[1] - rank - 1) / float(X_train.shape[1]-1)
    return np.mean(ranks, axis=1)

def communication_impact(
    adata: anndata.AnnData,
    database_name: str = None,
    pathway_name: str = None,
    pathway_sum_only: bool = False,
    heteromeric_delimiter: str = '_',
    normalize: bool = False,
    method: str = None,
    corr_method: str = "spearman",
    tree_method: str = "rf",
    tree_ntrees: int = 100,
    tree_repeat: int = 100,
    tree_max_depth: int = 5,
    tree_max_features: str = 'sqrt',
    tree_learning_rate: float = 0.1,
    tree_subsample: float = 1.0,
    tree_combined: bool = False,
    ds_genes: list = None,
    bg_genes: Union[list, int] = 100
):
    # Get a list of background genes using most 
    # variable genes if only given a number.
    adata_bg = adata.raw.to_adata()
    adata_all = adata.raw.to_adata()
    if normalize:
        sc.pp.normalize_total(adata_bg, inplace=True)
        sc.pp.log1p(adata_bg)
    if np.isscalar(bg_genes):
        ng_bg = int(bg_genes)
        sc.pp.highly_variable_genes(adata_bg, n_top_genes=ng_bg)
        adata_bg = adata_bg[:,adata_bg.var.highly_variable]
    else:
        adata_bg = adata_bg[:,bg_genes]
    # Prepare downstream or upstream genes
    ncell = adata.shape[0]
    col_names = []
    Ds_exps = []
    Ds_exp_total = np.zeros([ncell], float)
    for i in range(len(ds_genes)):
        Ds_exp = np.array(adata_all[:,ds_genes[i]].X.toarray()).reshape(-1)
        Ds_exps.append(Ds_exp)
        col_names.append(ds_genes[i])
        Ds_exp_total += Ds_exp
    Ds_exps.append(Ds_exp_total); col_names.append('average')
    # Impact analysis
    df_ligrec = adata.uns['commot-'+database_name+'-info']['df_ligrec']
    available_pathways = []
    for i in range(df_ligrec.shape[0]):
        _, _, tmp_pathway = df_ligrec.iloc[i,:]
        if not tmp_pathway in available_pathways:
            available_pathways.append(tmp_pathway)
    pathway_genes = [[] for i in range(len(available_pathways))]
    all_lr_genes = []
    for i in range(df_ligrec.shape[0]):
        tmp_lig, tmp_rec, tmp_pathway = df_ligrec.iloc[i,:]
        idx = available_pathways.index(tmp_pathway)
        tmp_ligs = tmp_lig.split(heteromeric_delimiter)
        tmp_recs = tmp_rec.split(heteromeric_delimiter)
        for lig in tmp_ligs:
            if not lig in pathway_genes[idx]:
                pathway_genes[idx].append(lig)
            if not lig in all_lr_genes:
                all_lr_genes.append(lig)
        for rec in tmp_recs:
            if not rec in pathway_genes[idx]:
                pathway_genes[idx].append(rec)
            if not rec in all_lr_genes:
                all_lr_genes.append(rec)
    bg_genes = list( adata_bg.var_names )

    sum_names = []
    exclude_lr_genes_list = []
    if pathway_name is None and not pathway_sum_only:
        for i in range(df_ligrec.shape[0]):
            tmp_lig, tmp_rec, _ = df_ligrec.iloc[i,:]
            sum_names.append("%s-%s" % (tmp_lig, tmp_rec))
            exclude_lr_genes_list.append(set(tmp_lig.split(heteromeric_delimiter)).union(set(tmp_rec.split(heteromeric_delimiter))))
        for tmp_pathway in available_pathways:
            sum_names.append(tmp_pathway)
            exclude_lr_genes_list.append(set(pathway_genes[available_pathways.index(tmp_pathway)]))
        sum_names.append('total-total')
        exclude_lr_genes_list.append(set(all_lr_genes))
    elif not pathway_name is None and not pathway_sum_only:
        for i in range(df_ligrec.shape[0]):
            tmp_lig, tmp_rec, tmp_pathway = df_ligrec.iloc[i,:]
            if tmp_pathway == pathway_name:
                sum_names.append("%s-%s" % (tmp_lig, tmp_rec))
                exclude_lr_genes_list.append(set(tmp_lig.split(heteromeric_delimiter)).union(set(tmp_rec.split(heteromeric_delimiter))))
        sum_names.append(pathway_name)
        exclude_lr_genes_list.append(set(pathway_genes[available_pathways.index(pathway_name)]))

    elif pathway_sum_only:
        sum_names = available_pathways
        for i in range(len(available_pathways)):
            exclude_lr_genes_list.append(set(pathway_genes[i]))

    nrows = 2 * len(sum_names)

    ncols = len(ds_genes) + 1
    impact_mat = np.empty([nrows, ncols], float)
    
    row_names_sender = []; row_names_receiver = []
    exclude_lr_genes_list = []
    for i in range(len(sum_names)):
        row_names_sender.append('s-%s' % sum_names[i])
        row_names_receiver.append('r-%s' % sum_names[i])
    row_names = row_names_sender + row_names_receiver
    exclude_lr_genes_list = exclude_lr_genes_list + exclude_lr_genes_list

    print(nrows, ncols)
    for j in range(ncols):
        print(j)
        if j == ncols-1:
            exclude_ds_genes = set(ds_genes)
        else:
            exclude_ds_genes = set([ds_genes[j]])
        if method == 'treebased_score' and tree_combined:
            exclude_lr_genes = set(all_lr_genes)
            exclude_genes = list(exclude_lr_genes.union(exclude_ds_genes))
            use_genes = list( set(bg_genes) - set(exclude_genes) )
            bg_mat = np.array( adata_bg[:,use_genes].X.toarray() )
            sum_mat = np.concatenate((adata.obsm['commot-'+database_name+'-sum-sender'][row_names_sender].values, \
                adata.obsm['commot-'+database_name+'-sum-receiver'][row_names_receiver].values), axis=1)
            r = treebased_score_multifeature(sum_mat, Ds_exps[j], bg_mat,
                n_trees = tree_ntrees, n_repeat = tree_repeat,
                max_depth = tree_max_depth, max_features = tree_max_features,
                learning_rate = tree_learning_rate, subsample = tree_subsample)
            impact_mat[:,j] = r[:]
        else:
            for i in range(nrows):
                row_name = row_names[i]
                exclude_lr_genes = exclude_lr_genes_list[i]

                exclude_genes = list(exclude_lr_genes.union(exclude_ds_genes))
                use_genes = list( set(bg_genes) - set(exclude_genes) )
                bg_mat = np.array( adata_bg[:,use_genes].X.toarray() )
                if row_name[0] == 's':
                    sum_vec = adata.obsm['commot-'+database_name+'-sum-sender'][row_name].values.reshape(-1,1)
                elif row_name[0] == 'r':
                    sum_vec = adata.obsm['commot-'+database_name+'-sum-receiver'][row_name].values.reshape(-1,1)
                if method == "partial_corr":
                    r,p = partial_corr(sum_vec, Ds_exps[j].reshape(-1,1), bg_mat, method=corr_method)
                elif method == "semipartial_corr":
                    r,p = semipartial_corr(sum_vec, Ds_exps[j].reshape(-1,1), ycov=bg_mat, method=corr_method)
                elif method == "treebased_score":
                    r = treebased_score(sum_vec, Ds_exps[j], bg_mat,
                        n_trees = tree_ntrees, n_repeat = tree_repeat,
                        max_depth = tree_max_depth, max_features = tree_max_features,
                        learning_rate = tree_learning_rate, subsample = tree_subsample)
                impact_mat[i,j] = r
    df_impact = pd.DataFrame(data=impact_mat, index = row_names, columns = col_names)
    return df_impact

def reorder(labels, cofactor_cluster, cofactor_sample, cmap):
    nlabels = np.max(labels) + 1
    cofactor = []
    for i in range(nlabels):
        tmp_idx = np.where(labels==i)[0]
        cofactor.append(cofactor_cluster[tmp_idx].mean())
    cluster_order = np.argsort(cofactor)
    idx = np.array([])
    colors = []
    for i in cluster_order:
        tmp_idx = np.where(labels==i)[0]
        tmp_order = np.argsort(cofactor_sample[tmp_idx])
        idx = np.concatenate((idx, tmp_idx[tmp_order]))
        for j in range(len(tmp_idx)):
            colors.append(cmap[i % len(cmap)])
    return np.array(idx, int), colors

def plot_communication_impact(
    df_impact: pd.DataFrame,
    summary: str = None,
    show_gene_names: str = True,
    show_comm_names: str = True,
    top_ngene: int = -1,
    top_ncomm: int = -1,
    colormap: str = 'rocket',
    font_scale: float = 1.4,
    filename: str = None,
    cluster_knn: str = 5,
    cluster_res: float = 0.5,
    cluster_colormap: str = "Plotly",
    linewidth = 0.0,
    vmin = 0.0,
    vmax = 1.0
):
    """
    Plot communication impact obtained by running the function :func:`commot.tl.communication_impact`.

    .. image:: communication_impact.png
        :width: 300pt

    Parameters
    ----------
    df_impact
        The output from ``tl.communication_impact``.
    summary
        If 'receiver', the received signals are plotted as rows. 
        If 'sender', the sent signals are plotted as rows.
        If None, both are plotted.
    show_gene_names
        Whether to plot gene names as x ticks.
    show_comm_names
        Whether to plot communication names as y ticks.
    top_ngene
        The number of most impacted genes to plot as columns.
        If -1, all genes in ``df_impact`` are plotted.
    top_ncomm
        The number of communications with most impacts to plot as rows.
        If -1, all communications in ``df_impact`` are plotted.
    colormap
        The colormap for the heatmap. Choose from available colormaps from ``seaborn``.
    font_scale
        Font size.
    filename
        Filename for saving the figure. Set the name to end with '.pdf' or 'png'
        to specify format.
    cluster_knn
        Number of nearest neighbors when clustering the rows and columns.
    cluster_res
        The resolution paratemeter when running leiden clustering.
    cluster_colormap
        The qualitative colormap for annotating gene cluster labels.
        Choose from 'Plotly', 'Alphabet', 'Light24', 'Dark24'.

    """
    index_names = list( df_impact.index )
    tmp_idx = []
    if summary == 'receiver':
        for i in range(len(index_names)):
            index_name = index_names[i]
            tmp_n = min(len(index_name), 8)
            if index_name[0] == 'r':
                tmp_idx.append(i)
    elif summary == 'sender':
        for i in range(len(index_names)):
            index_name = index_names[i]
            tmp_n = min(len(index_name), 6)
            if index_name[0] == 's':
                tmp_idx.append(i)
    elif summary is None:
        tmp_idx = [i for i in range(len(index_names))]
    tmp_idx = np.array(tmp_idx, int)
    df_plot = df_impact.iloc[tmp_idx]
    
    mat = df_plot.values
    sum_gene = np.sum(np.abs(mat), axis=0)
    sum_comm = np.sum(np.abs(mat), axis=1)
    if top_ngene == -1:
        top_ngene = mat.shape[1]
    else:
        top_ngene = min(top_ngene, mat.shape[1])
    if top_ncomm == -1:
        top_ncomm = mat.shape[0]
    else:
        top_ncomm = min(top_ncomm, mat.shape[0])
    row_idx = np.argsort(-sum_comm)[:top_ncomm]
    col_idx = np.argsort(-sum_gene)[:top_ngene]

    df_plot = ( df_plot.iloc[row_idx,:] ).iloc[:,col_idx]
    mat = df_plot.values
    cmap = get_cmap_qualitative(cluster_colormap)
    if mat.shape[1] > 10:
        mat_pca = PCA(n_components=np.min([10,mat.shape[1],mat.shape[0]]), svd_solver='full').fit_transform(mat)
    else:
        mat_pca = mat
    D = distance_matrix(mat_pca, mat_pca)
    labels = leiden_clustering(D, k=cluster_knn, resolution=cluster_res)
    row_idx, row_colors = reorder(labels, -np.abs(mat.sum(axis=1)), -np.abs(mat.sum(axis=1)), cmap)
    if mat.shape[0] > 10:
        mat_pca = PCA(n_components=np.min([10,mat.shape[1],mat.shape[0]]), svd_solver='full').fit_transform(mat.T)
    else:
        mat_pca = mat.T
    D = distance_matrix(mat_pca, mat_pca)
    labels = leiden_clustering(D, k=cluster_knn, resolution=cluster_res)
    col_idx, col_colors = reorder(labels, -np.abs(mat.sum(axis=0)), -np.abs(mat.sum(axis=0)), cmap)

    sns.set(font_scale=font_scale)
    g = sns.clustermap( ( df_plot.iloc[row_idx,:] ).iloc[:,col_idx], 
        row_cluster = False, 
        col_cluster  =False, 
        row_colors = row_colors,
        col_colors = col_colors,
        cmap = colormap,
        xticklabels = show_gene_names,
        yticklabels = show_comm_names,
        linewidths = linewidth,
        square = True,
        vmin = vmin,
        vmax = vmax)
    g.cax.set_position([0.01, .2, .03, .45])
    plt.savefig(filename, dpi=300)

# # # ############################# Downstream analysis: identify DEG
# # adata_dis500 = sc.read_h5ad("./adata.h5ad") #3355 × 32285
# # adata = sc.datasets.visium_sge(sample_id='V1_Mouse_Brain_Sagittal_Posterior')
# # adata_dis500.layers['counts'] = adata.X

# # PSAP_communication_matrix = adata_dis500.obsp['commot-cellchat-PSAP']
# # print(PSAP_communication_matrix)
# # print(adata_dis500.obsm['commot-cellchat-sum-receiver']) # receiver matrix, v
# # print(adata_dis500.obsm['commot-cellchat-sum-sender']) # sender matrix, u
# #                     r-Nrg3-Erbb4  r-Igf2-Itga6_Itgb4  r-Igf2-Igf2r  r-Igf2-Igf1r  ...     r-VIP  r-VISFATIN     r-WNT   r-ncWNT
# # AAACAAGTATCTCCCA-1      0.000000            0.660351           0.0      1.191168  ...  0.134266    0.000000  0.000000  0.586901
# # AAACACCAATAACTGC-1      0.269411            0.000000           0.0      0.000000  ...  0.000000    0.000000  1.511297  0.587423
# # AAACAGAGCGACTCCT-1      0.000000            0.000000           0.0      0.000000  ...  0.000000    0.958948  0.000000  0.094876
# # AAACAGCTTTCAGAAG-1      0.000000            0.000000           0.0      0.000000  ...  0.000000    0.000000  0.000000  0.414315
# # AAACAGGGTCTATATT-1      0.766001            0.000000           0.0      0.793427  ...  0.000000    0.000000  0.000000  0.346054
# # ...                          ...                 ...           ...           ...  ...       ...         ...       ...       ...
# # TTGTTCAGTGTGCTAC-1      0.000000            0.000000           0.0      0.000000  ...  0.000000    0.000000  0.000000  0.418423
# # TTGTTGTGTGTCAAGA-1      0.000000            0.000000           0.0      0.467993  ...  0.000000    0.598288  3.369346  0.583973
# # TTGTTTCACATCCAGG-1      0.329417            0.000000           0.0      0.404755  ...  0.000000    0.000000  0.000000  0.603888
# # TTGTTTCATTAGTCTA-1      0.263369            0.000000           0.0      0.591406  ...  0.000000    0.000000  0.000000  0.326903
# # TTGTTTCCATACAACT-1      0.343885            0.000000           0.0      0.473307  ...  0.000000    0.533995  0.000000  0.657530

# # input:
# # 1.gene_expression matrix: adata.layers['counts']
# # 2.gen names: adata.var_names
# # 3.cell names: adata.obs_names
# # 4:recieved info of one specific LR piar/ pathway: comm_sum 

# 1. generate input for R 
# df_deg, df_yhat = ct.tl.communication_deg_detection(adata_dis500,
#     database_name = 'cellchat', pathway_name='PSAP', summary = 'receiver')
adata = sc.read_10x_h5("data/V1_Breast_Cancer_Block_A_Section_1/filtered_feature_bc_matrix.h5") #3355 × 32285
print(adata)
print(stop)

# adata_deg, comm_sum, cell_weight = communication_deg_detection(adata,
#     database_name = 'cellchat', pathway_name='PSAP', summary = 'receiver')
# X = adata_deg.X.todense()
# X = X.T
# genes = adata_deg.var_names.values
# cells = adata_deg.obs_names.values

# df_counts = pd.DataFrame(data=X,columns=cells,index=genes)
# df_counts.to_csv("tradeSeq/input/counts.csv")
# np.savetxt("tradeSeq/input/comm_sum.csv",comm_sum,delimiter=',')
# np.savetxt("tradeSeq/input/cell_weight.csv",cell_weight,delimiter=',')


# 2. read output from R
df_assoRes = pd.read_csv("tradeSeq/output/assoRes.csv",index_col=0)
df_yhat = pd.read_csv("tradeSeq/output/yhatScaled.csv",index_col=0)

df_deg = df_assoRes.rename(columns={'waldStat_1':'waldStat', 'df_1':'df', 'pvalue_1':'pvalue'})
df_deg = df_deg[['waldStat','df','pvalue']]
idx = np.argsort(-df_deg['waldStat'].values)
df_deg = df_deg.iloc[idx]

import pickle
deg_result = {"df_deg": df_deg, "df_yhat": df_yhat}
with open('tradeSeq/deg_LR.pkl', 'wb') as handle:
    pickle.dump(deg_result, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("tradeSeq/deg_LR.pkl", 'rb') as file:
    deg_result = pickle.load(file)

# 3.communication_deg_clustering
df_deg_clus, df_yhat_clus = communication_deg_clustering(df_deg, df_yhat, deg_clustering_res=0.4)
print(df_deg_clus)
print(df_yhat_clus)

# 4.plot plot_communication_dependent_genes
top_de_genes_PSAP = plot_communication_dependent_genes(df_deg_clus, df_yhat_clus, top_ngene_per_cluster=5,
    filename='tradeSeq/heatmap_deg_LR.pdf', font_scale=1.2, return_genes=True)

# # 5.Plot some example signaling DE genes.
# X_sc = adata_dis500.obsm['spatial']
# fig, ax = plt.subplots(1,3, figsize=(15,4))
# colors = adata_dis500.obsm['commot-cellchat-sum-receiver']['r-PSAP'].values
# idx = np.argsort(colors)
# ax[0].scatter(X_sc[idx,0],X_sc[idx,1], c=colors[idx], cmap='coolwarm', s=10)
# colors = adata_dis500[:,'Ctxn1'].X.toarray().flatten()
# idx = np.argsort(colors)
# ax[1].scatter(X_sc[idx,0],X_sc[idx,1], c=colors[idx], cmap='coolwarm', s=10)
# colors = adata_dis500[:,'Gpr37'].X.toarray().flatten()
# idx = np.argsort(colors)
# ax[2].scatter(X_sc[idx,0],X_sc[idx,1], c=colors[idx], cmap='coolwarm', s=10)
# ax[0].set_title('Amount of received signal')
# ax[1].set_title('An example negative DE gene (Ctxn1)')
# ax[2].set_title('An example positive DE gene (Gpr37)')
# # plt.show()

# 6. Further quantify impact of signaling on the DE genes
# df_impact_PSAP = ct.tl.communication_impact(adata_dis500, database_name='cellchat', pathway_name = 'PSAP',\
#     tree_combined = True, method = 'treebased_score', tree_ntrees=100, tree_repeat = 100, tree_method = 'rf', \
#     ds_genes = top_de_genes_PSAP, bg_genes = 500, normalize=True)
df_impact_PSAP = communication_impact(adata_dis500, database_name='cellchat', pathway_name = 'PSAP',\
    tree_combined = True, method = 'treebased_score', tree_ntrees=100, tree_repeat = 100, tree_method = 'rf', \
    ds_genes = top_de_genes_PSAP, bg_genes = 500, normalize=True)

plot_communication_impact(df_impact_PSAP, summary = 'receiver', top_ngene= 30, top_ncomm = 5, colormap='coolwarm',
    font_scale=1.2, linewidth=0, show_gene_names=True, show_comm_names=True, cluster_knn=2,
    filename = 'heatmap_impact_PSAP.pdf')