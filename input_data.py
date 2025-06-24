import networkx as nx
import scipy.sparse as sp

import os
import numpy as np
import pandas as pd 
import pickle

import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, to_undirected
from Citation import load_npz_dataset, train_test_split

from torch_geometric.datasets import WebKB



def load_data(dataset_name = 'cora_ml', directed=True):
    if dataset_name == 'wisconsin':
        dataset = WebKB(root='./data', name='Wisconsin')
        data    = dataset[0]
        if not directed:
            data.edge_index = to_undirected(data.edge_index)
    elif dataset_name == 'cornell':
        dataset = WebKB(root='./data', name='Cornell')
        data    = dataset[0]
        if not directed:
            data.edge_index = to_undirected(data.edge_index)
    elif dataset_name == 'texas':
        dataset = WebKB(root='./data', name='Texas')
        data    = dataset[0]
        if not directed:
            data.edge_index = to_undirected(data.edge_index)
    # human breast cancer
    elif dataset_name == 'HBC': 
        data = load_data_from_csv(dataset_name, directed) # PCA genes, Adj_dist
        # data = load_data_from_anndata(dataset_name, directed) # LR genes, Adj_dist
        # data = generate_graph_from_anndata(dataset_name, directed) # LR genes, Adj_LR
    elif dataset_name == 'StereoSeq': 
        data = load_data_from_csv(dataset_name, directed) # PCA genes, Adj_dist
        # data = load_data_from_anndata(dataset_name, directed) # LR genes, Adj_dist
    elif dataset_name == 'MERFISH': 
        data = load_data_from_merfish(dataset_name, directed) 
    elif dataset_name == 'seqFISH': 
        data = load_data_from_seqfish(dataset_name, directed) 
    elif dataset_name == 'HDST_ob' or dataset_name == 'HDST_cancer': 
        data = load_data_from_HDST(dataset_name, directed) 
    else:
        data = load_data_from_npz(dataset_name, directed)

    return data

def read_dataset(input_exp, input_adj, filter_num=None, add_number=None):
    exp = pd.read_csv(open(input_exp))
    adj = pd.read_csv(open(input_adj))

    if not (adj.values==adj.values.T).all():
        raise ImportError('The input adjacency matrix is not a symmetric matrix, please check the input.')
    if not np.diag(adj.values).sum()==0:
        raise ImportError('The diagonal elements of input adjacency matrix are not all 0, please check the input.')

    if filter_num is not None:
        exp = filter_gene(exp, filter_num)
    if add_number is not None:
        exp = log_transform(exp, add_number)

    return exp, adj

def read_coordinate(input_coord):
    coord = pd.read_csv(open(input_coord))  
    return coord

def read_cell_label(input_label):
    label = pd.read_csv(open(input_label))  
    return label

def read_LR(input_LR,LR_type):
    LR_df = pd.read_csv(open(input_LR)) 
    if LR_type:
        LR_df = LR_df[LR_df['annotation']==LR_type]
    # LR_df = LR_df[LR_df['annotation']=='Cell-Cell Contact']
    # LR_df = LR_df[LR_df['annotation']=='Secreted Signaling']
    # LR_df = LR_df[LR_df['annotation']!='ECM-Receptor']

    interaction = LR_df['interaction_name'].values
    Ligand_list = []
    Receptor_list = []
    Receptor_all = []
    for interaction_i in interaction:
        interaction_i = interaction_i.split('_')
        Ligand_list.append(interaction_i[0])
        Receptor_list.append(interaction_i[1:]) 
        # Receptor_all = Receptor_all + interaction_i[1:]
    return Ligand_list, Receptor_list, interaction

def get_data(data_path, data_name, lambda_I):

    ############## load distance matrix and cell type
    data_file = data_path + data_name +'/'
    with open(data_file + 'Adjacent', 'rb') as fp:
        adj_dist = pickle.load(fp)

    X_data = np.load(data_file + 'features.npy')

    if data_name == 'HBC':
        ############## load distance matrix and cell type
        # Adjacent_LR_secreted, Adjacent_LR_cellcellcontact, Adjacent_LR_ECM
        with open(data_file + 'Adjacent_LR_contact_secreted', 'rb') as fp:
            adj_LR = pickle.load(fp)

        cell_type_indeces = np.load(data_file + 'cell_types.npy')

        # Healthy_1, Healthy_2, 
        # Tumor_edge_1, Tumor_edge_2, Tumor_edge_3, Tumor_edge_4, Tumor_edge_5, Tumor_edge_6
        # DCIS/LCIS_1, DCIS/LCIS_2, DCIS/LCIS_3, DCIS/LCIS_4, DCIS/LCIS_5
        # IDC_1, IDC_2, IDC_3, IDC_4, IDC_5, IDC_6, IDC_7
        cell_type_dict = {0:0,1:0, 2:1,3:1,4:1,5:1,6:1,7:1, 8:2,9:2,10:2,11:2,12:2, 13:3,14:3,15:3,16:3,17:3,18:3,19:3}
        cell_type = [] 
        for cell_i in cell_type_indeces:
            cell_type.append(cell_type_dict[cell_i])
    else:
        cell_type_indeces = np.load(data_file + 'cell_types.npy',allow_pickle=True)
        cell_type = cell_type_indeces.astype(np.int32)

    ############## decide whether integrate distance matrix into sender_reciever_matrix 
    # num_points = X_data.shape[0]
    # adj_I = np.eye(num_points)
    # adj_I = sp.csr_matrix(adj_I)
    # adj = (1-lambda_I)*adj_0 + lambda_I*adj_I
    adj = adj_dist
    # adj = adj_LR
    # adj = (1-lambda_I)*adj_dist + lambda_I*adj_LR
    # adj = adj_dist.todense() + adj_LR.todense()
    # adj[adj>1]=1
    # adj = sp.csr_matrix(adj)
    
    # return adj, X_data, cell_type_indeces
    return adj, X_data,  np.array(cell_type)


def load_data_from_csv(data_name = 'cora_ml', directed=True):
    # Import and pack datasets

    # python DeepLinc.py -exp ./dataset/seqFISH/counts.csv -adj ./dataset/seqFISH/adj.csv 
    # -coordinate ./dataset/seqFISH/coord.csv -reference ./dataset/seqFISH/cell_type_1.csv

    data_path = 'generated_data/'
    # data_name = 'V1_Breast_Cancer_Block_A_Section_1/'
    lambda_I = 0.8
    adj, X_data, cell_type_indeces = get_data(data_path, data_name, lambda_I)
    # adj, X_data, cell_type_indeces, genes_index = get_data(data_path, data_name, lambda_I)

    adj, features, labels = sp.csr_matrix(adj), sp.csr_matrix(X_data), cell_type_indeces
    print('adj:',np.shape(adj)) #(2995, 2995) -> (1597, 1597)
    print('features:',np.shape(features)) #(2995, 2879) -> (1597, 125)
    print('labels',np.shape(labels)) # (2995,) -> (1597,)

    # return adj, features, labels

    if not directed:
        adj  = (adj + adj.T) / 2.0
        
    # mask = train_test_split(labels, seed=1020, train_examples_per_class=20, val_size=500, test_size=None)
    mask = train_test_split(labels, seed=1020, train_examples_per_class=10, val_size=500, test_size=None)
    
    mask['train'] = torch.from_numpy(mask['train']).bool()
    mask['val'] = torch.from_numpy(mask['val']).bool()
    mask['test'] = torch.from_numpy(mask['test']).bool()

    adj = sp.csr_matrix(adj)

    coo = adj.tocoo()

    indices = np.vstack((coo.row, coo.col))
    indices = torch.from_numpy(indices).long()
    
    values = coo.data
    values = torch.from_numpy(values).float()
    
    features = torch.from_numpy(features.todense()).float()
    
    labels   = torch.from_numpy(labels).long()
    
    edge_index  = indices
    edge_weight = values
    
    data = Data(x=features, edge_index=edge_index, edge_weight=edge_weight, y=labels)

    data.train_mask = mask['train']
    data.val_mask = mask['val']
    data.test_mask = mask['test']
    # data.genes_index = genes_index
    
    if data_name == 'HBC':
        adata_X = np.load('processed_data/V1_Breast_Cancer_Block_A_Section_1/adata_wopca.npy',allow_pickle=True)
        adata_X = torch.from_numpy(adata_X).float()
        data.adata_X = adata_X
    data.adj = adj

    return data

def load_data_from_anndata(dataset_name = 'cora_ml', directed=True):
    # Import and pack datasets

    # python DeepLinc.py -exp ./dataset/seqFISH/counts.csv -adj ./dataset/seqFISH/adj.csv 
    # -coordinate ./dataset/seqFISH/coord.csv -reference ./dataset/seqFISH/cell_type_1.csv

    data_path = 'generated_data/'
    data_name = dataset_name+'/'
    lambda_I = 0.3
    
    #############  load ligand-Receptor, and generate sender_reciever_matrix 
    if dataset_name == 'V1_Breast_Cancer_Block_A_Section_1':
        LR_file = 'data/LR/CellChatDB.human/interaction.csv'
    if dataset_name == 'StereoSeq':
        LR_file = 'data/LR/CellChatDB.mouse/interaction.csv'
    LR_type = 'Cell-Cell Contact' #'Cell-Cell Contact', 'Secreted Signaling',False, 'ECM-Receptor'
    Ligand_list, Receptor_list, interaction_name = read_LR(LR_file,LR_type)
    assert len(Ligand_list) == len(Receptor_list) == len(interaction_name) #1939
    print(len(interaction_name))

    if dataset_name == 'V1_Breast_Cancer_Block_A_Section_1':
        X_data = np.load('processed_data/V1_Breast_Cancer_Block_A_Section_1/adata_wopca.npy',allow_pickle=True)
        genes = np.load('processed_data/V1_Breast_Cancer_Block_A_Section_1/genes_filter.npy',allow_pickle=True)
    if dataset_name == 'StereoSeq':
        # after normalization and gene filter
        X_data = np.load('processed_data/StereoSeq/X.npy',allow_pickle=True)
        genes_original = np.load('processed_data/StereoSeq/genes.npy',allow_pickle=True)
        genes = [gene.upper() for gene in genes_original]

    exp_df = pd.DataFrame(data=X_data,columns=genes)
    print(exp_df)

    u_feat = []
    v_feat = []
    use_LR_list = []
    for index in range(len(Ligand_list)):
        l_gene = Ligand_list[index]
        r_gene = Receptor_list[index]
        # print('l_gene:',l_gene)
        # print('r_gene:',r_gene)
        lf_gene = [l_gene] + r_gene
        if (set(lf_gene) < set(list(genes))):
            l_expression = exp_df[l_gene].values
            if len(r_gene) == 1:
                r_expression = exp_df[r_gene[0]].values
            else:
                r_expression = exp_df[r_gene[0]].values
                for ri in range(1,len(r_gene)):
                    r_expression = r_expression + exp_df[r_gene[ri]].values

            u_feat.append(l_expression)
            v_feat.append(r_expression)
            use_LR_list.append(interaction_name[index])

    u_feat = np.array(u_feat)
    v_feat = np.array(v_feat)
    u_feat = u_feat.T
    v_feat = v_feat.T
    print('u_feat: ',np.shape(u_feat))
    print('v_feat: ',np.shape(v_feat))
    print('use_LR_list: ', len(use_LR_list))
    use_LR_list = np.array(use_LR_list)

    ############## load distance matrix and cell type
    data_file = data_path + data_name +'/'
    with open(data_file + 'Adjacent', 'rb') as fp:
        adj = pickle.load(fp)

    if data_name == 'V1_Breast_Cancer_Block_A_Section_1':
        cell_type_indeces = np.load(data_file + 'cell_types.npy')
    else:
        cell_type_indeces = np.load(data_file + 'cell_types.npy',allow_pickle=True)
        cell_type_indeces = cell_type_indeces.astype(np.int32)

    adj, features, labels = sp.csr_matrix(adj), sp.csr_matrix(X_data), cell_type_indeces
    print('adj:',np.shape(adj)) #(2995, 2995) -> (1597, 1597)
    print('features:',np.shape(features)) #(2995, 2879) -> (1597, 125)
    print('labels',np.shape(labels)) # (2995,) -> (1597,)

    u_feat = sp.csr_matrix(u_feat)
    v_feat = sp.csr_matrix(v_feat)

    # return adj, features, labels

    if not directed:
        adj  = (adj + adj.T) / 2.0
        
    # mask = train_test_split(labels, seed=1020, train_examples_per_class=20, val_size=500, test_size=None)
    mask = train_test_split(labels, seed=1020, train_examples_per_class=10, val_size=500, test_size=None)
    
    mask['train'] = torch.from_numpy(mask['train']).bool()
    mask['val'] = torch.from_numpy(mask['val']).bool()
    mask['test'] = torch.from_numpy(mask['test']).bool()

    adj = sp.csr_matrix(adj)

    coo = adj.tocoo()

    indices = np.vstack((coo.row, coo.col))
    indices = torch.from_numpy(indices).long()
    
    values = coo.data
    values = torch.from_numpy(values).float()
    
    features = torch.from_numpy(features.todense()).float()
    
    labels   = torch.from_numpy(labels).long()

    edge_index  = indices
    edge_weight = values
    
    data = Data(x=features, edge_index=edge_index, edge_weight=edge_weight, y=labels)

    data.train_mask = mask['train']
    data.val_mask = mask['val']
    data.test_mask = mask['test']
    # data.genes_index = genes_index

    u_feat = torch.from_numpy(u_feat.todense()).float()
    v_feat = torch.from_numpy(v_feat.todense()).float()
    data.u_feat = u_feat
    data.v_feat = v_feat
    data.LR_name = use_LR_list
    data.adj = adj

    return data

def generate_graph_from_anndata(dataset_name = 'cora_ml', directed=True):
    # Import and pack datasets

    # python DeepLinc.py -exp ./dataset/seqFISH/counts.csv -adj ./dataset/seqFISH/adj.csv 
    # -coordinate ./dataset/seqFISH/coord.csv -reference ./dataset/seqFISH/cell_type_1.csv

    data_path = 'generated_data/'
    data_name = 'V1_Breast_Cancer_Block_A_Section_1/'
    lambda_I = 0.3
    LR_type = False #'Cell-Cell Contact', 'Secreted Signaling',False, 'ECM-Receptor'

    # # ############## load distance matrix
    data_file = data_path + data_name +'/'
    # with open(data_file + 'Adjacent', 'rb') as fp:
    #     adj = pickle.load(fp)
    # adj = adj.todense()
    # print(adj)

    #############  load ligand-Receptor, and generate sender_reciever_matrix 
    LR_file = 'data/LR/CellChatDB.human/interaction.csv'
    Ligand_list, Receptor_list = read_LR(LR_file,LR_type)
    assert len(Ligand_list) == len(Receptor_list) #1939

    # exp_df = pd.read_csv('processed_data/V1_Breast_Cancer_Block_A_Section_1/raw_adata.csv',index_col=0)
    # exp_df = pd.read_csv('processed_data/V1_Breast_Cancer_Block_A_Section_1/normalized_adata.csv',index_col=0)
    # ligand_receptor_exp = exp_df[ligand_receptor_genes].values
    X_data = np.load('processed_data/V1_Breast_Cancer_Block_A_Section_1/adata_wopca.npy',allow_pickle=True)
    genes = np.load('processed_data/V1_Breast_Cancer_Block_A_Section_1/genes_filter.npy',allow_pickle=True)
    # print(exp_df.columns.values)
    exp_df = pd.DataFrame(data=X_data,columns=genes)
    print(exp_df)

    adj_LR = np.zeros((len(exp_df),len(exp_df)))
    u_feat = []
    v_feat = []
    use_ligand_list = []
    use_receptor_list = []
    for index in range(len(Ligand_list)):
        l_gene = Ligand_list[index]
        r_gene = Receptor_list[index]
        use_ligand_list.append(l_gene)
        use_receptor_list.append(r_gene)
        # print('l_gene:',l_gene)
        # print('r_gene:',r_gene)
        lf_gene = [l_gene] + r_gene
        if (set(lf_gene) < set(list(genes))):
            l_expression = exp_df[l_gene].values
            if len(r_gene) == 1:
                r_expression = exp_df[r_gene[0]].values
            else:
                r_expression = exp_df[r_gene[0]].values
                for ri in range(1,len(r_gene)):
                    r_expression = r_expression + exp_df[r_gene[ri]].values

            u_feat.append(l_expression)
            v_feat.append(r_expression)

            # method1: top3 high expression cells
            sender_index = np.argsort(l_expression)[::-1]
            reciever_index = np.argsort(r_expression)[::-1]
            # print(sender_index) 
            # print(reciever_index)
            for i in range(5):
                for j in range(5):
                    adj_LR[sender_index[i],reciever_index[j]] = 1
                    # print(sender_index[i],reciever_index[j])

            # method2: Z*Z(T), select top3 high  
    
    u_feat = np.array(u_feat)
    v_feat = np.array(v_feat)
    u_feat = u_feat.T
    v_feat = v_feat.T
    print('u_feat: ',np.shape(u_feat)) #(3798, 831)
    print('v_feat: ',np.shape(v_feat))

    # save adj_LR
    adj_LR = np.float32(adj_LR) ## do not normalize adjcent matrix
    adj_LR_crs = sp.csr_matrix(adj_LR)
    with open(data_file + 'Adjacent_LR', 'wb') as fp:
        pickle.dump(adj_LR_crs, fp)
    print(stop)

    # # method2: Z*Z(T), select top3 high  
    # exp_matrix = np.matmul(u_feat,v_feat.T)
    # print(exp_matrix)
    # adj_LR = np.zeros((len(exp_df),len(exp_df)))
    # for i in range(np.shape(exp_matrix)[0]):
    #     row_exp = exp_matrix[i,:]
    #     index = np.argsort(row_exp)[::-1]
    #     for j in range(3):
    #         adj_LR[i,index[j]] = 1

    # adj = adj_LR
    # adj = adj + adj_LR

    ############### load cell type
    cell_type_indeces = np.load(data_file + 'cell_types.npy')

    adj, features, labels = sp.csr_matrix(adj), sp.csr_matrix(X_data), cell_type_indeces
    print('adj:',np.shape(adj)) #(2995, 2995) -> (1597, 1597)
    print('features:',np.shape(features)) #(2995, 2879) -> (1597, 125)
    print('labels',np.shape(labels)) # (2995,) -> (1597,)

    u_feat = sp.csr_matrix(u_feat)
    v_feat = sp.csr_matrix(v_feat)

    # return adj, features, labels

    if not directed:
        adj  = (adj + adj.T) / 2.0
        
    # mask = train_test_split(labels, seed=1020, train_examples_per_class=20, val_size=500, test_size=None)
    mask = train_test_split(labels, seed=1020, train_examples_per_class=10, val_size=500, test_size=None)
    
    mask['train'] = torch.from_numpy(mask['train']).bool()
    mask['val'] = torch.from_numpy(mask['val']).bool()
    mask['test'] = torch.from_numpy(mask['test']).bool()

    adj = sp.csr_matrix(adj)

    coo = adj.tocoo()

    indices = np.vstack((coo.row, coo.col))
    indices = torch.from_numpy(indices).long()
    
    values = coo.data
    values = torch.from_numpy(values).float()
    
    features = torch.from_numpy(features.todense()).float()
    
    labels   = torch.from_numpy(labels).long()

    edge_index  = indices
    edge_weight = values
    
    data = Data(x=features, edge_index=edge_index, edge_weight=edge_weight, y=labels)

    data.train_mask = mask['train']
    data.val_mask = mask['val']
    data.test_mask = mask['test']
    # data.genes_index = genes_index

    u_feat = torch.from_numpy(u_feat.todense()).float()
    v_feat = torch.from_numpy(v_feat.todense()).float()
    data.u_feat = u_feat
    data.v_feat = v_feat

    return data

def load_data_from_seqfish(dataset_name = 'cora_ml', directed=True):
    # Import and pack datasets

    # python DeepLinc.py -exp ./dataset/seqFISH/counts.csv -adj ./dataset/seqFISH/adj.csv 
    # -coordinate ./dataset/seqFISH/coord.csv -reference ./dataset/seqFISH/cell_type_1.csv

    data_path = 'data/'
    data_name = 'seqFISH/'
    lambda_I = 0.3

    exp_file = 'data/seqFISH/counts.csv'
    adj_file = 'data/seqFISH/adj.csv'
    coord_file = 'data/seqFISH/coord.csv'
    label_file = 'data/seqFISH/cell_type.csv'
    exp_df, adj_df = read_dataset(exp_file, adj_file, None, None)
    exp, adj = exp_df.values, adj_df.values

    coord_df = read_coordinate(coord_file)
    coord = coord_df.values
    cell_label_df = read_cell_label(label_file)
    cell_label = cell_label_df['Cell_class_id'].values

    adj, features, labels = sp.csr_matrix(adj), sp.csr_matrix(exp), cell_label
    print('adj:',np.shape(adj)) #(2995, 2995) -> (1597, 1597)
    print('features:',np.shape(features)) #(2995, 2879) -> (1597, 125)
    print('labels',np.shape(labels)) # (2995,) -> (1597,)

    # return adj, features, labels

    if not directed:
        adj  = (adj + adj.T) / 2.0
        
    # mask = train_test_split(labels, seed=1020, train_examples_per_class=20, val_size=500, test_size=None)
    mask = train_test_split(labels, seed=1020, train_examples_per_class=10, val_size=500, test_size=None)
    
    mask['train'] = torch.from_numpy(mask['train']).bool()
    mask['val'] = torch.from_numpy(mask['val']).bool()
    mask['test'] = torch.from_numpy(mask['test']).bool()

    adj = sp.csr_matrix(adj)

    coo = adj.tocoo()

    indices = np.vstack((coo.row, coo.col))
    indices = torch.from_numpy(indices).long()
    
    values = coo.data
    values = torch.from_numpy(values).float()
    
    features = torch.from_numpy(features.todense()).float()
    
    labels   = torch.from_numpy(labels).long()
    
    edge_index  = indices
    edge_weight = values
    
    data = Data(x=features, edge_index=edge_index, edge_weight=edge_weight, y=labels)

    data.train_mask = mask['train']
    data.val_mask = mask['val']
    data.test_mask = mask['test']
    # data.genes_index = genes_index
    data.adj = adj

    return data

def load_data_from_merfish(dataset_name = 'cora_ml', directed=True):
    # Import and pack datasets

    # python DeepLinc.py -exp ./dataset/seqFISH/counts.csv -adj ./dataset/seqFISH/adj.csv 
    # -coordinate ./dataset/seqFISH/coord.csv -reference ./dataset/seqFISH/cell_type_1.csv

    data_path = 'data/'
    data_name = 'MERFISH/'
    lambda_I = 0.3

    ############## load distance matrix and cell type
    data_file = data_path + data_name +'/'
    with open(data_file + 'Adjacent', 'rb') as fp:
        adj = pickle.load(fp)
    features_df = pd.read_csv(data_file + 'counts.csv')
    features = features_df.values
    labels_df = pd.read_csv(data_file + 'cell_type.csv')
    labels = labels_df['Cell_class_id'].values

    adj, features, labels = sp.csr_matrix(adj), sp.csr_matrix(features), labels
    print('adj:',np.shape(adj)) #(2995, 2995) -> (1597, 1597)
    print('features:',np.shape(features)) #(2995, 2879) -> (1597, 125)
    print('labels',np.shape(labels)) # (2995,) -> (1597,)

    # return adj, features, labels

    if not directed:
        adj  = (adj + adj.T) / 2.0
        
    # mask = train_test_split(labels, seed=1020, train_examples_per_class=20, val_size=500, test_size=None)
    mask = train_test_split(labels, seed=1020, train_examples_per_class=10, val_size=500, test_size=None)
    
    mask['train'] = torch.from_numpy(mask['train']).bool()
    mask['val'] = torch.from_numpy(mask['val']).bool()
    mask['test'] = torch.from_numpy(mask['test']).bool()

    adj = sp.csr_matrix(adj)

    coo = adj.tocoo()

    indices = np.vstack((coo.row, coo.col))
    indices = torch.from_numpy(indices).long()
    
    values = coo.data
    values = torch.from_numpy(values).float()
    
    features = torch.from_numpy(features.todense()).float()
    
    labels   = torch.from_numpy(labels).long()
    
    edge_index  = indices
    edge_weight = values
    
    data = Data(x=features, edge_index=edge_index, edge_weight=edge_weight, y=labels)

    data.train_mask = mask['train']
    data.val_mask = mask['val']
    data.test_mask = mask['test']
    # data.genes_index = genes_index
    data.adj = adj

    return data

def load_data_from_HDST(dataset_name = 'cora_ml', directed=True, filter_num=None, add_number=None):
    # Import and pack datasets

    # python DeepLinc.py -exp ./dataset/seqFISH/counts.csv -adj ./dataset/seqFISH/adj.csv 
    # -coordinate ./dataset/seqFISH/coord.csv -reference ./dataset/seqFISH/cell_type_1.csv

    data_path = 'data/'
    data_name = dataset_name + '/'
    lambda_I = 0.3

    data_file = data_path + data_name +'/'
    exp_df = pd.read_csv(open(data_file + 'counts.csv'))
    adj_df = pd.read_csv(open(data_file + 'adj.csv'))

    if not (adj_df.values==adj_df.values.T).all():
        raise ImportError('The input adjacency matrix is not a symmetric matrix, please check the input.')
    if not np.diag(adj_df.values).sum()==0:
        raise ImportError('The diagonal elements of input adjacency matrix are not all 0, please check the input.')

    if filter_num is not None:
        exp = filter_gene(exp, filter_num)
    if add_number is not None:
        exp = log_transform(exp, add_number)

    exp, adj = exp_df.values, adj_df.values
    ############## load distance matrix and cell type
    labels_df = pd.read_csv(data_file + 'cell_type.csv')
    labels = labels_df['Cell_class_id'].values

    adj, features, labels = sp.csr_matrix(adj), sp.csr_matrix(exp), labels
    print('adj:',np.shape(adj)) #(2995, 2995) -> (1597, 1597)
    print('features:',np.shape(features)) #(2995, 2879) -> (1597, 125)
    print('labels',np.shape(labels)) # (2995,) -> (1597,)

    # return adj, features, labels

    if not directed:
        adj  = (adj + adj.T) / 2.0
        
    # mask = train_test_split(labels, seed=1020, train_examples_per_class=20, val_size=500, test_size=None)
    mask = train_test_split(labels, seed=1020, train_examples_per_class=10, val_size=500, test_size=None)
    
    mask['train'] = torch.from_numpy(mask['train']).bool()
    mask['val'] = torch.from_numpy(mask['val']).bool()
    mask['test'] = torch.from_numpy(mask['test']).bool()

    adj = sp.csr_matrix(adj)

    coo = adj.tocoo()

    indices = np.vstack((coo.row, coo.col))
    indices = torch.from_numpy(indices).long()
    
    values = coo.data
    values = torch.from_numpy(values).float()
    
    features = torch.from_numpy(features.todense()).float()
    
    labels   = torch.from_numpy(labels).long()
    
    edge_index  = indices
    edge_weight = values
    
    data = Data(x=features, edge_index=edge_index, edge_weight=edge_weight, y=labels)

    data.train_mask = mask['train']
    data.val_mask = mask['val']
    data.test_mask = mask['test']
    # data.genes_index = genes_index
    data.adj = adj

    return data


def load_data_from_npz(dataset_name = 'cora_ml', directed=True):
    dataset_path = os.path.join('./data/{}/raw'.format(dataset_name), '{}.npz'.format(dataset_name))
    g = load_npz_dataset(dataset_path)
    adj, features, labels = g['A'], g['X'], g['z']
    print('adj:',np.shape(adj)) #(2995, 2995)
    print('features:',np.shape(features)) #(2995, 2879)
    print('labels',np.shape(labels)) # (2995,)

    if not directed:
        adj  = (adj + adj.T) / 2.0
        
    mask = train_test_split(labels, seed=1020, train_examples_per_class=20, val_size=500, test_size=None)
    
    mask['train'] = torch.from_numpy(mask['train']).bool()
    mask['val'] = torch.from_numpy(mask['val']).bool()
    mask['test'] = torch.from_numpy(mask['test']).bool()

    coo = adj.tocoo()

    indices = np.vstack((coo.row, coo.col))
    indices = torch.from_numpy(indices).long()
    
    values = coo.data
    values = torch.from_numpy(values).float()
    
    features = torch.from_numpy(features.todense()).float()
    
    labels   = torch.from_numpy(labels).long()
    
    edge_index  = indices
    edge_weight = values
    
    data = Data(x=features, edge_index=edge_index, edge_weight=edge_weight, y=labels)

    data.train_mask = mask['train']
    data.val_mask = mask['val']
    data.test_mask = mask['test']

    return data



def gravity_load_data(dataset, load_features=True):
    if dataset == 'cora_ml':
        data    = load_data(dataset, directed=True)
        adj      = nx.to_scipy_sparse_matrix(to_networkx(data))
        if load_features:
            feature_array = data.x.detach().clone().numpy()
        else:
            feature_array = np.identity(adj.shape[0])        

    elif dataset == 'citeseer':
        data     = load_data(dataset, directed=True)
        adj      = nx.to_scipy_sparse_matrix(to_networkx(data))
        if load_features:
            feature_array = data.x.detach().clone().numpy()
        else:
            feature_array = np.identity(adj.shape[0])

    elif dataset == 'wisconsin':
        data     = load_data(dataset, directed=True)
        adj      = nx.to_scipy_sparse_matrix(to_networkx(data))
        if load_features:
            feature_array = data.x.detach().clone().numpy()
        else:
            feature_array = np.identity(adj.shape[0])

    elif dataset == 'cornell':
        data     = load_data(dataset, directed=True)
        adj      = nx.to_scipy_sparse_matrix(to_networkx(data))
        if load_features:
            feature_array = data.x.detach().clone().numpy()
        else:
            feature_array = np.identity(adj.shape[0])

    elif dataset == 'texas':
        data     = load_data(dataset, directed=True)
        adj      = nx.to_scipy_sparse_matrix(to_networkx(data))
        if load_features:
            feature_array = data.x.detach().clone().numpy()
        else:
            feature_array = np.identity(adj.shape[0])

    return adj, feature_array


# Adapted/copied from:
# https://github.com/deezer/gravity_graph_autoencoders
def original_gravity_load_data(dataset):

    if dataset == 'cora_ml':
        adj = nx.adjacency_matrix(nx.read_edgelist("./data/cora.cites",
                                                   delimiter='\t',
                                                   create_using=nx.DiGraph()))
        # Transpose the adjacency matrix, as Cora raw dataset comes with a
        # <ID of cited paper> <ID of citing paper> edgelist format.
        adj = adj.T
        features = sp.identity(adj.shape[0])

    elif dataset == 'citeseer':
        adj = nx.adjacency_matrix(nx.read_edgelist("./data/citeseer.cites",
                                                   delimiter='\t',
                                                   create_using=nx.DiGraph()))
        # Transpose the adjacency matrix, as Citeseer raw dataset comes with a
        # <ID of cited paper> <ID of citing paper> edgelist format.
        adj = adj.T
        features = sp.identity(adj.shape[0])
    else:
        raise ValueError('Undefined dataset!')

    return adj, features
