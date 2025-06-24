import math
import numpy as np
import pandas as pd 
import torch
from torch_geometric.utils import to_undirected
import scipy.sparse as sp


# def general_train_test_split_edges(data, val_ratio=0.05, test_ratio=0.1, directed=True):
#     r"""Splits the edges of a :obj:`torch_geometric.data.Data` object
#     into positive and negative train/val/test edges, and adds attributes of
#     `train_pos_edge_index`, `train_neg_adj_mask`, `val_pos_edge_index`,
#     `val_neg_edge_index`, `test_pos_edge_index`, and `test_neg_edge_index`
#     to :attr:`data`.

#     Args:
#         data (Data): The data object.
#         val_ratio (float, optional): The ratio of positive validation
#             edges. (default: :obj:`0.05`)
#         test_ratio (float, optional): The ratio of positive test
#             edges. (default: :obj:`0.1`)

#     :rtype: :class:`torch_geometric.data.Data`
#     """

#     assert 'batch' not in data  # No batch-mode.

#     num_nodes = data.num_nodes
#     row, col = data.edge_index
#     data.edge_index = None

#     if not directed:
#         # Return upper triangular portion.
#         mask = row < col
#         row, col = row[mask], col[mask]

#     n_v = int(math.floor(val_ratio * row.size(0)))
#     n_t = int(math.floor(test_ratio * row.size(0)))

#     # Positive edges.
#     perm = torch.randperm(row.size(0))
#     row, col = row[perm], col[perm]

#     r, c = row[:n_v], col[:n_v]
#     data.val_pos_edge_index = torch.stack([r, c], dim=0)
#     r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
#     data.test_pos_edge_index = torch.stack([r, c], dim=0)
#     print(data.test_pos_edge_index)

#     r, c = row[n_v + n_t:], col[n_v + n_t:]
#     data.train_pos_edge_index = torch.stack([r, c], dim=0)

#     if not directed:
#         data.train_pos_edge_index = to_undirected(data.train_pos_edge_index)

#     # Negative edges.
#     neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)

#     if not directed:
#         neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)

#     neg_adj_mask = neg_adj_mask.to(torch.bool)
#     neg_adj_mask[row, col] = 0

#     neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
#     perm = torch.randperm(neg_row.size(0))[:n_v + n_t]
#     neg_row, neg_col = neg_row[perm], neg_col[perm]

#     neg_adj_mask[neg_row, neg_col] = 0
#     data.train_neg_adj_mask = neg_adj_mask

#     row, col = neg_row[:n_v], neg_col[:n_v]
#     data.val_neg_edge_index = torch.stack([row, col], dim=0)

#     row, col = neg_row[n_v:n_v + n_t], neg_col[n_v:n_v + n_t]
#     data.test_neg_edge_index = torch.stack([row, col], dim=0)
#     print(data.test_neg_edge_index)

#     return data

def sampling_test_edges_neg(n, test_edges, edges_double):
    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, n)
        idx_j = np.random.randint(0, n)
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_double):
            continue
        if ismember([idx_j, idx_i], edges_double):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])
    return test_edges_false

def sparse2tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def ismember(tmp1, tmp2, tol=5):
    """
    Judge whether there are overlapping elements in tmp1 and tmp2
    """
    rows_close = np.all(np.round(tmp1 - tmp2[:, None], tol) == 0, axis=-1)
    if True in np.any(rows_close, axis=-1).tolist():
        return True
    elif True not in np.any(rows_close, axis=-1).tolist():
        return False

def general_train_test_split_edges(data, val_ratio=0.05, test_ratio=0.1, directed=True):
    r"""Splits the edges of a :obj:`torch_geometric.data.Data` object
    into positive and negative train/val/test edges, and adds attributes of
    `train_pos_edge_index`, `train_neg_adj_mask`, `val_pos_edge_index`,
    `val_neg_edge_index`, `test_pos_edge_index`, and `test_neg_edge_index`
    to :attr:`data`.

    Args:
        data (Data): The data object.
        val_ratio (float, optional): The ratio of positive validation
            edges. (default: :obj:`0.05`)
        test_ratio (float, optional): The ratio of positive test
            edges. (default: :obj:`0.1`)

    :rtype: :class:`torch_geometric.data.Data`
    """

    assert 'batch' not in data  # No batch-mode.

    ################################# DeepLink split
    adj_values = data.adj
    # Get the id of all edges
    edges_single = sparse2tuple(sp.triu(adj_values))[0]  #single direction of edges, (2863, 2)
    edges_double = sparse2tuple(adj_values)[0]  #double direction of edges, (5726, 2)

    if test_ratio > 1:
        test_ratio = test_ratio/edges_single.shape[0]

    # Split into train and test sets
    num_test = int(np.floor(edges_single.shape[0] * test_ratio))
    all_edges_idx = list(range(edges_single.shape[0]))
    np.random.shuffle(all_edges_idx)
    test_edges_idx = all_edges_idx[:num_test]
    test_edges = edges_single[test_edges_idx]
    if (adj_values.shape[0]**2-adj_values.sum()-adj_values.shape[0])/2 < 2*len(test_edges):
        raise ImportError('The network is too dense, please reduce the proportion of test set or delete some edges in the network.')
    else:
        test_edges_false = sampling_test_edges_neg(adj_values.shape[0], test_edges, edges_double)
        test_edges_false = np.array(test_edges_false)

    print(test_edges)
    print(test_edges_false)

    r, c = test_edges[:,0],test_edges[:,1]
    test_edges = torch.stack([torch.from_numpy(r), torch.from_numpy(c)], dim=0)
    test_edges = to_undirected(test_edges)

    r, c = test_edges_false[:,0],test_edges_false[:,1]
    test_edges_false = torch.stack([torch.from_numpy(r), torch.from_numpy(c)], dim=0)
    test_edges_false = to_undirected(test_edges_false)

    train_edges = np.delete(edges_single, test_edges_idx, axis=0)
    r, c = train_edges[:,0],train_edges[:,1]
    train_edges = torch.stack([torch.from_numpy(r), torch.from_numpy(c)], dim=0)
    train_edges = to_undirected(train_edges)

    data.val_pos_edge_index = test_edges.long()
    data.test_pos_edge_index = test_edges.long()
    data.train_pos_edge_index = train_edges.long()
    # print(data.val_pos_edge_index)
    # print(data.train_pos_edge_index)

    ########################################

    num_nodes = data.num_nodes
    row, col = data.edge_index
    data.edge_index = None

    # if not directed:
    #     # Return upper triangular portion.
    #     mask = row < col
    #     row, col = row[mask], col[mask]

    # n_v = int(math.floor(val_ratio * row.size(0)))
    # n_t = int(math.floor(test_ratio * row.size(0)))

    # # Positive edges.
    # perm = torch.randperm(row.size(0))
    # row, col = row[perm], col[perm]

    # r, c = row[:n_v], col[:n_v]
    # data.val_pos_edge_index = torch.stack([r, c], dim=0)
    # r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
    # data.test_pos_edge_index = torch.stack([r, c], dim=0)
    # print(data.test_pos_edge_index)

    # r, c = row[n_v + n_t:], col[n_v + n_t:]
    # data.train_pos_edge_index = torch.stack([r, c], dim=0)

    # if not directed:
    #     data.train_pos_edge_index = to_undirected(data.train_pos_edge_index)

    # # Negative edges.
    # neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)

    # # if not directed:
    # #     neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
    # neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)

    # neg_adj_mask = neg_adj_mask.to(torch.bool)
    # neg_adj_mask[row, col] = 0

    # # neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
    # # perm = torch.randperm(neg_row.size(0))[:n_v + n_t]
    # # neg_row, neg_col = neg_row[perm], neg_col[perm]

    # # neg_adj_mask[neg_row, neg_col] = 0
    # data.train_neg_adj_mask = neg_adj_mask

    # row, col = neg_row[:n_v], neg_col[:n_v]
    # data.val_neg_edge_index = torch.stack([row, col], dim=0)

    # row, col = neg_row[n_v:n_v + n_t], neg_col[n_v:n_v + n_t]
    # data.test_neg_edge_index = torch.stack([row, col], dim=0)
    # print(data.test_neg_edge_index)

    data.val_neg_edge_index = test_edges_false.long()
    data.test_neg_edge_index = test_edges_false.long()
    # print(data.val_neg_edge_index)
    # print(data.test_neg_edge_index)

    return data

def fake_train_test_split_edges(data, fake_ratio=1, directed=True):
    r"""Splits the edges of a :obj:`torch_geometric.data.Data` object
    into positive and negative train/val/test edges, and adds attributes of
    `train_pos_edge_index`, `train_neg_adj_mask`, `val_pos_edge_index`,
    `val_neg_edge_index`, `test_pos_edge_index`, and `test_neg_edge_index`
    to :attr:`data`.

    Args:
        data (Data): The data object.
        val_ratio (float, optional): The ratio of positive validation
            edges. (default: :obj:`0.05`)
        test_ratio (float, optional): The ratio of positive test
            edges. (default: :obj:`0.1`)

    :rtype: :class:`torch_geometric.data.Data`
    """

    assert 'batch' not in data  # No batch-mode.

    num_nodes = data.num_nodes
    row, col = data.edge_index
    data.edge_index = None

    if not directed:
        # Return upper triangular portion.
        mask = row < col
        row, col = row[mask], col[mask]

    # n_v = int(math.floor(val_ratio * row.size(0)))
    # n_t = int(math.floor(test_ratio * row.size(0)))
    n_fake = int(math.floor(fake_ratio * row.size(0)))
    print('n_fake:',n_fake)

    # Positive edges.
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]

    # r, c = row[:n_v], col[:n_v]
    # data.val_pos_edge_index = torch.stack([r, c], dim=0)
    # r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
    # data.test_pos_edge_index = torch.stack([r, c], dim=0)
    # print(data.test_pos_edge_index)
    # r, c = row[n_v + n_t:], col[n_v + n_t:]
    # data.train_pos_edge_index = torch.stack([r, c], dim=0)

    data.val_pos_edge_index = torch.stack([row, col], dim=0)
    data.test_pos_edge_index = torch.stack([row, col], dim=0)
    # data.train_pos_edge_index = torch.stack([row, col], dim=0)

    # if not directed:
    #     data.train_pos_edge_index = to_undirected(data.train_pos_edge_index)

    # Negative edges.
    neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)

    if not directed:
        neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)

    neg_adj_mask = neg_adj_mask.to(torch.bool)
    neg_adj_mask[row, col] = 0
    # print(neg_adj_mask) #0:true edges, 1:negtive edges

    neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
    # perm = torch.randperm(neg_row.size(0))[:n_v + n_t] 
    perm = torch.randperm(neg_row.size(0))
    neg_row, neg_col = neg_row[perm], neg_col[perm] # shuffle val and test edges

    # add by Jin-Xian
    train_r = torch.cat((row, neg_row[:n_fake]),0)
    train_c = torch.cat((col, neg_col[:n_fake]),0)
    data.train_pos_edge_index = torch.stack([train_r, train_c], dim=0)

    # neg_adj_mask[neg_row, neg_col] = 0 # delete val and test neg edges in train 
    neg_adj_mask[neg_row[:n_fake], neg_col[:n_fake]] = 0 # delete neg edges in train 
    data.train_neg_adj_mask = neg_adj_mask

    # row, col = neg_row[:n_v], neg_col[:n_v]
    # data.val_neg_edge_index = torch.stack([row, col], dim=0)

    # row, col = neg_row[n_v:n_v + n_t], neg_col[n_v:n_v + n_t]
    # data.test_neg_edge_index = torch.stack([row, col], dim=0)
    # print(data.test_neg_edge_index)
    row, col = neg_row[:n_fake], neg_col[:n_fake]
    data.val_neg_edge_index = torch.stack([row, col], dim=0)
    data.test_neg_edge_index = torch.stack([row, col], dim=0)

    return data

def celltypespecific_train_test_split_edges(data, val_ratio=0.05, test_ratio=0.1, directed=True):
    r"""Splits the edges of a :obj:`torch_geometric.data.Data` object
    into positive and negative train/val/test edges, and adds attributes of
    `train_pos_edge_index`, `train_neg_adj_mask`, `val_pos_edge_index`,
    `val_neg_edge_index`, `test_pos_edge_index`, and `test_neg_edge_index`
    to :attr:`data`.

    Args:
        data (Data): The data object.
        val_ratio (float, optional): The ratio of positive validation
            edges. (default: :obj:`0.05`)
        test_ratio (float, optional): The ratio of positive test
            edges. (default: :obj:`0.1`)

    :rtype: :class:`torch_geometric.data.Data`
    """

    assert 'batch' not in data  # No batch-mode.

    num_nodes = data.num_nodes
    row, col = data.edge_index
    data.edge_index = None

    # Microglia(5)->Astrocyte(0)

    print(data.y)
    print(data.y.numpy())
    cell_type = data.y.numpy()
    cell_map = {}
    for i in range(len(cell_type)):
        cell_map[i] = cell_type[i]
    print(cell_map[5])

    row_celltype = []
    col_celltype = []
    row_index = row.numpy()
    col_index = col.numpy()
    for i in range(len(row)):
    # for i in range(10):
        row_celltype.append(cell_map[row_index[i]])
        col_celltype.append(cell_map[col_index[i]])
    df_index = pd.DataFrame({'row_celltype':row_celltype,'col_celltype':col_celltype})
    print(df_index)
    print(df_index[(df_index['row_celltype']==5) & (df_index['row_celltype']==0)])
    print(stop)

    if not directed:
        # Return upper triangular portion.
        mask = row < col
        row, col = row[mask], col[mask]

    n_v = int(math.floor(val_ratio * row.size(0)))
    n_t = int(math.floor(test_ratio * row.size(0)))

    # Positive edges.
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]

    r, c = row[:n_v], col[:n_v]
    data.val_pos_edge_index = torch.stack([r, c], dim=0)
    r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
    data.test_pos_edge_index = torch.stack([r, c], dim=0)
    print(data.test_pos_edge_index)

    r, c = row[n_v + n_t:], col[n_v + n_t:]
    data.train_pos_edge_index = torch.stack([r, c], dim=0)

    if not directed:
        data.train_pos_edge_index = to_undirected(data.train_pos_edge_index)

    # Negative edges.
    neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)

    if not directed:
        neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)

    neg_adj_mask = neg_adj_mask.to(torch.bool)
    neg_adj_mask[row, col] = 0

    neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
    perm = torch.randperm(neg_row.size(0))[:n_v + n_t]
    neg_row, neg_col = neg_row[perm], neg_col[perm]

    neg_adj_mask[neg_row, neg_col] = 0
    data.train_neg_adj_mask = neg_adj_mask

    row, col = neg_row[:n_v], neg_col[:n_v]
    data.val_neg_edge_index = torch.stack([row, col], dim=0)

    row, col = neg_row[n_v:n_v + n_t], neg_col[n_v:n_v + n_t]
    data.test_neg_edge_index = torch.stack([row, col], dim=0)
    print(data.test_neg_edge_index)

    return data

def biased_train_test_split_edges(data, val_ratio=0.05, test_ratio=0.10, directed=True):
    num_nodes = data.num_nodes
    edge_index = data.edge_index.clone()

    data.edge_index = None

    num_edges = edge_index.size(1)
    n_v       = int(math.floor(val_ratio * num_edges))
    n_t       = int(math.floor(test_ratio * num_edges))

    edge_set = set([tuple(pair) for pair in edge_index.clone().numpy().T.tolist()])

    u_row  = []
    u_col  = []
    b_row = []
    b_col = []

    for (a, b) in edge_set:
        if (b, a) not in edge_set:
            u_row.append(a)
            u_col.append(b)
        else:
            b_row.append(a)
            b_col.append(b)

    n_u  = len(u_row)
    n_b  = len(b_row)
    n_vt = n_v + n_t
    assert(n_vt <= n_u)

    u_perm = np.random.permutation(range(n_u))
    u_row = np.array(u_row)
    u_col = np.array(u_col)
    u_row, u_col = u_row[u_perm], u_col[u_perm]

    b_perm = np.random.permutation(range(n_b))
    b_row = np.array(b_row)
    b_col = np.array(b_col)
    b_row, b_col = b_row[b_perm], b_col[b_perm]

    row = np.hstack([u_row, b_row])
    col = np.hstack([u_col, b_col])

    row = torch.from_numpy(row).long()
    col = torch.from_numpy(col).long()

    r, c = row[:n_v], col[:n_v]
    data.val_pos_edge_index = torch.stack([r, c], dim=0)
    data.val_neg_edge_index = torch.stack([c, r], dim=0)

    r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
    data.test_pos_edge_index = torch.stack([r, c], dim=0)
    data.test_neg_edge_index = torch.stack([c, r], dim=0)

    r, c = row[n_v + n_t:], col[n_v + n_t:]
    data.train_pos_edge_index = torch.stack([r, c], dim=0)

    return data


def bidirectional_train_test_split_edges(data, val_ratio=0.05, test_ratio=0.10, directed=True):

    num_nodes = data.num_nodes
    edge_index = data.edge_index.clone()
    data.edge_index = None

    num_edges = edge_index.size(1)
    n_v       = int(math.floor(val_ratio * num_edges))
    n_t       = int(math.floor(test_ratio * num_edges))

    edge_set = set([tuple(pair) for pair in edge_index.clone().numpy().T.tolist()])

    # removed edges in the training set
    r_row  = []
    r_col  = []

    # kept edges in the training set
    k_row = []
    k_col = []

    # unidirectional edges
    u_row = []
    u_col = []

    for (a, b) in edge_set:
        if (b, a) in edge_set:
            if a > b:
                r_row.append(a)
                r_col.append(b)
        else:
            u_row.append(a)
            u_col.append(b)

    # XXX  shuffling/permutation for r, k arrays        
    k_row = r_col.copy()
    k_col = r_row.copy()

    n_r = len(r_row)
    n_k = len(k_row)
    n_u = len(u_row)

    u_perm = np.random.permutation(range(n_u))
    u_row = np.array(u_row)
    u_col = np.array(u_col)

    u_row, u_col = u_row[u_perm], u_col[u_perm]
    k_row = np.array(k_row)
    k_col = np.array(k_col)

    r = np.hstack([u_row, k_row])
    c = np.hstack([u_col, k_col])
    r = torch.from_numpy(r).long()
    c = torch.from_numpy(c).long()
    data.train_pos_edge_index = torch.stack([r, c], dim=0)

    assert(n_u >= n_r)
    nu_row = u_col[:n_r].copy()
    nu_col = u_row[:n_r].copy()
    r_row = np.array(r_row)
    r_col = np.array(r_col)
    
    r = torch.from_numpy(r_row).long()
    c = torch.from_numpy(r_col).long()
    data.test_pos_edge_index = torch.stack([r, c], dim=0)

    r = torch.from_numpy(nu_row).long()
    c = torch.from_numpy(nu_col).long()
    data.test_neg_edge_index = torch.stack([r, c], dim=0)

    return data
