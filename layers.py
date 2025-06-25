import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

import numpy as np 


################################################################################
# DECODER for UNDDIRECTED models
################################################################################
class InnerProductDecoder(torch.nn.Module):
    def forward(self, z, edge_index, sigmoid=True):
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self, z, sigmoid=True):
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj


################################################################################
# DECODER for DIRECTED models
################################################################################
class DirectedInnerProductDecoder(torch.nn.Module):
    def forward(self, s, t, edge_index, sigmoid=True):
        value = (s[edge_index[0]] * t[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self, s, t, sigmoid=True):
        adj = torch.matmul(s, t.t())
        return torch.sigmoid(adj) if sigmoid else adj


    
################################################################################
# UNDIRECTED model layers: BASIC version
################################################################################
class GCNConv(MessagePassing):
    def  __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


################################################################################
# Heterogeneous DIRECTED model layers: alpha, beta are supplied, 4 cell types
################################################################################
class Heterogeneous_DirectedGCNConv_4celltype(MessagePassing):
    def __init__(self, in_channels, out_channels, cell_types, alpha=1.0, beta=0.0, self_loops=True, adaptive=False):
        super(Heterogeneous_DirectedGCNConv_4celltype, self).__init__(aggr='add')
        self.lin = torch.nn.Linear(in_channels, out_channels)

        self.out_channels = out_channels

        self.alpha      = alpha
        self.beta       = beta

        self.self_loops = self_loops
        self.adaptive   = adaptive

        self.cell_type_0_mask = torch.tensor(np.array(cell_types==0))
        self.cell_type_1_mask = torch.tensor(np.array(cell_types==1))
        self.cell_type_2_mask = torch.tensor(np.array(cell_types==2))
        self.cell_type_3_mask = torch.tensor(np.array(cell_types==3))
        self.cell_types = cell_types

    def compute_norm(self,edge_index_celltype_reindex):
        ######################## celltype norm 
        row, col  = edge_index_celltype_reindex

        in_degree  = degree(col)
        out_degree = degree(row)

        alpha = self.alpha
        beta  = self.beta 

        in_norm_inv  = pow(in_degree,  -alpha)
        out_norm_inv = pow(out_degree, -beta)

        in_norm  = in_norm_inv[col]
        out_norm = out_norm_inv[row]
        norm     = in_norm * out_norm
        return norm


    def extract_celltype_edge(self,x,edge_index,celltype_index):
        cell_index  = np.arange(x.size(0))
        lst_celltype0 = cell_index[np.array(self.cell_types==celltype_index)].tolist()
        # print(lst_celltype0)

        temp = torch.isin(edge_index,torch.tensor(lst_celltype0))

        # test = temp[0,:] & temp[1,:]
        test = temp[0,:] | temp[1,:]
        edge_index_celltype0 = edge_index[:,test]
        node_index = edge_index_celltype0.reshape(-1)
        node_index = node_index.numpy().tolist()
        node_index = list(set(node_index) - set(lst_celltype0))
        node_index = sorted(node_index)
        select_node = lst_celltype0 + node_index

        mapping = {}
        for i in range(len(lst_celltype0)):
            mapping[lst_celltype0[i]] = i
        for i in range(len(node_index)):
            mapping[node_index[i]] = len(lst_celltype0)+i 

        edge_index_celltype_reindex = torch.zeros(edge_index_celltype0.size(0), edge_index_celltype0.size(1))
        for i in range(edge_index_celltype0.size(0)):
            for j in range(edge_index_celltype0.size(1)):
                temp = edge_index_celltype0[i,j].item()
                edge_index_celltype_reindex[i,j] = mapping[temp]
        edge_index_celltype_reindex = edge_index_celltype_reindex.long()
        return edge_index_celltype_reindex, select_node, lst_celltype0
    
    # only the 
    def forward(self, x, edge_index):
        if self.self_loops is True:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)

        # change by hu :
        edge_index_celltype0_reindex,select_node0, lst_celltype0 = self.extract_celltype_edge(x,edge_index,0) #[2, 2793]
        edge_index_celltype1_reindex,select_node1, lst_celltype1 = self.extract_celltype_edge(x,edge_index,1) #[2, 4195]
        edge_index_celltype2_reindex,select_node2, lst_celltype2 = self.extract_celltype_edge(x,edge_index,2) #[2, 4845]
        edge_index_celltype3_reindex,select_node3, lst_celltype3 = self.extract_celltype_edge(x,edge_index,3) #[2, 9785]

        norm0 = self.compute_norm(edge_index_celltype0_reindex)
        norm1 = self.compute_norm(edge_index_celltype1_reindex)
        norm2 = self.compute_norm(edge_index_celltype2_reindex)
        norm3 = self.compute_norm(edge_index_celltype3_reindex)
        
        # return self.propagate(edge_index, x=x, norm=norm)
        # test = self.propagate(edge_index_celltype0_reindex, x=x[self.cell_type_0_mask], norm=norm)

        emb_node_features = torch.zeros(x.shape[0], self.out_channels)
        emb_node_features[self.cell_type_0_mask] = self.propagate(edge_index_celltype0_reindex, x=x[select_node0], norm=norm0)[:len(lst_celltype0),:]
        emb_node_features[self.cell_type_1_mask] = self.propagate(edge_index_celltype1_reindex, x=x[select_node1], norm=norm1)[:len(lst_celltype1),:]
        emb_node_features[self.cell_type_2_mask] = self.propagate(edge_index_celltype2_reindex, x=x[select_node2], norm=norm2)[:len(lst_celltype2),:]
        emb_node_features[self.cell_type_3_mask] = self.propagate(edge_index_celltype3_reindex, x=x[select_node3], norm=norm3)[:len(lst_celltype3),:]

        return emb_node_features

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

################################################################################
# Heterogeneous DIRECTED model layers: alpha, beta are supplied, 7 cell types
################################################################################
# device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Heterogeneous_DirectedGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, cell_types, alpha=1.0, beta=0.0, self_loops=True, adaptive=False):
        super(Heterogeneous_DirectedGCNConv, self).__init__(aggr='add')
        self.lin = torch.nn.Linear(in_channels, out_channels)

        self.out_channels = out_channels

        self.alpha      = alpha
        self.beta       = beta

        self.self_loops = self_loops
        self.adaptive   = adaptive

        self.cell_type_0_mask = torch.tensor(np.array(cell_types==0))
        self.cell_type_1_mask = torch.tensor(np.array(cell_types==1))
        self.cell_type_2_mask = torch.tensor(np.array(cell_types==2))
        self.cell_type_3_mask = torch.tensor(np.array(cell_types==3))
        self.cell_type_4_mask = torch.tensor(np.array(cell_types==4))
        self.cell_type_5_mask = torch.tensor(np.array(cell_types==5))
        self.cell_type_6_mask = torch.tensor(np.array(cell_types==6))
        
        # #### modified by jinxian: 20250423 
        # self.cell_type_0_mask = torch.tensor(cell_types==0)
        # self.cell_type_1_mask = torch.tensor(cell_types==1)
        # self.cell_type_2_mask = torch.tensor(cell_types==2)
        # self.cell_type_3_mask = torch.tensor(cell_types==3)
        # self.cell_type_4_mask = torch.tensor(cell_types==4)
        # self.cell_type_5_mask = torch.tensor(cell_types==5)
        # self.cell_type_6_mask = torch.tensor(cell_types==6)

        self.cell_types = cell_types

    def compute_norm(self,edge_index_celltype_reindex):
        ######################## celltype norm 
        row, col  = edge_index_celltype_reindex

        in_degree  = degree(col)
        out_degree = degree(row)

        alpha = self.alpha
        beta  = self.beta 

        in_norm_inv  = pow(in_degree,  -alpha)
        out_norm_inv = pow(out_degree, -beta)

        in_norm  = in_norm_inv[col]
        out_norm = out_norm_inv[row]
        norm     = in_norm * out_norm
        return norm


    def extract_celltype_edge(self,x,edge_index,celltype_index):
        cell_index  = np.arange(x.size(0))
        lst_celltype0 = cell_index[np.array(self.cell_types==celltype_index)].tolist()

        # temp = torch.isin(edge_index,torch.tensor(lst_celltype0))
        # modified by jinxian 20250423
        temp = torch.isin(edge_index,torch.tensor(lst_celltype0).cuda())

        # test = temp[0,:] & temp[1,:]
        test = temp[0,:] | temp[1,:]
        edge_index_celltype0 = edge_index[:,test]
        node_index = edge_index_celltype0.reshape(-1)
        # node_index = node_index.numpy().tolist()
        # modified by jinxian 20250423
        node_index = node_index.cpu().numpy().tolist()
        node_index = list(set(node_index) - set(lst_celltype0))
        node_index = sorted(node_index)
        select_node = lst_celltype0 + node_index
        # print(select_node)

        mapping = {}
        for i in range(len(lst_celltype0)):
            mapping[lst_celltype0[i]] = i
        # print(mapping)  
        for i in range(len(node_index)):
            mapping[node_index[i]] = len(lst_celltype0)+i 

        edge_index_celltype_reindex = torch.zeros(edge_index_celltype0.size(0), edge_index_celltype0.size(1))
        for i in range(edge_index_celltype0.size(0)):
            for j in range(edge_index_celltype0.size(1)):
                temp = edge_index_celltype0[i,j].item()
                edge_index_celltype_reindex[i,j] = mapping[temp]
        edge_index_celltype_reindex = edge_index_celltype_reindex.long()
        # print(edge_index_celltype_reindex)
        return edge_index_celltype_reindex, select_node, lst_celltype0
    
    # only the 
    def forward(self, x, edge_index):
        if self.self_loops is True:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)
        x = x.cuda()

        # change by huo :
        edge_index_celltype0_reindex,select_node0, lst_celltype0 = self.extract_celltype_edge(x,edge_index,0) #[2, 2793]
        edge_index_celltype1_reindex,select_node1, lst_celltype1 = self.extract_celltype_edge(x,edge_index,1) #[2, 4195]
        edge_index_celltype2_reindex,select_node2, lst_celltype2 = self.extract_celltype_edge(x,edge_index,2) #[2, 4845]
        edge_index_celltype3_reindex,select_node3, lst_celltype3 = self.extract_celltype_edge(x,edge_index,3) #[2, 9785]
        edge_index_celltype4_reindex,select_node4, lst_celltype4 = self.extract_celltype_edge(x,edge_index,4) #[2, 9785]
        edge_index_celltype5_reindex,select_node5, lst_celltype5 = self.extract_celltype_edge(x,edge_index,5) #[2, 9785]
        edge_index_celltype6_reindex,select_node6, lst_celltype6 = self.extract_celltype_edge(x,edge_index,6) #[2, 9785]

        norm0 = self.compute_norm(edge_index_celltype0_reindex)
        norm1 = self.compute_norm(edge_index_celltype1_reindex)
        norm2 = self.compute_norm(edge_index_celltype2_reindex)
        norm3 = self.compute_norm(edge_index_celltype3_reindex)
        norm4 = self.compute_norm(edge_index_celltype4_reindex)
        norm5 = self.compute_norm(edge_index_celltype5_reindex)
        norm6 = self.compute_norm(edge_index_celltype6_reindex)

        
        # return self.propagate(edge_index, x=x, norm=norm)
        # test = self.propagate(edge_index_celltype0_reindex, x=x[self.cell_type_0_mask], norm=norm)

        emb_node_features = torch.zeros(x.shape[0], self.out_channels).cuda()
        # print("emb_node_features:", emb_node_features.device)
        # print("self.cell_type_0_mask:", self.cell_type_0_mask.device)
        # print("edge_index_celltype0_reindex:", edge_index_celltype0_reindex.device)
        # print("x[select_node0]:", x[select_node0].device)
        # print("norm0:", norm0.device)
        # modified by jinxian, 20250423 
        emb_node_features[self.cell_type_0_mask.cuda()] = self.propagate(edge_index_celltype0_reindex.cuda(), x=x[select_node0], norm=norm0.cuda())[:len(lst_celltype0),:]
        emb_node_features[self.cell_type_1_mask.cuda()] = self.propagate(edge_index_celltype1_reindex.cuda(), x=x[select_node1], norm=norm1.cuda())[:len(lst_celltype1),:]
        emb_node_features[self.cell_type_2_mask.cuda()] = self.propagate(edge_index_celltype2_reindex.cuda(), x=x[select_node2], norm=norm2.cuda())[:len(lst_celltype2),:]
        emb_node_features[self.cell_type_3_mask.cuda()] = self.propagate(edge_index_celltype3_reindex.cuda(), x=x[select_node3], norm=norm3.cuda())[:len(lst_celltype3),:]
        emb_node_features[self.cell_type_4_mask.cuda()] = self.propagate(edge_index_celltype4_reindex.cuda(), x=x[select_node4], norm=norm4.cuda())[:len(lst_celltype4),:]
        emb_node_features[self.cell_type_5_mask.cuda()] = self.propagate(edge_index_celltype5_reindex.cuda(), x=x[select_node5], norm=norm5.cuda())[:len(lst_celltype5),:]
        emb_node_features[self.cell_type_6_mask.cuda()] = self.propagate(edge_index_celltype6_reindex.cuda(), x=x[select_node6], norm=norm6.cuda())[:len(lst_celltype6),:]

        return emb_node_features

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


    
class Heterogeneous_SourceGCNConvEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, cell_type, alpha=1.0, beta=0.0, self_loops=True, adaptive=False):
        super(Heterogeneous_SourceGCNConvEncoder, self).__init__()
        self.conv1 = Heterogeneous_DirectedGCNConv(in_channels, hidden_channels, cell_type, alpha, beta, self_loops, adaptive)
        self.conv2 = Heterogeneous_DirectedGCNConv(hidden_channels, out_channels, cell_type, alpha, beta, self_loops, adaptive)
        

    def forward(self, x, edge_index):
        ###### modified by jinxian 20250423 
        edge_index = edge_index.cuda()

        x = F.relu(self.conv1(x, edge_index))
        # x = self.conv1(x, edge_index)
        
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, torch.flip(edge_index, [0]))

        return x

    

class Heterogeneous_TargetGCNConvEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, cell_type, alpha=1.0, beta=0.0, self_loops=True, adaptive=False):
        super(Heterogeneous_TargetGCNConvEncoder, self).__init__()
        self.conv1 = Heterogeneous_DirectedGCNConv(in_channels, hidden_channels, cell_type, alpha, beta, self_loops, adaptive)
        self.conv2 = Heterogeneous_DirectedGCNConv(hidden_channels, out_channels, cell_type, alpha, beta, self_loops, adaptive)
        

    def forward(self, x, edge_index):

        x = F.relu(self.conv1(x, torch.flip(edge_index, [0])))
        # x = self.conv1(x, torch.flip(edge_index, [0]))

        # x = F.dropout(x, p=0.5, training=self.training) 
        x = self.conv2(x, edge_index)

        return x



class Heterogeneous_DirectedGCNConvEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, cell_type, alpha=1.0, beta=0.0, self_loops=True, adaptive=False):
        super(Heterogeneous_DirectedGCNConvEncoder, self).__init__()
        self.source_conv = Heterogeneous_SourceGCNConvEncoder(in_channels, hidden_channels, out_channels, cell_type, alpha, beta, self_loops, adaptive)
        self.target_conv = Heterogeneous_TargetGCNConvEncoder(in_channels, hidden_channels, out_channels, cell_type, alpha, beta, self_loops, adaptive)

    def forward(self, s, t, edge_index):
        s = self.source_conv(s, edge_index) #[3798, 32]
        t = self.target_conv(t, edge_index) #[3798, 32]
        return s, t

    
################################################################################
# DIRECTED model layers: alpha, beta are supplied, BASIC version
################################################################################
class DirectedGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, alpha=1.0, beta=0.0, self_loops=True, adaptive=False):
        super(DirectedGCNConv, self).__init__(aggr='add')
        self.lin = torch.nn.Linear(in_channels, out_channels)

        # if adaptive is True:
        #     self.alpha = torch.nn.Parameter(torch.Tensor([alpha]))
        #     self.beta  = torch.nn.Parameter(torch.Tensor([beta]))
        # else:
        #     self.alpha      = alpha
        #     self.beta       = beta

        self.alpha      = alpha
        self.beta       = beta

        self.self_loops = self_loops
        self.adaptive   = adaptive

    
    def forward(self, x, edge_index):
        if self.self_loops is True:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)
        row, col  = edge_index

        in_degree  = degree(col)
        out_degree = degree(row)

        alpha = self.alpha
        beta  = self.beta 

        in_norm_inv  = pow(in_degree,  -alpha)
        out_norm_inv = pow(out_degree, -beta)

        in_norm  = in_norm_inv[col]
        out_norm = out_norm_inv[row]
        norm     = in_norm * out_norm

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


    
class SourceGCNConvEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, alpha=1.0, beta=0.0, self_loops=True, adaptive=False):
        super(SourceGCNConvEncoder, self).__init__()
        self.conv1 = DirectedGCNConv(in_channels, hidden_channels, alpha, beta, self_loops, adaptive)
        self.conv2 = DirectedGCNConv(hidden_channels, out_channels, alpha, beta, self_loops, adaptive)
        

    def forward(self, x, edge_index):

        x = F.relu(self.conv1(x, edge_index))
        # x = self.conv1(x, edge_index)
        
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, torch.flip(edge_index, [0]))

        return x

    

class TargetGCNConvEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, alpha=1.0, beta=0.0, self_loops=True, adaptive=False):
        super(TargetGCNConvEncoder, self).__init__()
        self.conv1 = DirectedGCNConv(in_channels, hidden_channels, alpha, beta, self_loops, adaptive)
        self.conv2 = DirectedGCNConv(hidden_channels, out_channels, alpha, beta, self_loops, adaptive)
        

    def forward(self, x, edge_index):

        x = F.relu(self.conv1(x, torch.flip(edge_index, [0])))
        # x = self.conv1(x, torch.flip(edge_index, [0]))

        # x = F.dropout(x, p=0.5, training=self.training) 
        x = self.conv2(x, edge_index)

        return x



class DirectedGCNConvEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, alpha=1.0, beta=0.0, self_loops=True, adaptive=False):
        super(DirectedGCNConvEncoder, self).__init__()
        self.source_conv = SourceGCNConvEncoder(in_channels, hidden_channels, out_channels, alpha, beta, self_loops, adaptive)
        self.target_conv = TargetGCNConvEncoder(in_channels, hidden_channels, out_channels, alpha, beta, self_loops, adaptive)

    def forward(self, s, t, edge_index):
        s = self.source_conv(s, edge_index) #[3798, 32]
        t = self.target_conv(t, edge_index) #[3798, 32]
        return s, t


################################################################################
# DIRECTED models: single layer
################################################################################
class SingleLayerSourceGCNConvEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, alpha=1.0, beta=0.0, self_loops=True, adaptive=False):
        super(SingleLayerSourceGCNConvEncoder, self).__init__()
        self.conv = DirectedGCNConv(in_channels, out_channels, alpha, beta, self_loops, adaptive)

    def forward(self, x, edge_index):
        # x = F.relu(self.conv1(x, edge_index))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv(x, torch.flip(edge_index, [0]))

        return x

    

class SingleLayerTargetGCNConvEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, alpha=1.0, beta=0.0, self_loops=True, adaptive=False):
        super(SingleLayerTargetGCNConvEncoder, self).__init__()
        self.conv = DirectedGCNConv(in_channels, out_channels, alpha, beta, self_loops, adaptive)
        
    def forward(self, x, edge_index):
        # x = F.relu(self.conv1(x, torch.flip(edge_index, [0])))
        # x = F.dropout(x, p=0.5, training=self.training) 
        x = self.conv(x, edge_index)

        return x


    
class SingleLayerDirectedGCNConvEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, alpha=1.0, beta=0.0, self_loops=True, adaptive=False):
        super(SingleLayerDirectedGCNConvEncoder, self).__init__()
        self.source_conv = SingleLayerSourceGCNConvEncoder(in_channels, out_channels, alpha, beta, self_loops, adaptive)
        self.target_conv = SingleLayerTargetGCNConvEncoder(in_channels, out_channels, alpha, beta, self_loops, adaptive)

    def forward(self, s_0, t_0, edge_index):
        s_1 = self.source_conv(t_0, edge_index)
        t_1 = self.target_conv(s_0, edge_index)
        return s_1, t_1



class DummyEncoder(nn.Module):
    def __init__(self):
        super(DummyEncoder, self).__init__()
        self.linear = torch.nn.Linear(1, 1)
        
    def forward(self, x, edge_index):
        return x


class DummyPairEncoder(nn.Module):
    def __init__(self):
        super(DummyPairEncoder, self).__init__()
        self.linear = torch.nn.Linear(1, 1)
        
    def forward(self, s, t, edge_index):
        return s, t

