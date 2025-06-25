################################################################################
# Heterogeneous DIRECTED model layers: alpha, beta are supplied, BASIC version
################################################################################
class Heterogeneous_DirectedGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, cell_types, alpha=1.0, beta=0.0, self_loops=True, adaptive=False):
        super(Heterogeneous_DirectedGCNConv, self).__init__(aggr='add')
        self.lin = torch.nn.Linear(in_channels, out_channels)

        # if adaptive is True:
        #     self.alpha = torch.nn.Parameter(torch.Tensor([alpha]))
        #     self.beta  = torch.nn.Parameter(torch.Tensor([beta]))
        # else:
        #     self.alpha      = alpha
        #     self.beta       = beta

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
        mapping = {}
        for i in range(len(lst_celltype0)):
            mapping[lst_celltype0[i]] = i
        # print(mapping)

        temp = torch.isin(edge_index,torch.tensor(lst_celltype0))

        test = temp[0,:] & temp[1,:]
        edge_index_celltype0 = edge_index[:,test]
        # print(edge_index_celltype0)

        edge_index_celltype_reindex = torch.zeros(edge_index_celltype0.size(0), edge_index_celltype0.size(1))
        for i in range(edge_index_celltype0.size(0)):
            for j in range(edge_index_celltype0.size(1)):
                temp = edge_index_celltype0[i,j].item()
                edge_index_celltype_reindex[i,j] = mapping[temp]
        edge_index_celltype_reindex = edge_index_celltype_reindex.long()
        # print(edge_index_celltype_reindex)
        return edge_index_celltype_reindex
    
    # only the 
    def forward(self, x, edge_index):
        if self.self_loops is True:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)

        # change by huo :
        edge_index_celltype0_reindex = self.extract_celltype_edge(x,edge_index,0) #[2, 2793]
        edge_index_celltype1_reindex = self.extract_celltype_edge(x,edge_index,1) #[2, 4195]
        edge_index_celltype2_reindex = self.extract_celltype_edge(x,edge_index,2) #[2, 4845]
        edge_index_celltype3_reindex = self.extract_celltype_edge(x,edge_index,3) #[2, 9785]

        norm0 = self.compute_norm(edge_index_celltype0_reindex)
        norm1 = self.compute_norm(edge_index_celltype1_reindex)
        norm2 = self.compute_norm(edge_index_celltype2_reindex)
        norm3 = self.compute_norm(edge_index_celltype3_reindex)
        
        # return self.propagate(edge_index, x=x, norm=norm)
        # test = self.propagate(edge_index_celltype0_reindex, x=x[self.cell_type_0_mask], norm=norm)

        emb_node_features = torch.zeros(x.shape[0], self.out_channels)
        emb_node_features[self.cell_type_0_mask] = self.propagate(edge_index_celltype0_reindex, x=x[self.cell_type_0_mask], norm=norm0)
        emb_node_features[self.cell_type_1_mask] = self.propagate(edge_index_celltype1_reindex, x=x[self.cell_type_1_mask], norm=norm1)
        emb_node_features[self.cell_type_2_mask] = self.propagate(edge_index_celltype2_reindex, x=x[self.cell_type_2_mask], norm=norm2)
        emb_node_features[self.cell_type_3_mask] = self.propagate(edge_index_celltype3_reindex, x=x[self.cell_type_3_mask], norm=norm3)
        # print(emb_node_features)
        # print(emb_node_features.size())
        # print(stop)

        return emb_node_features

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


    
class Heterogeneous_SourceGCNConvEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, cell_type, alpha=1.0, beta=0.0, self_loops=True, adaptive=False):
        super(Heterogeneous_SourceGCNConvEncoder, self).__init__()
        self.conv1 = Heterogeneous_DirectedGCNConv(in_channels, hidden_channels, cell_type, alpha, beta, self_loops, adaptive)
        self.conv2 = Heterogeneous_DirectedGCNConv(hidden_channels, out_channels, cell_type, alpha, beta, self_loops, adaptive)
        

    def forward(self, x, edge_index):

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
