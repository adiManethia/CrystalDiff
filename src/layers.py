import torch 
import torch.nn as nn 

class EGNNLayer(nn.Module):
    """
    Equivariant GNN.
    Update node features (h) and coordinates (x) while respecting rotation.
    """ 

    def __init__(self, c_in, c_out):
        super().__init__()

        # Edge MLP: Compute message based on features and distance
        # input: h_i + h_j + distance(1)
        self.edge_mlp = nn.Sequential(
            nn.Linear(c_in * 2 + 1, c_out), # c_in * 2 + 1 means we concatenate h_i, h_j and the distance scalar
            nn.SiLU(),
            nn.Linear(c_out, c_out),
            nn.SiLU()
        )
        # Node MLP: Update atom features 
        # Input: h_i + aggregated_message
        self.node_mlp = nn.Sequential(
            nn.Linear(c_in + c_out, c_out),
            nn.SiLU(),
            nn.Linear(c_out, c_out)
        )

        # Coord MLP: Update position (x)
        # Input : message (c_out)
        self.coord_mlp = nn.Sequential(
            nn.Linear(c_out, 1), # output a single scalar 'weight' for the coordinate update
            nn.Tanh() # keeps updates stable (-1 to 1)
        )

    def forward(self, h, x, edge_index):
        """
        h: Node features (N, c_in)
        x: Coordinates (N, 3)
        edge_index: Adjacency list (2, E) where E is number of edges-> who connects to whom 
        """

        row, col = edge_index   # row = source, col = target 

        # setp 1 : calculate distance
        # get coordinates of source and target nodes
        x_i = x[row] # (E, 3)
        x_j = x[col] # (E, 3),
        # example: if edge_index has [0, 1] in row and [2, 3] in col, 
        # then x_i will have coordinates of nodes 0 and 1, while x_j will have coordinates of nodes 2 and 3

        # calculate squared distance (rotation invariant)
        dist_sq = torch.sum((x_i - x_j)**2, dim=-1, keepdim=True) 
        # sum(-1) means we sum over the coordinate dimension, resulting in a scalar distance for each edge. keepdim=True keeps the output shape as (E, 1)

        # step 2 : calculate edge messages
        # Concatenate: Feature_i, Feature_j, Distance 
        # h[row] for source node features, h[col] for target node features
        edge_input = torch.cat([h[row], h[col], dist_sq], dim=-1) # (E, c_in*2 + 1)
        
        # pass through edge MLP to get messages
        m_ij = self.edge_mlp(edge_input) 

        # step 3 : Update coordinates ( Equivariant part)
        # predict a weight for vector (x_i - x_j) based on the message
        coord_weight = self.coord_mlp(m_ij) 

        # Update x_new = x + sum((x_i - x_j) * weight), transform
        trans = (x_i - x_j) * coord_weight 

        # Aggregate coordinate updates using scatter_add_ (preserves autograd)
        idx_exp = row.unsqueeze(-1).expand(-1, x.size(-1))  # (E, 3)
        x_agg = torch.zeros_like(x)
        x_agg = x_agg.scatter_add_(0, idx_exp, trans)

        x_new = x + x_agg

        # step 4 : Update node features
        m_idx_exp = row.unsqueeze(-1).expand(-1, m_ij.size(-1))
        m_agg = torch.zeros(h.shape[0], m_ij.shape[1], device=h.device)
        m_agg = m_agg.scatter_add_(0, m_idx_exp, m_ij)

        # Combine old features with new message 

        h_input = torch.cat([h, m_agg], dim=-1)
        h_new = self.node_mlp(h_input)

        return h_new, x_new