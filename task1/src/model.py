import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d
from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool

class ChEBIGIN(torch.nn.Module):
    def __init__(self, node_dim=80, edge_dim=6, hidden_dim=128, num_classes=500, num_layers=4):
        super(ChEBIGIN, self).__init__()
        
        # Initial node embedding to align the input feature dimension
        self.node_emb = Linear(node_dim, hidden_dim)
        
        # Edge embedding layer so edge features match hidden_dim for GINEConv
        self.edge_emb = Linear(edge_dim, hidden_dim)
        
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        
        # Construct multiple GINE convolution layers
        for _ in range(num_layers):
            # MLP for the GIN aggregation
            mlp = Sequential(
                Linear(hidden_dim, 2 * hidden_dim),
                BatchNorm1d(2 * hidden_dim),
                ReLU(),
                Linear(2 * hidden_dim, hidden_dim)
            )
            # GINEConv allows handling of edge attributes natively
            self.convs.append(GINEConv(nn=mlp, edge_dim=hidden_dim))
            self.batch_norms.append(BatchNorm1d(hidden_dim))
            
        # Classification MLP after global pooling
        # We concatenate mean and max pooling resulting in 2 * hidden_dim
        self.classifier = Sequential(
            Linear(2 * hidden_dim, hidden_dim),
            ReLU(),
            BatchNorm1d(hidden_dim),
            torch.nn.Dropout(p=0.5),
            Linear(hidden_dim, num_classes)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        # 1. Embed Input Node and Edge features
        x = self.node_emb(x)
        
        # Handle cases where molecules might have 0 edges
        if edge_attr is not None and edge_attr.numel() > 0:
            edge_attr = self.edge_emb(edge_attr)
        else:
            edge_attr = torch.empty((0, x.size(-1)), device=x.device)

        # 2. Message Passing steps
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            
        # 3. Global Pooling (Graph level readout)
        # Represents the entire molecule as a single vector
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        
        # Concatenate both pooling strategies
        x = torch.cat([x_mean, x_max], dim=1)
        
        # 4. Classification
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    # Simulate a small batch to ensure shapes map correctly
    import warnings
    warnings.filterwarnings('ignore')
    
    # 2 molecules, e.g. 5 atoms total, 4 edges total
    x = torch.randn((5, 80))
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    edge_attr = torch.randn((4, 6))
    batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long) # 3 atoms in mol 0, 2 atoms in mol 1
    
    model = ChEBIGIN(node_dim=80, edge_dim=6, hidden_dim=128, num_classes=500, num_layers=4)
    print("Model initialized.")
    out = model(x, edge_index, edge_attr, batch)
    print("Output shape:", out.shape) # Should be [2, 500]
