import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from model import ChEBIGIN
from dataset import ChEBIDataset
import copy
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(loader, desc="Training"):
        batch = batch.to(device)
        
        optimizer.zero_grad()
        
        # Omit 'y' from inputs. Model signature: x, edge_index, edge_attr, batch
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        
        # ChEBI dataset targets are [batch_size, 500]
        # BCEWithLogitsLoss expects raw logits, which our model produces.
        target = batch.y
        loss = criterion(out, target)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch.num_graphs
        
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    
    for batch in tqdm(loader, desc="Evaluating"):
        batch = batch.to(device)
        
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        target = batch.y
        
        loss = criterion(out, target)
        total_loss += loss.item() * batch.num_graphs
        
    return total_loss / len(loader.dataset)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Dataset
    print("Loading data...")
    try:
        # Ładowanie całego wygenerowanego zbioru treningowego
        full_dataset = ChEBIDataset(root="processed_data", file_name="train_graphs.pt")
    except Exception as e:
        print("Please run vectorize_data.py first to generate processed_data/example_graphs.pt")
        return

    # Split dataset into Train and Val (e.g. 80/20)
    dataset_size = len(full_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # 2. Initialize Model
    model = ChEBIGIN(
        node_dim=80, 
        edge_dim=6, 
        hidden_dim=256, 
        num_classes=500, 
        num_layers=4
    ).to(device)
    
    # 3. Training Setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    epochs = 5 # Just a short run for verification
    best_loss = float('inf')
    best_model_weights = None
    
    print("\nStarting Training...")
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            
    print("\nTraining complete!")
    print(f"Best Val Loss: {best_loss:.4f}")
    
    # Save the model
    os.makedirs("models", exist_ok=True)
    torch.save(best_model_weights, "models/best_chebi_gin.pth")
    print("Saved best model to models/best_chebi_gin.pth")

if __name__ == "__main__":
    main()
