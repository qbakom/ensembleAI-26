import torch
import pandas as pd
from torch_geometric.loader import DataLoader
from model import ChEBIGIN
from dataset import ChEBIDataset
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load the Empty Test Dataset
    print("Loading test data (features without labels)...")
    try:
        test_dataset = ChEBIDataset(root="processed_data", file_name="test_empty_graphs.pt")
    except Exception as e:
        print("Run vectorize_data.py on the test parquet first! Error:", e)
        return
        
    # We do not shuffle test data to maintain ordering for the submission, 
    # though mol_id provides a solid mapping.
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    print(f"Total test graphs: {len(test_dataset)}")
    
    # 2. Load the specifically trained PyTorch model
    model = ChEBIGIN(
        node_dim=80, 
        edge_dim=6, 
        hidden_dim=256, 
        num_classes=500, 
        num_layers=4
    ).to(device)
    
    try:
        model.load_state_dict(torch.load("models/best_chebi_gin.pth", map_location=device, weights_only=True))
        print("Successfully loaded model weights from models/best_chebi_gin.pth")
    except Exception as e:
        print("Failed to load model weights. Did you train it fully? Error:", e)
        # We will continue anyway to show the pipeline structure (with random weights if not found)
        # return
        
    model.eval()
    
    # 3. Predict all graphs in batches
    all_mol_ids = []
    all_predictions = []
    
    print("Running predictions...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            batch = batch.to(device)
            # Forward pass provides raw logits
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            
            # Apply Sigmoid to convert logits to probabilities [0, 1]
            probs = torch.sigmoid(logits)
            
            # Convert probabilities to binary predictions (Threshold = 0.5)
            # You can tune this threshold later for precision/recall balance
            preds = (probs > 0.5).int()
            
            all_predictions.append(preds.cpu())
            all_mol_ids.extend(batch.mol_id)
            
    # Concatenate all batches
    all_predictions_tensor = torch.cat(all_predictions, dim=0)
    
    # 4. Save to Parquet Submission format
    # The output needs to be a DataFrame with 'mol_id' and 'class_0' to 'class_499'
    print("Formatting output DataFrame...")
    
    # Create column names
    class_columns = [f"class_{i}" for i in range(500)]
    
    # We create the dataframe efficiently from the numpy array map
    predictions_np = all_predictions_tensor.numpy()
    df = pd.DataFrame(predictions_np, columns=class_columns)
    
    # Insert mol_id at the first position
    df.insert(0, "mol_id", all_mol_ids)
    
    output_file = "my_submission.parquet"
    df.to_parquet(output_file, index=False)
    
    print(f"Success! Predictions saved to {output_file}")
    
    # Small preview
    print("\nPreview of the submission file:")
    print(df.head())

if __name__ == "__main__":
    main()
