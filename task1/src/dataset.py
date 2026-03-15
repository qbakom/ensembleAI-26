import os
import torch
from torch_geometric.data import Dataset

class ChEBIDataset(Dataset):
    def __init__(self, root, file_name, transform=None, pre_transform=None):
        """
        Args:
            root (str): Directory where the dataset file is stored.
            file_name (str): Name of the `.pt` file containing the list of PyG Data objects.
        """
        self.file_path = os.path.join(root, file_name)
        super(ChEBIDataset, self).__init__(root, transform, pre_transform)
        
        # Load the entire list of graphs into RAM
        # For huge datasets, we would keep this on disk and load in `get()`,
        # but 11k graphs (or even 100k) usually fit in RAM nicely.
        print(f"Loading dataset from {self.file_path}...")
        self.data_list = torch.load(self.file_path, weights_only=False)
        print(f"Loaded {len(self.data_list)} graphs.")

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [os.path.basename(self.file_path)]

    def download(self):
        pass

    def process(self):
        pass

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    # Test loading
    try:
        dataset = ChEBIDataset(root="processed_data", file_name="example_graphs.pt")
        print(f"Dataset length: {len(dataset)}")
        print(f"First element: {dataset[0]}")
    except Exception as e:
        print("Dataset test failed (file might not exist):", e)
