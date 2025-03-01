import torch
from torch import nn
from torch.utils.data import Dataset
from typing import List, Tuple
import pandas as pd

class CategoryDataset(Dataset):
    def __init__(self, data: pd.DataFrame, cat_cols: List[str], num_cols: List[str], target_col: List[str]) -> None:
        """
        Categorical features should be encoded
        """
        self.data = data
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.target_col = target_col

        self.cat_data = self.data[self.cat_cols].values
        self.num_data = self.data[self.num_cols].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return categorical data, numerical data and target
        """
        x_cat = self.cat_data[index]
        x_num = self.num_data[index]
        y = self.data[self.target_col].values[index]

        return torch.tensor(x_cat, dtype=torch.int64), torch.tensor(x_num, dtype=torch.float32), torch.tensor(y, dtype=torch.int64)

class CategoryEmbeddingModel(nn.Module):
    def __init__(self, embedding_dims, n_num_cols):
        super().__init__()
        self.category_embeddings = nn.ModuleList([nn.Embedding(x, y) for x, y in embedding_dims])  # embedding for each categorical feature

        in_features = n_num_cols + sum(dim[1] for dim in embedding_dims)
        self.n_cat_cols = len(embedding_dims)
        self.n_num_cols = n_num_cols
        
        self.backbone = nn.Sequential(*[
            nn.Linear(in_features, 64, bias=True),
            nn.ReLU(),
            nn.Linear(64, 32, bias=True),
            nn.ReLU()
        ])

        self.normalizing_batch_norm = nn.BatchNorm1d(n_num_cols)

        self.head = nn.Sequential(*[
            nn.Linear(32, 2, bias=True)
        ])

        cat_num_categories = torch.tensor([dim[0] for dim in embedding_dims], dtype=torch.int64)
        self.register_buffer("cat_num_categories", cat_num_categories)

        # init weights
        for layer in self.backbone:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x_cat, x_num):
        # work-around unknown categories
        x_cat += 1  # 0 is <UNKNOWN>
        expanded_categories = self.cat_num_categories.unsqueeze(0).expand(x_cat.size(0), -1)  # [batch_size, num_cat_features]
        x_cat = torch.where(x_cat < expanded_categories, x_cat, 0)
        
        # encode each categorical
        x_cat_encoded = [self.category_embeddings[i](x_cat[:, i]) for i in range(self.n_cat_cols)]

        # concatenate output from each categorical embedder
        x_cat_encoded = torch.cat(x_cat_encoded, dim=1)

        # preprocess num_features with normalizing
        x_num_encoded = self.normalizing_batch_norm(x_num)

        # concatenate categorical and numarical
        x = torch.cat([x_cat_encoded, x_num_encoded], dim=1)

        # pass to the mlp
        x = self.backbone(x)
        x = self.head(x)
        return x