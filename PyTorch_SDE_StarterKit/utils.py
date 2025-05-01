import os
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

def get_dataloaders(config):
    X_train = torch.randn(500, config.input_dim)
    y_train = torch.randint(0, config.output_dim, (500,))
    train_ds = TensorDataset(X_train, y_train)
    return DataLoader(train_ds, batch_size=config.batch_size), None

def save_model(model, output_dir, name):
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, name))
    print(f"Model saved to {os.path.join(output_dir, name)}")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
