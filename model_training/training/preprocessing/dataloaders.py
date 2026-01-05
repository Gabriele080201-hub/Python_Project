import torch 
from torch.utils.data import TensorDataset, DataLoader

def create_dataloaders(X_train, y_train, X_val, y_val, batch_size = 64, shuffle_train = True):
    X_train_t = torch.tensor(X_train, dtype = torch.float32)
    y_train_t = torch.tensor(y_train, dtype = torch.float32).unsqueeze(1)

    X_val_t = torch.tensor(X_val, dtype = torch.float32)
    y_val_t = torch.tensor(y_val, dtype = torch.float32).unsqueeze(1)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=shuffle_train)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


