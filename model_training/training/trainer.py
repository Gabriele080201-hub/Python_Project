import torch 
from typing import Dict, List, Tuple 
from pathlib import Path

def train_one_epoch(model, dataloader, optimizer, criterion, device) -> float:
    model.train()
    running_loss = 0.0

    for xb, yb in dataloader:
        xb = xb.to(device).unsqueeze(-1)
        yb = yb.to(device)

        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.size(0)

    return running_loss / len(dataloader.dataset)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device) -> float:
    """
    Valuta il modello su validation/test set.
    """
    model.eval()
    running_loss = 0.0

    for xb, yb in dataloader:
        xb = xb.to(device).unsqueeze(-1)
        yb = yb.to(device)

        preds = model(xb)
        loss = criterion(preds, yb)

        running_loss += loss.item() * xb.size(0)

    return running_loss / len(dataloader.dataset)


def train_loop(model, train_loader, val_loader, optimizer, criterion, epochs: int, scheduler=None, verbose: bool = True, device = torch.device("cpu"), save_path = "models/best_model.pt", config=None, feature_cols=None) -> Dict[str, List[float]]:
    """
    Loop di training completo.

    Ritorna:
        history = {
            "train_loss": [...],
            "val_loss": [...]
        }

    Il miglior modello viene salvato su disco nel path `save_path`.
    """
    history = {
        "train_loss": [],
        "val_loss": []
    }

    best_val = float("inf")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_loss = evaluate(
            model, val_loader, criterion, device
        )

        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if verbose:
            print(
                f"Epoch [{epoch:02d}/{epochs}] | "
                f"Train RMSE: {train_loss:.3f} | "
                f"Val RMSE: {val_loss:.3f}"
            )

        if val_loss < best_val:
            best_val = val_loss
            checkpoint = {"model_state_dict": model.state_dict()}
            if config is not None:
                checkpoint["config"] = config
            if feature_cols is not None:
                checkpoint["feature_cols"] = feature_cols
            torch.save(checkpoint, save_path)
            if verbose:
                print(f"Best model saved with RMSE = {val_loss:.3f})")

    return history