import numpy as np
import pandas as pd
import torch

from wav_preproccess import DF_PATH
LABELS = sorted(pd.read_csv(DF_PATH)["label"].unique())
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}
NUM_CLASSES = len(LABELS)

def metrics_and_loss(model, dataloader, loss, device):
    with torch.no_grad():
        was_training = model.training
        model.eval()

        matrix = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.long, device=device)
        hits = 0
        total = 0
        validation_loss = 0.0

        for X, y in dataloader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            y_prob = model(X)
            y_pred = y_prob.argmax(dim=1)

            idx = y * NUM_CLASSES + y_pred
            matrix.view(-1).scatter_add_(
                0, idx, torch.ones_like(idx, dtype=matrix.dtype)
            )

            hits += (y == y_pred).sum().item()
            total += y.size(0)
            validation_loss += loss(y_prob, y).item() * y.size(0)

        if was_training:
            model.train()

        return {
            "accuracy": hits / total,
            "confusion matrix": matrix.cpu(),
            "validation loss": validation_loss / total,
        }
