import numpy as np

from base_pipeline import NUM_CLASSES

def accuracy(model, dataloader, device) -> float:
    hits = 0

    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)

        y_pred = model(X).argmax(dim=1)
        if y == y_pred:
            hits += 1

    return hits / len(dataloader)

def confusion_matrix(model, dataloader, device) -> np.array:
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))

    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)

        y_pred = model(X).argmax(dim=1)
        matrix[y, y_pred] += 1

    return matrix
