from torchmetrics import Accuracy, F1Score, Precision, Recall
import torch

def evaluate(model, dataloader, num_classes, loss, device):
    with torch.no_grad():
        model.eval()
        model.to(device)

        acc = Accuracy(task="multiclass", num_classes=num_classes).to(device)
        f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device)
        prec = Precision(task="multiclass", num_classes=num_classes, average="macro").to(device)
        rec = Recall(task="multiclass", num_classes=num_classes, average="macro").to(device)

        validation_loss = 0.0
        total = 0

        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            probs = model(X)
            preds = probs.argmax(dim=1)

            validation_loss += loss(probs, y).item()*y.size(0)
            total += y.size(0)

            acc.update(preds,y)
            f1.update(preds,y)
            prec.update(preds,y)
            rec.update(preds,y)

        validation_loss /= total

        return {
            "accuracy": acc.compute().item(),
            "f1": f1.compute().item(),
            "prec": prec.compute().item(),
            "rec": rec.compute().item(),
            "validation loss": validation_loss
        }


def train(model, epochs: int, train_loader, validate_loader, optimizer, loss, num_classes, device):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        print(f"Epoch {epoch}")

        train_loss = 0.0
        total = 0

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            output = model(X)
            loss_value = loss(output, y)

            loss_value.backward()
            optimizer.step()

            train_loss += loss_value.item()*y.size(0)
            total += y.size(0)

        train_loss /= total
        print(f"Train loss: {train_loss}")

        metrics_loss = evaluate(model, validate_loader, num_classes, loss, device)
        
        print(metrics_loss)


