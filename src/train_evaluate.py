# from torchmetrics import Accuracy, F1Score, Precision, Recall
import torch
import os

def evaluate(model, dataloader, num_classes, loss, device):
    with torch.no_grad():
        model.eval()
        model.to(device)

        # acc = Accuracy(task="multiclass", num_classes=num_classes).to(device)
        # f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device)
        # prec = Precision(task="multiclass", num_classes=num_classes, average="macro").to(device)
        # rec = Recall(task="multiclass", num_classes=num_classes, average="macro").to(device)

        acc = 0 
        f1 = 0
        prec = 0
        rec = 0


        validation_loss = 0.0
        total = 0

        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            probs = model(X)
            # preds = probs.argmax(dim=1)

            validation_loss += loss(probs, y).item()*y.size(0)
            total += y.size(0)

            # acc.update(preds,y)
            # f1.update(preds,y)
            # prec.update(preds,y)
            # rec.update(preds,y)

        validation_loss /= total

        return {
            "accuracy": acc,
            "f1": f1,
            "prec": prec,
            "rec": rec,
            "validation loss": validation_loss
        }

def save_model(model, optimizer, epoch, best_metrics, best_val_loss, save_path):
    dir = os.path.dirname(save_path)
    os.makedirs(dir, exist_ok=True)

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_metrics": best_metrics,
        "best_val_loss": best_val_loss,
    }, save_path)

    print(f"saved model to: {save_path}")

def load_model(model, optimizer, load_path, device):

    dct = torch.load(load_path, map_location=device)

    model.load_state_dict(dct["model_state_dict"])
    optimizer.load_state_dict(dct["optimizer_state_dict"])

    start_epoch = dct["epoch"]
    best_val_loss = dct["best_val_loss"]
    best_metrics = dct["best_metrics"]

    print(f"model path: {load_path}")
    print(f"epoch: {start_epoch}, best_val_loss = {best_val_loss:.6f}")

    return start_epoch, best_val_loss, best_metrics

def train(model, epochs: int, train_loader, validate_loader, optimizer, loss, num_classes, device, eps, epochs_to_wait, save_path, ask = True, load_path = None):
    model.to(device)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_metrics = None
    start_epoch = 0

    if load_path is not None:
        start_epoch, best_val_loss, best_metrics = load_model(model, optimizer,load_path, device)

    epoch = start_epoch
    while(epoch<epochs):
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

        val_loss = metrics_loss["validation loss"]

        epoch+=1

        if val_loss < best_val_loss - eps:
            best_val_loss = val_loss
            best_metrics = metrics_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement+=1

        if epochs_without_improvement >= epochs_to_wait:
            if ask:
                ans = input("should continue? yes or no")

                if (ans == "yes"):
                    epochs_without_improvement = 0
                    continue
                else:
                    break
            else:
                break

    print(f'finished, best_val_loss: {best_val_loss}')
    print(f'best metrics: {best_metrics}')

    save_model(model, optimizer, epoch, best_metrics, best_val_loss, save_path)



