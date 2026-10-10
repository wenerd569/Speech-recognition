# from torchmetrics import Accuracy, F1Score, Precision, Recall
import math
import os

import seaborn as sb
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from common.metrics import metrics_and_loss


def evaluate(model, dataloader, num_classes, loss, device):
    with torch.no_grad():
        model.eval()
        model.to(device)

        # acc = Accuracy(task="multiclass", num_classes=num_classes).to(device)
        # f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device)
        # prec = Precision(task="multiclass", num_classes=num_classes, average="macro").to(device)
        # rec = Recall(task="multiclass", num_classes=num_classes, average="macro").to(device)

        # pyrefly: ignore [missing-argument]
        return metrics_and_loss(model, dataloader, loss, device, num_classes)

def save_model(model, optimizer, scheduler, epoch, best_metrics, best_val_loss, save_path):
    dir = os.path.dirname(save_path)
    os.makedirs(dir, exist_ok=True)

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_metrics": best_metrics,
        "best_val_loss": best_val_loss,
    }, save_path)

    print(f"saved model to: {save_path}")

def load_model(model, optimizer, scheduler, load_path, device):

    dct = torch.load(load_path, map_location=device)

    model.load_state_dict(dct["model_state_dict"])
    optimizer.load_state_dict(dct["optimizer_state_dict"])
    scheduler.load_state_dict(dct["scheduler_state_dict"])

    start_epoch = dct["epoch"]
    best_val_loss = dct["best_val_loss"]
    best_metrics = dct["best_metrics"]

    print(f"model path: {load_path}")
    print(f"epoch: {start_epoch}, best_val_loss = {best_val_loss:.6f}")

    return start_epoch, best_val_loss, best_metrics

def step_decay_mult(epoch):
    drop = 0.4
    epochs_drop = 10.0
    return max(math.pow(drop, math.floor(epoch / epochs_drop)), 0.04)

def train(model, epochs: int, train_loader, validate_loader, optimizer, loss, scheduler, num_classes, device, eps, epochs_to_wait, save_path, ask = False, load_path = None):
    model.to(device)

    best_val_loss = float("inf")
    best_val_acc = float("-inf")
    epochs_without_improvement = 0
    best_metrics = None
    start_epoch = 0

    if load_path is not None:
        start_epoch, best_val_loss, best_metrics = load_model(model, optimizer, scheduler, load_path, device)

    epoch = start_epoch

    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"tensorboard log_dir: {log_dir}")
    global_step = start_epoch * len(train_loader)

    while(epoch<epochs):
        model.train()
        print(f"Epoch {epoch}")

        train_loss = 0.0
        total = 0

        for X, y in tqdm(train_loader):
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            output = model(X)
            loss_value = loss(output, y)
            loss_value.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))
            optimizer.step()

            train_loss += loss_value.item()*y.size(0)
            total += y.size(0)

            if global_step % 100 == 0:
                writer.add_scalar("Loss/train_batch", loss_value.item(), global_step)
                writer.add_scalar("GradNorm/train_batch", grad_norm.item(), global_step)
            global_step+=1
        
        train_loss /= total
        print(f"Train loss: {train_loss}")

        metrics_loss = evaluate(model, validate_loader, num_classes, loss, device)
        
        print(f'accuracy: {metrics_loss["accuracy"]}, validation loss: {metrics_loss["validation loss"]}')

        val_loss = metrics_loss["validation loss"]
        val_acc = metrics_loss["accuracy"]
        confusion_matrix = metrics_loss['confusion matrix']

        writer.add_scalar("Loss/train_epoch", train_loss, epoch)
        writer.add_scalar("Loss/val_epoch", val_loss, epoch)
        writer.add_scalar("Accuracy/val_epoch", val_acc, epoch)

        # pyrefly: ignore [bad-argument-type]
        writer.add_figure("confusion matrix", sb.heatmap(confusion_matrix, annot=True).get_figure(), epoch)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f'lr after epoch {epoch}: {current_lr}')

        if val_acc > best_val_acc + eps:
            best_val_acc = val_acc
            best_metrics = metrics_loss
            best_val_loss = val_loss
            print('model is better, saving')
            save_model(model, optimizer, scheduler, epoch, best_metrics, best_val_loss, save_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement+=1

        epoch+=1

        if epochs_without_improvement >= epochs_to_wait:
            if ask:
                ans = input("should continue? yes or no ")

                if (ans.strip().lower() == "yes"):
                    epochs_without_improvement = 0
                    continue
                else:
                    break
            #else:
            #   break

    writer.close()

    print(f'finished, best_val_loss: {best_val_loss}')
    print(f'best metrics: {best_metrics}')


