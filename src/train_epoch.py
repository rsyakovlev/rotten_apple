import sys
from tqdm.notebook import tqdm
import torch


def train_epoch(model,
                train_dataloader,
                optimizer,
                criterion):

    model.train()
    total_loss = 0
    num_batches = 0
    correct = 0
    total = 0

    with tqdm(total=len(train_dataloader), file=sys.stdout) as prbar:
        for images, labels in train_dataloader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            _, pred_classes = torch.max(outputs, 1)
            batch_accuracy = (pred_classes == labels).float().mean().item() * 100
            correct += (pred_classes == labels).sum().item()
            total += labels.size(0)

            total_loss += loss.item()
            num_batches += 1

            prbar.set_description(f"Loss: {round(loss.item(), 4)}. Accuracy: {round(batch_accuracy, 4)}%")
            prbar.update(1)

    epoch_loss = total_loss/num_batches
    epoch_accuracy = 100*correct/total

    prbar.set_description(f"Loss: {round(epoch_loss, 4)}. Accuracy: {round(epoch_accuracy, 4)}%")
    prbar.update(1)

    metrics = {'loss': round(epoch_loss, 4),
               'accuracy': round(epoch_accuracy, 4)}

    return metrics