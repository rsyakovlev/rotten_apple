import torch
from torchmetrics.classification import BinaryF1Score


def validate(model,
             criterion,
             dataloader):

    model = model.eval()
    total_loss = 0
    num_batches = 0

    correct = 0
    total = 0

    preds = []
    trues = []

    for images, labels in dataloader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        _, pred_classes = torch.max(outputs, 1)
        batch_accuracy = (pred_classes == labels).float().mean().item() * 100

        correct += (pred_classes == labels).sum().item()
        total += labels.size(0)

        preds += pred_classes.tolist()
        trues += labels.tolist()

        total_loss += loss.item()
        num_batches += 1

    epoch_loss = total_loss/num_batches
    epoch_accuracy = 100*correct/total
    epoch_f1_score = 100*BinaryF1Score()(torch.tensor(preds), torch.tensor(trues)).item()

    print(f"Loss: {round(epoch_loss, 4)}. Accuracy: {round(epoch_accuracy, 4)}%. F1 score: {round(epoch_f1_score, 4)}%")

    metrics = {'loss': round(epoch_loss, 4),
               'accuracy': round(epoch_accuracy, 4),
               'F1 score': round(epoch_f1_score, 4)}

    return metrics