import os

import torch
import torch.nn as nn
import torchvision.models as torch_models
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryF1Score

import click


def get_test_data(data_dir="data/test", img_size=224, batch_size=32):

    og_transform = transforms.Compose([transforms.Resize((img_size, img_size)),
                                       transforms.ToTensor()])
    folder_path = os.path.join(".", data_dir)
    test_dataset = datasets.ImageFolder(folder_path,
                                        transform=og_transform)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size)

    return test_dataloader


@click.command()
@click.option('-m', '--model_dir', default="models/my_model")
@click.option('-f', '--data_dir', default="data/test")
@click.option('-s', '--img_size', default=224)
@click.option('-b', '--batch_size', default=32)
@click.option('-e', '--export_to_file', default=1)
def validate(model_dir, data_dir, img_size, batch_size, export_to_file):

    model = torch_models.shufflenet_v2_x1_5(weights=None)
    model.load_state_dict(torch.load(model_dir))
    model.eval()

    total_loss = 0
    num_batches = 0

    correct = 0
    total = 0

    preds = []
    trues = []

    dataloader = get_test_data(data_dir, img_size, batch_size)
    criterion = nn.CrossEntropyLoss()

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

    dataloader_loss = total_loss/num_batches
    dataloader_accuracy = 100*correct/total
    dataloader_f1_score = 100*BinaryF1Score()(torch.tensor(preds), torch.tensor(trues)).item()

    print(f"Loss: {round(dataloader_loss, 4)}. Accuracy: {round(dataloader_accuracy, 4)}%. F1 score: {round(dataloader_f1_score, 4)}%")

    if export_to_file:
        row = f"Loss: {round(dataloader_loss, 4)}. Accuracy: {round(dataloader_accuracy, 4)}%. F1 score: {round(dataloader_f1_score, 4)}%"
        with open("./validation_metrics.txt", "w") as file:
            file.write(row)


if __name__ == "__main__":
    validate()
