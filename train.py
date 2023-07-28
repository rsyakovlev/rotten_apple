import os
from collections import namedtuple
from utils import seed_everything

import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.models as torch_models
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryF1Score

import click


def get_data(data_folder="data", img_size=224, train_batch_size=8, test_batch_size=32):

    og_transform = transforms.Compose([transforms.Resize((img_size, img_size)),
                                       transforms.ToTensor()])

    folder_path = os.path.join(".", data_folder)
    train_folder_path = os.path.join(folder_path, "train")
    test_folder_path = os.path.join(folder_path, "test")

    train_dataset = datasets.ImageFolder(train_folder_path,
                                         transform=og_transform)
    test_dataset = datasets.ImageFolder(test_folder_path,
                                        transform=og_transform)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=train_batch_size,
                                  shuffle=True)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=test_batch_size,
                                 shuffle=True)

    return train_dataloader, test_dataloader


def make_model(lr=0.0001, sch_total_iters=5):

    model_shufflenet = torch_models.shufflenet_v2_x1_5(weights='DEFAULT')
    optimizer = optim.Adam(model_shufflenet.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=sch_total_iters)
    criterion = nn.CrossEntropyLoss()

    return model_shufflenet, optimizer, scheduler, criterion


def train_epoch(model,
                train_dataloader,
                optimizer,
                criterion):

    model.train()
    total_loss = 0
    num_batches = 0
    correct = 0
    total = 0

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

    epoch_loss = total_loss/num_batches
    epoch_accuracy = 100*correct/total

    metrics = {'loss': round(epoch_loss, 4),
               'accuracy': round(epoch_accuracy, 4)}

    return metrics


def evaluate(model,
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


def fit(model,
        optimizer,
        scheduler,
        criterion,
        epochs,
        train_dataloader,
        test_dataloader):

    LossInfo = namedtuple('LossInfo', ['train_epoch_losses', 'train_epoch_accuracy', 'test_epoch_losses', 'test_epoch_accuracy'])

    epoch_train_losses = []
    epoch_train_accuracy = []

    epoch_test_losses = []
    epoch_test_accuracy = []

    for epoch in range(1, epochs+1):
        print(f"Train Epoch: {epoch}")
        train_metrics = train_epoch(model=model,
                                    train_dataloader=train_dataloader,
                                    optimizer=optimizer,
                                    criterion=criterion)
        epoch_train_losses.append(train_metrics['loss'])
        epoch_train_accuracy.append(train_metrics['accuracy'])

        print(f"Validation Epoch: {epoch}")
        with torch.no_grad():
            validation_metrics = evaluate(model=model,
                                          criterion=criterion,
                                          dataloader=test_dataloader)
        epoch_test_losses.append(validation_metrics['loss'])
        epoch_test_accuracy.append(validation_metrics['accuracy'])

        if validation_metrics['accuracy'] > 99.99:
            break
        if (len(epoch_test_accuracy) > 2 and (epoch_test_accuracy[-3] > epoch_test_accuracy[-2]
                                              and epoch_test_accuracy[-2] > epoch_test_accuracy[-1])):
            break
        print()
        print()
        scheduler.step()

    return LossInfo(epoch_train_losses, epoch_train_accuracy, epoch_test_losses, epoch_test_accuracy)


@click.command()
@click.option('-s', '--img_size', default=224)
@click.option('-f', '--data_folder', default="data")
@click.option('-b', '--train_batch_size', default=8)
@click.option('-l', '--lr', default=0.0001)
@click.option('-i', '--sch_total_iters', default=5)
@click.option('-e', '--epochs', default=10)
@click.option('-m', '--model_dir', default="models/my_model")
def train(img_size,
          data_folder,
          train_batch_size,
          lr,
          sch_total_iters,
          epochs,
          model_dir,
          seed=123):

    seed_everything(seed)

    train_dataloader, test_dataloader = get_data(data_folder=data_folder,
                                                 img_size=img_size,
                                                 train_batch_size=train_batch_size)
    model_shufflenet, optimizer, scheduler, criterion = make_model(lr=lr,
                                                                   sch_total_iters=sch_total_iters)

    info = fit(model=model_shufflenet,
               optimizer=optimizer,
               scheduler=scheduler,
               criterion=criterion,
               epochs=epochs,
               train_dataloader=train_dataloader,
               test_dataloader=test_dataloader)

    torch.save(model_shufflenet.state_dict(), os.path.join(".", model_dir))


if __name__ == "__main__":
    train()
