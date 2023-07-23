import torch
from collections import namedtuple
from train_epoch import train_epoch
from validate import validate


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
            validation_metrics = validate(model=model,
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

