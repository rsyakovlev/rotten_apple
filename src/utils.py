import matplotlib.pyplot as plt
import random
import os
import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def plot_metrics_graph(info, model='Model', show_numbers=False):

    x = list(range(len(info.test_epoch_accuracy)))
    x_ticks = list(range(1, len(info.test_epoch_accuracy)+1))

    fig, ax = plt.subplots(1, 2, figsize=(16,6))

    ax[0].grid()
    ax[0].plot(info.train_epoch_accuracy, color='blue', label='train')
    ax[0].plot(info.test_epoch_accuracy, color='red', label='test')
    ax[0].legend()
    ax[0].set_title('Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy')
    if show_numbers:
        for index in x:
            ax[0].text(x[index], info.test_epoch_accuracy[index], info.test_epoch_accuracy[index], size=10)
    ax[0].set_xticks(x, x_ticks, size=10)

    ax[1].grid()
    ax[1].plot(info.train_epoch_losses, color='blue', label='train')
    ax[1].plot(info.test_epoch_losses, color='red', label='test')
    ax[1].legend()
    ax[1].set_title('Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    if show_numbers:
        for index in x:
            ax[1].text(x[index], info.test_epoch_losses[index], info.test_epoch_losses[index], size=10)
    ax[1].set_xticks(x, x_ticks, size=10)

    fig.suptitle(model, fontsize=16)


def show_model_predictions(model, images_num, dataloader):

    assert images_num <= 32

    fig, ax = plt.subplots(1, images_num, figsize=(12,8))

    for images, labels in dataloader:
        _, preds = torch.max(model(images), 1)
        for i in range(images_num):
            ax[i].imshow(images[i].view(3, -1).T.reshape((224, 224, 3)).numpy())
            ax[i].axis('off')
            ax[i].set_title(f"true: {labels[i].item()}; pred: {preds[i].item()}", fontsize=10)
        break


def count_params_num(model):
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in trainable_params])