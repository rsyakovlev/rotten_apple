import os
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader


def get_data(data_folder="apples", img_size=224, train_batch_size=8, test_batch_size=32):
    og_transform = transforms.Compose([transforms.Resize((img_size, img_size)),
                                       transforms.ToTensor()])

    folder_path = os.path.join("../", data_folder)
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