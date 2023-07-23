import torchvision.models as torch_models
import torch.optim as optim
import torch.nn as nn


def make_model(lr=0.0001, sch_total_iters=5):

    model_shufflenet = torch_models.shufflenet_v2_x1_5(weights='DEFAULT')
    optimizer = optim.Adam(model_shufflenet.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=sch_total_iters)
    criterion = nn.CrossEntropyLoss()

    return model_shufflenet, optimizer, scheduler, criterion