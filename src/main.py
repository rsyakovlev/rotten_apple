from get_data import get_data
from make_model import make_model
from fit import fit
from utils import seed_everything

import os
import torch

import click


@click.command()
@click.option('-s', '--img_size', default=180)
@click.option('-f', '--data_folder', default="apples")
@click.option('-b', '--train_batch_size', default=8)
@click.option('-l', '--lr', default=0.0001)
@click.option('-i', '--sch_total_iters', default=5)
@click.option('-e', '--epochs', default=10)
@click.option('-o', '--model_dir', default="models/my_model")
def main(img_size,
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

    torch.save(model_shufflenet.state_dict(), os.path.join("../", model_dir))


if __name__ == "__main__":
    main()