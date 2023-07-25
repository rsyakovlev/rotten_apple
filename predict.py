import os

import torch
import torchvision.models as torch_models
import torchvision.transforms as transforms
from PIL import Image

import click


@click.command()
@click.option('-o', '--model_dir', default="./models/my_model")
@click.option('-f', '--img_path', default="./apples/examples")
@click.option('-s', '--img_size', default=224)
@click.option('-e', '--export_to_file', default=True)
def predict(model_dir, img_path, img_size, export_to_file):

    model = torch_models.shufflenet_v2_x1_5(weights=None)
    model.load_state_dict(torch.load(model_dir))
    model.eval()

    image_paths = []

    if os.path.isfile(img_path):
        image_paths.append(img_path)
    elif os.path.isdir(img_path):
        file_paths = [file for file in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, file))]
        for file_path in file_paths:
            image_paths.append(file_path)

    og_transform = transforms.Compose([transforms.Resize((img_size, img_size)),
                                       transforms.ToTensor()])
    preds = []
    class_to_classname = {0: "fresh", 1: "rotten"}

    for img_path in image_paths:
        image_obj = Image.open(img_path)
        x = og_transform(image_obj)
        x = x.unsqueeze(0).detach().clone()
        output = model(x)
        _, pred_class = torch.max(output, 1)
        preds.append(class_to_classname[pred_class])
        print("Predicted class for {0}: {1}".format(img_path, class_to_classname[pred_class]))

    if export_to_file:
        with open(r"./predictions/preds.txt", "w") as file:
            n = len(image_paths)
            for i in range(n):
                img_path = image_paths[i]
                pred = preds[i]
                row = r"{0}: {1}\n".format(img_path, pred)
                file.write(row)


if __name__ == "__main__":
    predict()