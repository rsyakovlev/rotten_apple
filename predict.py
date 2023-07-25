import os

import torch
import torchvision.models as torch_models
import torchvision.transforms as transforms
from PIL import Image

import click


@click.command()
@click.option('-m', '--model_dir', default="./models/my_model")
@click.option('-i', '--img_path', default="./apples/examples")
@click.option('-s', '--img_size', default=224)
@click.option('-e', '--export_to_file', default=1)
def predict(model_dir, img_path, img_size, export_to_file):

    model = torch_models.shufflenet_v2_x1_5(weights=None)
    model.load_state_dict(torch.load(model_dir))
    model.eval()

    img_names = []

    if os.path.isfile(img_path):
        img_names.append(img_path)
    elif os.path.isdir(img_path):
        img_names = [file for file in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, file))]

    og_transform = transforms.Compose([transforms.Resize((img_size, img_size)),
                                       transforms.ToTensor()])
    preds = []
    class_to_classname = {0: "fresh", 1: "rotten"}

    for img_name in img_names:
        image_obj = Image.open(os.path.join(img_path, img_name))
        x = og_transform(image_obj)
        x = x.unsqueeze(0).detach().clone()
        output = model(x)
        _, pred_class_tensor = torch.max(output, 1)
        pred_class = pred_class_tensor.item()
        preds.append(class_to_classname[pred_class])
        print(r'Predicted class for "{0}": {1}'.format(img_name, class_to_classname[pred_class]))

    if export_to_file:
        with open(r"./predictions/preds.txt", "w") as file:
            n = len(img_names)
            for i in range(n):
                img_name = img_names[i]
                pred = preds[i]
                row = f"{img_name}: {pred}\n"
                file.write(row)


if __name__ == "__main__":
    predict()
