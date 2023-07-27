import os

import cv2
import torch
import torchvision.models as torch_models
import torchvision.transforms as transforms

from flask import Flask, render_template, request


def get_model(model_dir='models/my_model'):
    model = torch_models.shufflenet_v2_x1_5(weights=None)
    model.load_state_dict(torch.load(model_dir))
    model.eval()
    return model


def predict(model, img_path, img_size=224):
    class_to_classname = ["fresh", "rotten"]
    # og_transform = transforms.Compose([transforms.Resize((img_size, img_size)),
    #                                    transforms.ToTensor()])
    # image_obj = Image.open(img_path)
    # x = og_transform(image_obj)
    image_obj = cv2.imread(img_path)
    image_obj = cv2.resize(src=image_obj, dsize=(img_size, img_size), interpolation=cv2.INTER_AREA)
    image_obj = cv2.cvtColor(image_obj, cv2.COLOR_BGR2RGB)
    transform = transforms.ToTensor()
    x = transform(image_obj)
    x = x.unsqueeze(0).detach().clone()
    output = model(x)
    _, pred_class_tensor = torch.max(output, 1)
    pred_class = pred_class_tensor.item()
    pred = class_to_classname[pred_class]

    return pred


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template("home.html")


@app.route("/prediction", methods = ['GET','POST'])
def prediction():

    if request.method == 'POST':
        file = request.files['file']
        filename = file.filename
        img_path = os.path.join('./static', filename)
        file.save(img_path)
        model = get_model(model_dir='models/my_model')
        pred = predict(model=model, img_path=img_path, img_size=224)
        print(pred)

    return render_template('prediction.html', pred=pred, user_image=img_path)


if __name__ == "__main__":
    app.run()