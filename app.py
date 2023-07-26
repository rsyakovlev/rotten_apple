import os

from PIL import Image
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
    og_transform = transforms.Compose([transforms.Resize((img_size, img_size)),
                                       transforms.ToTensor()])
    image_obj = Image.open(img_path)
    x = og_transform(image_obj)
    x = x.unsqueeze(0).detach().clone()
    output = model(x)
    _, pred_class_tensor = torch.max(output, 1)
    pred_class = pred_class_tensor.item()
    pred = class_to_classname[pred_class]

    return pred


def is_fileformat_allowed(filename):
    allowed_formats = {'jpg', 'jpeg', 'png'}
    return '.' in filename and filename.rsplit('.', 1)[1] in allowed_formats


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template("home.html")


@app.route("/prediction", methods = ['GET','POST'])
def prediction():

    if request.method == 'POST':
        file = request.files['file']
        filename = file.filename
        if not is_fileformat_allowed(filename):
            raise Exception
        # file_path = os.path.join(r'C:/Users/nEW u/Flask/static/', filename)
        img_path = os.path.join('./static', filename)
        file.save(img_path)
        model = get_model(model_dir='models/my_model')
        pred = predict(model=model, img_path=img_path, img_size=224)
        print(pred)

    return render_template('prediction.html', pred=pred, user_image=img_path)


if __name__ == "__main__":
    app.run()