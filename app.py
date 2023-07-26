import os
import uuid
import flask
import urllib
from flask import Flask, render_template, request, send_file

from PIL import Image
import torch
import torchvision.models as torch_models
import torchvision.transforms as transforms

model_dir = 'models/my_model'
model = torch_models.shufflenet_v2_x1_5(weights=None)
model.load_state_dict(torch.load(model_dir))
model.eval()

app = Flask(__name__)


def allowed_file(filename):
    allowed_formats = set(['jpg', 'jpeg', 'png'])
    return '.' in filename and filename.rsplit('.', 1)[1] in allowed_formats


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


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/success', methods=['GET', 'POST'])
def success():
    error = ''
    target_img = os.path.join(os.getcwd() , 'static/images')
    if request.method == 'POST':
        if(request.form):
            link = request.form.get('link')
            try :
                resource = urllib.request.urlopen(link)
                unique_filename = str(uuid.uuid4())
                filename = unique_filename+".jpg"
                img_path = os.path.join(target_img, filename)
                output = open(img_path , "wb")
                output.write(resource.read())
                output.close()
                img = filename
                # class_result , prob_result = predict(img_path , model)
                class_result = predict(model, img_path, img_size=224)
                # predictions = {
                #     "class1":class_result[0],
                #     "class2":class_result[1],
                #     "class3":class_result[2],
                #     "prob1": prob_result[0],
                #     "prob2": prob_result[1],
                #     "prob3": prob_result[2],
                # }
                predictions = {
                    "class1":class_result,
                    "prob1": 1
                }
            except Exception as e :
                print(str(e))
                error = 'This image from this site is not accesible or inappropriate input'
            if(len(error) == 0):
                return render_template('success.html', img=img, predictions=predictions)
            else:
                return render_template('index.html', error=error)

        elif (request.files):
            file = request.files['file']
            if file and allowed_file(file.filename):
                file.save(os.path.join(target_img, file.filename))
                img_path = os.path.join(target_img, file.filename)
                img = file.filename
                # class_result, prob_result = predict(img_path, model)
                class_result = predict(model, img_path, img_size=224)
                # predictions = {
                #     "class1":class_result[0],
                #     "class2":class_result[1],
                #     "class3":class_result[2],
                #     "prob1": prob_result[0],
                #     "prob2": prob_result[1],
                #     "prob3": prob_result[2],
                # }
                predictions = {
                    "class1":class_result,
                    "prob1": 1
                }

        else:
                error = "Please upload images of jpg , jpeg and png extension only"
            if(len(error) == 0):
                return  render_template('success.html', img=img, predictions=predictions)
            else:
                return render_template('index.html', error=error)
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)