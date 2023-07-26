from PIL import Image
import torch
import torchvision.models as torch_models
import torchvision.transforms as transforms

model_dir = 'models/my_model'
model = torch_models.shufflenet_v2_x1_5(weights=None)
model.load_state_dict(torch.load(model_dir))
model.eval()

def predict(model, img_name, img_size=224):

    og_transform = transforms.Compose([transforms.Resize((img_size, img_size)),
                                       transforms.ToTensor()])
    class_to_classname = ["fresh", "rotten"]

    image_obj = Image.open(img_name)
    x = og_transform(image_obj)
    x = x.unsqueeze(0).detach().clone()
    output = model(x)
    _, pred_class_tensor = torch.max(output, 1)
    pred_class = pred_class_tensor.item()
    pred = class_to_classname[pred_class]
    print(r'Predicted class for "{0}": {1}'.format(img_name, class_to_classname[pred_class]))