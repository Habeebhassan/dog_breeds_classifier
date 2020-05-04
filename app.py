from flask import Flask, jsonify, request, render_template
import io
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import json
import requests


app = Flask(__name__)

imagenet_index = json.load(open("/Users/habeebhassan/Documents/DL_Project/cat_dog/imagenet_class_index.json"))
model = models.densenet121(pretrained = True)
model.eval()

def transform_image(img):
    transform = transforms.Compose([transforms.Resize(225), 
    transforms.CenterCrop(224), 
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
    [0.229, 0.224, 0.225])])
    image  = Image.open(io.BytesIO(img))
    return transform(image).unsqueeze(0)


def get_prediction(img):
    tensor = transform_image(img=img)
    output = model.forward(tensor)
    _, y_hat = output.max(1)
    pred_idx = str(y_hat.item())
    return imagenet_index[pred_idx]

@app.route('/predict', methods =['GET','POST'])
def predict():
    if request.method == 'POST':
        file=request.files['file']
        #resp = requests.post("http://localhost:5000/predict", files = {"file":file})
        img = file.read()
        class_id, class_name = get_prediction(img = img)
        return render_template('result.html', class_id=class_id,
        class_name=class_name)
    return render_template('index.html')

        #return jsonify({'class_id': class_id, 'class_name': 'class_name'})



'''with open("/Users/habeebhassan/Documents/DL_Project/cat_dog/beng.jpg", 'rb') as f:
    img = f.read()
    #tensor = transform_image(img=img)
    #print(tensor)
    print(get_prediction(img=img))'''

if __name__ == '__main__':
    app.run()
