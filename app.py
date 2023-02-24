
import os

import numpy as np

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from flask import Flask, redirect, render_template, url_for, request

app=Flask(__name__)

model_path='Image_Classification/vgg19.h5'

# model = load_model(model_path)
# model.make_predict_function()

# preprocessing and prediction function
def model_predict(img_path,model):
    img= image.load(img_path, target_size=(224,224))
    x= image.img_to_array(img)
    x=np.expand_dims(x,axis=0)
    x=preprocess_input(x)
    preds = model.predict(x)
    return preds

@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')

# @app.route('/predict',methods=["GET","POST"])
# def upload():
#     if request.method=="POST":
#         f=request.file['file']
#         base_path = os.path.dirname(__file__)
#         file_path= os.path.join(base_path,'uploads',secure_filename(f.filename))
#         f.save(file_path)

#         # start prediction
#         pred= model_predict(file_path,model)


if __name__=='__main__':
    app.run(debug=True)