# -*- coding: utf-8 -*-

from flask import Flask, render_template, request, session, redirect, url_for, flash
import os
import tensorflow as tf
import keras
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
from skimage import transform
import math
import os, shutil
from keras.utils.generic_utils import CustomObjectScope
import keras.applications

UPLOAD_FOLDER = './flask app/assets/images'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
# Create Database if it doesnt exist

app = Flask(__name__,static_url_path='/assets',
            static_folder='./flask app/assets', 
            template_folder='./flask app')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def root():
   return render_template('index.html')

@app.route('/index.html')
def index():
   return render_template('index.html')

@app.route('/contact.html')
def contact():
   return render_template('contact.html')

@app.route('/news.html')
def news():
   return render_template('news.html')

@app.route('/about.html')
def about():
   return render_template('about.html')

@app.route('/cancer.html')
def cancer():
   return render_template('cancer.html')

@app.route('/faqs.html')
def faqs():
   return render_template('faqs.html')

@app.route('/prevention.html')
def prevention():
   return render_template('prevention.html')

@app.route('/upload.html')
def upload():
   return render_template('upload.html')

@app.route('/upload_chest.html')
def upload_chest():
   return render_template('upload_chest.html')

@app.route('/uploaded_chest', methods = ['POST', 'GET'])
def uploaded_chest():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file found')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            # filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'upload_chest.jpg'))


    #image upload
    classes =['Bacterial Pneumonia', 'COVID', 'Normal', 'Tuberculosis', 'Viral Pneumonia']

    path = "./flask app/assets/images/upload_chest.jpg"

    #for vgg model
    img = tf.keras.preprocessing.image.load_img(path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) 
    vgg_chest = load_model('models/Xray_Model.h5')

    #predicting with vgg model
    predictions = vgg_chest.predict(img_array)
    print("Prediction: ", classes[np.argmax(predictions)], f"{predictions[0][np.argmax(predictions)]*100}%")
    result_vgg = predictions[0][np.argmax(predictions)]*100
    result_vgg_class =  classes[np.argmax(predictions)]
    print(result_vgg_class)

    request_data = {
      'vgg_chest_predclass':result_vgg_class,
      'vgg_chest_pred' : result_vgg
   }

    return render_template('results_chest.html', **request_data)


if __name__ == '__main__':
   app.secret_key = ".."
   app.run()