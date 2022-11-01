from re import A
from flask import Flask, request, jsonify
import gdown
import os
import tensorflow as tf
import cv2
import numpy as np
from skimage import io

models = {}

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    # gdown.download(id='1s8LHcr-slG1M8_72zBtb9Wsh5xHsfnfr', use_cookies=False)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass
    
    @app.route('/', methods=['GET'])
    def index():
        return 'hello world'

    @app.route('/api/prediction', methods=['GET'])
    def prediction():
        new_model = tf.keras.models.load_model('Vit-b32.h5')

        new_model.summary()
        models['Vit-b32'] = new_model
        response = []
        labels = ['Normal', 'Cancer', 'Benign']
        for i in request.json['images']:
            img = io.imread('https://drive.google.com/uc?export=view&id=%s' % i['id'])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (128, 128))
            img = img.astype(np.float32) / 255.
            img = img.reshape(1, 128, 128, 3)
            response.append({
                    "id": i['id'],
                    "name": "1-004.jpg",
                    "prediction": {
                        "label": labels[np.argmax(models['Vit-b32'].predict(img),axis=1)[0]],
                        "value": str(models['Vit-b32'].predict(img)[0][np.argmax(models['Vit-b32'].predict(img),axis=1)[0]])
                        }
                })
        return jsonify(response)
    
    return app
    
    
    
