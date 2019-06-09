# coding=utf-8
import base64

import cv2
import numpy as np
# Keras
from keras.models import load_model

# Flask utils
from flask import Flask, request, render_template
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = './models/model.h5'

# Load your trained model
model = load_model(MODEL_PATH)
# print('Model loaded. Start serving...')

print('Model loaded. Check http://127.0.0.1:8086/')


colors = np.array([
    [0x65, 0x55, 0xed],
    [0x52, 0xc1, 0x8c],
    [0xad, 0xcf, 0x48],
    [0xc0, 0x87, 0xec],
    [0x40, 0x16, 0x66]], dtype='uint8')


def predict(model, im):
    h, w, _ = im.shape
    inputs = cv2.resize(im, (480, 480))
    inputs = inputs.astype('float32')
    inputs.shape = (1,) + inputs.shape
    inputs /= 255
    mask = model.predict(inputs)
    mask.shape = mask.shape[1:]
    mask = cv2.resize(mask, (w, h))
    mask.shape = h, w, 1
    return mask


def recolor(im, mask, color=(0x40, 0x16, 0x66)):
    color = np.array(color, dtype='float')
    x = im.max(axis=2, keepdims=True)
    x = -np.log(1 - x / 256)
    x_target = -np.log(1 - color.max() / 256)
    x_mean = np.sum(x * mask) / np.sum(mask)
    x = x_target / x_mean * x
    x = 255 * (1 - np.exp(-x))
    color /= color.max()
    color.shape = 1, 1, 3
    im = im * (1 - mask) + x * color * mask
    return im


def model_predict(image, model):
    im = cv2.imdecode(np.fromfile(image, dtype='uint8'), -1)
    # Preprocessing the image
    mask = predict(model, im)
    im = recolor(im, mask)
    _, im = cv2.imencode('.jpg', im)
    im = im.tobytes()
    return im


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def upload():
    file = request.files['file']
    result = model_predict(file, model)
    result = base64.b64encode(result)
    return result


if __name__ == '__main__':
    # Serve the app with gevent
    http_server = WSGIServer(('', 8086), app)
    http_server.serve_forever()
