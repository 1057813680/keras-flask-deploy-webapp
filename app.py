# coding=utf-8
import os

import cv2
import scipy.fftpack
import numpy as np
# Keras
from keras.models import load_model

# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = '../easy12306/model.h5'

# Load your trained model
model = load_model(MODEL_PATH)
texts = open('texts.txt').readlines()
# model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# 加载图片分类器
data = np.load('images.npz')
images, labels = data['images'], data['labels']
labels = labels.argmax(axis=1)

print('Model loaded. Check http://127.0.0.1:12306/')


def get_text(img):
    # 得到图像中的文本部分
    return img[3:22, 120:177]


def phash(im):
    im = cv2.resize(im, (32, 32), interpolation=cv2.INTER_CUBIC)
    im = scipy.fftpack.dct(scipy.fftpack.dct(im, axis=0), axis=1)
    im = im[:8, :8]
    med = np.median(im)
    im = im > med
    im = np.packbits(im)
    return im


def _get_imgs(img):
    interval = 5
    length = 67
    for x in range(40, img.shape[0] - length, interval + length):
        for y in range(interval, img.shape[1] - length, interval + length):
            yield img[x:x + length, y:y + length]


def get_imgs(img):
    imgs = []
    for img in _get_imgs(img):
        imgs.append(phash(img))
    return imgs


def model_predict(img_path, model):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    text = get_text(img)
    imgs = get_imgs(img)

    # Preprocessing the image
    text = text / 255.0
    text.shape = (1,) + text.shape + (1,)

    preds = model.predict(text)

    _labels = []
    for img in imgs:
        img.dtype = np.uint64
        img = img[0]

        data = images ^ img
        n = data.shape[0]
        data.dtype = np.uint8
        data = np.unpackbits(data)
        data.shape = (n, -1)
        data = data.sum(axis=1)
        idx = data.argmin()

        label = labels[idx]
        _labels.append(label)

    return preds, _labels


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds, imgs = model_predict(file_path, model)

        # Process your result for human
        pred_class = preds.argmax(axis=-1)  # Simple argmax
        result = texts[pred_class[0]]       # Convert to string
        result = f'text: {result}, images: '
        result = result + ', '.join(texts[img] for img in imgs)
        return result
    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 12306), app)
    http_server.serve_forever()
