from flask import request, render_template, Response
import os
from app import app
import pickle
import tensorflow as tf
import cv2

from app.mtcnn_pytorch import MTCNN

import CONFIG as CONFIG
from app.utils.inference import InfernceSVM, InfernceKeras, inference_SVM, inference_keras

global mtcnn, modelCNN, pca_age, pca_gender, model_age, model_gender, cap

cap = cv2.VideoCapture(0)

# load MTCNN model
mtcnn = MTCNN()

# load CNN model
modelCNN = tf.keras.models.load_model(CONFIG.MODEL_KERAS_PATH)

# load PCA transformer to convert apply PAC on image data
pca_age = pickle.load(open(CONFIG.PCA_AGE_PATH, 'rb'))
pca_gender = pickle.load(open(CONFIG.PCA_GENDER_PATH, 'rb'))

# load the SVM models
model_age = pickle.load(open(CONFIG.MODEL_AGE_PATH, 'rb'))
model_gender = pickle.load(open(CONFIG.MODEL_GENDER_PATH, 'rb'))

@app.route('/')
def index():
    global cap
    if cap.isOpened():
        cap.release()

    return render_template('index.html')

@app.route('/run_svm_model', methods=["GET", "POST"])
def run_svm_model():
    global mtcnn, pca_age, pca_gender, model_age, model_gender, cap
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    if request.method == 'GET':
        cam = InfernceSVM(pca_age, pca_gender, model_age, model_gender, mtcnn, cap)
        print("inference SVM ended")
        return Response(inference_SVM(cam), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/run_keras_model', methods=["GET", "POST"])
def run_keras_model():
    global mtcnn, modelCNN, cap
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    if request.method == 'GET':
        cam = InfernceKeras(modelCNN, mtcnn, cap)
        return Response(inference_keras(cam), mimetype='multipart/x-mixed-replace; boundary=frame')
