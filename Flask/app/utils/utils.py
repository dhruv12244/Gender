import cv2
import pickle
import tensorflow as tf
from app.mtcnn_pytorch import MTCNN

import CONFIG

def draw(image, x_min, y_min, x_max, y_max, label, offset=0):
    result_image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), CONFIG.TEXT_BACKGROUD_COLOR, 3)

    # For the text background
    # Finds space required by the text so that we can put a background with that amount of width.
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_thickness = 2
    (text_w, text_h), _ = cv2.getTextSize(label, font, CONFIG.FONT_SCALE, font_thickness)

    # checking if bounding box end going outside the image
    if label is not None:
        if (x_min + text_w) > result_image.shape[1]:
            differance = (x_min + text_w) - result_image.shape[1]
            x, y = x_min - differance, y_min - offset
            cv2.rectangle(image, (x, y), (x + text_w, y + text_h), CONFIG.TEXT_BACKGROUD_COLOR, -1)
            result_image = cv2.putText(result_image, label, (x, y + text_h  - 1), font, CONFIG.FONT_SCALE, CONFIG.TEXT_COLOR, font_thickness)
        else:
            x, y = x_min, y_min - offset
            cv2.rectangle(image, (x, y), (x + text_w, y + text_h), CONFIG.TEXT_BACKGROUD_COLOR, -1)
            result_image = cv2.putText(result_image, label, (x, y + text_h  - 1), font, CONFIG.FONT_SCALE, CONFIG.TEXT_COLOR, font_thickness)
    return result_image

# Loading Dataset
def cropPhoto(image, box, size=0.1):
    [x1, y1, x2, y2] = box
    box_w, box_h = (x2 - x1), (y2 - y1)
    img_h, img_w = image.shape[:2]
    
    x1c = int(max(0, x1 - (box_w * size)))
    y1c = int(max(0, y1 - (box_h * size)))
    x2c = int(min(img_w, x2 + (box_w * size)))
    y2c = int(min(img_h, y2 + (box_h * size)))
    crop = image[y1c:y2c, x1c:x2c]
    return crop

def loadModels():
    CONFIG.PCA_AGE = pickle.load(open(CONFIG.PCA_AGE_PATH, 'rb'))
    CONFIG.PCA_GENDER = pickle.load(open(CONFIG.PCA_GENDER_PATH, 'rb'))
    
    # load the models from disk
    CONFIG.MODEL_AGE = pickle.load(open(CONFIG.MODEL_AGE_PATH, 'rb'))
    CONFIG.MODEL_GENDER = pickle.load(open(CONFIG.MODEL_GENDER_PATH, 'rb'))
    
    # load MTCNN model
    CONFIG.MTCNN = MTCNN()
    
    # load keras model
    CONFIG.MODEL_KERAS = tf.keras.models.load_model(CONFIG.MODEL_KERAS_PATH)
