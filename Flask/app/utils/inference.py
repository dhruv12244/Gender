import cv2
import numpy as np
from PIL import Image

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
np.set_printoptions(suppress=True)

from app.utils import utils
import CONFIG

class InfernceSVM():
    def __init__(self, pca_age, pca_gender, model_age, model_gender, mtcnn, cap):
        self.stream = cap
        self.pca_age = pca_age
        self.pca_gender = pca_gender
        self.age_model = model_age
        self.gender_model = model_gender
        self.face_detector = mtcnn

    # def __del__(self):
    #     self.stream.stop()

    def inference(self):    
        # frame = self.stream.read()
        ret, frame = self.stream.read()
        frame = cv2.putText(frame, "Running SVM Models", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        img_pil = Image.fromarray(frame)
        img_w, img_h = img_pil.size
        
        # Get cropped and prewhitened image tensor
        boxes, probs, points = self.face_detector.detect(img_pil, landmarks=True)
        
        # skip if no face is detected
        if boxes is not None:
            for box, prob in zip(boxes, probs):
                if prob < 0.9:
                    continue

                # if there are multiple faces, pick the one with maximum box area
                xmin, ymin, xmax, ymax = [int(v) for v in box]
                
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(img_w, xmax)
                ymax = min(img_h, ymax)
            
                # nose_landmark = (int(landmarks[2][0]), int(landmarks[2][1]))      

                # crop the face# crop the face
                img = utils.cropPhoto(frame, [xmin, ymin, xmax, ymax], size=0.3)
                # img = frame[ymin:ymax, xmin:xmax]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Preprocess 1.
                # resize img to a fixed 
                img = cv2.resize(img, CONFIG.INPUT_SIZE_SVM, interpolation=cv2.INTER_AREA)

                # normalize image
                norm_image = (img - np.min(img)) / (np.max(img) - np.min(img))
                reshaped_image = norm_image.reshape((1, -1))
                pca_image_age = self.pca_age.transform(reshaped_image)
                pca_image_gender = self.pca_gender.transform(reshaped_image)

                predicted_age = CONFIG.LABEL_MAPPING_AGE[self.age_model.predict(pca_image_age)[0]]
                predicted_gender = CONFIG.LABEL_MAPPING_GENDER[self.gender_model.predict(pca_image_gender)[0]]
                label = f"Gender: {predicted_gender} | Age: {predicted_age}"

                # drawing bounding box around sign
                frame = utils.draw(frame, xmin, ymin, xmax, ymax, label)

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()


def inference_SVM(camera):
    while True:
        frame = camera.inference()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


class InfernceKeras():
    def __init__(self, model, mtcnn, cap):
        self.stream = cap
        self.face_detector = mtcnn
        self.model = model

    # def __del__(self):
    #     self.stream.stop()

    def inference(self):    
        ret, frame = self.stream.read()
        frame = cv2.putText(frame, "Running CNN Model", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        # frame = cv2.resize(frame, (0, 0), fx=self.resize_scale, fy=self.resize_scale)

        img_pil = Image.fromarray(frame)
        img_w, img_h = img_pil.size
        
        # Get cropped and prewhitened image tensor
        boxes, probs, points = self.face_detector.detect(img_pil, landmarks=True)
        
        # skip if no face is detected
        if boxes is not None:
            for box, prob in zip(boxes, probs):
                if prob < 0.9:
                    continue

                # if there are multiple faces, pick the one with maximum box area
                xmin, ymin, xmax, ymax = [int(v) for v in box]
                
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(img_w, xmax)
                ymax = min(img_h, ymax)

                # crop the face# crop the face
                img = utils.cropPhoto(frame, [xmin, ymin, xmax, ymax], size=0.5)
                # img = frame[ymin:ymax, xmin:xmax]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Preprocess 1.
                # resize img to a fixed 
                img = cv2.resize(img, CONFIG.INPUT_SIZE_KERAS_MODEL, interpolation=cv2.INTER_AREA)

                # normalize image
                norm_image = (img - np.min(img)) / (np.max(img) - np.min(img))

                images_batch = np.expand_dims(norm_image, axis=0)
                predictions_age, predictions_gender = self.model.predict(images_batch, verbose=0)

                prediction_age = CONFIG.LABEL_MAPPING_AGE[np.argmax(predictions_age, axis=1)[0]]
                prediction_gender = CONFIG.LABEL_MAPPING_GENDER[np.argmax(predictions_gender, axis=1)[0]]
                
                label = f"Age: {prediction_age} | Gender: {prediction_gender}"
                # drawing bounding box around sign
                frame = utils.draw(frame, xmin, ymin, xmax, ymax, label)

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()


def inference_keras(camera):
    while True:
        frame = camera.inference()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
