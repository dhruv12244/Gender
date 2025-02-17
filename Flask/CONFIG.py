import os

################ Configuration File #######################

PCA_AGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app/static/models/Dataset__Age__PCA.pkl')
PCA_GENDER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app/static/models/Dataset__Gender__PCA.pkl')

MODEL_AGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app/static/models/Age_Prediction__SVC.pkl')
MODEL_GENDER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app/static/models/Gender_Prediction__SVC.pkl')

MODEL_KERAS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app/static/models/Age_Gender_Prediction__Keras.h5')

assert os.path.exists(PCA_AGE_PATH), f"path doesn't exist: {PCA_AGE_PATH}"

age_model_name = 'Age_Prediction__SVC.pkl'
gender_model_name = 'Gender_Prediction__SVC.pkl'

################# Parameters to Consider #####################

INPUT_SIZE_SVM = (128, 128)
INPUT_SIZE_KERAS_MODEL = (196, 196)


LABEL_MAPPING_AGE = {
    0: "(0, 3)",
    1: "(4, 6)",
    2: "(8, 13)",
    3: "(15, 20)",
    4: "(25, 32)",
    5: "(35, 43)",
    6: "(45, 53)",
    7: "(60, 150)"
}

LABEL_MAPPING_GENDER = {
    0: "female",
    1: "male"
}

###################### Drawing Parmeters ##################################

COLORS = {
    "RED": (0, 0, 255),
    "GREEN": (0, 255, 0),
    "BLUE": (255, 0, 0),
    "PINK": (255, 0, 255),
    "YELLOW": (0, 255, 255),
    "CYAN": (255, 255, 0),
}

FONT_SCALE = 0.6    # must be betweek 0 and 2
TEXT_COLOR = COLORS["YELLOW"]
TEXT_BACKGROUD_COLOR = COLORS["RED"]


###################### DO NOT REMOVE #######################################
PCA_AGE = None
PCA_GENDER = None
MODEL_AGE = None
MODEL_GENDER = None
MODEL_KERAS = None
MTCNN = None

