from flask import Flask
from app.utils.utils import loadModels

app = Flask(__name__)
app.secret_key = "hhfsdfhs00390dsafjsdafkh30940"
# cors = CORS(app, resources={r"/*": {"origins": "*"}})

# print("INFO: Loading Models...")
# loadModels()

from app import views
