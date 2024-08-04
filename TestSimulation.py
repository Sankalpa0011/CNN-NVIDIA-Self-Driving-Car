print("Setting UP")
import os
import eventlet.wsgi
import socketio
import eventlet
import numpy as np
from flask import Flask
from keras.models import load_model
from keras.losses import MeanSquaredError
import base64
from io import BytesIO
from PIL import Image
import cv2
import socketio.server


sio = socketio.Server()
app = Flask(__name__) # "__main__"
maxSpeed = 10



def preProcess(img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255.0
    return img



# 3. Importing image
@sio.on("telemetry")
def telemetry(sid, data):
    speed = float(data["speed"])
    image = Image.open(BytesIO(base64.b64decode(data["image"])))
    image = np.asarray(image)
    image = preProcess(image)
    image = np.array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    steering = float(model.predict(image))
    throttle = 1.0 - speed / maxSpeed
    print("{} {} {}".format(steering, throttle, speed))
    sendControl(steering, throttle)



# 2. Connecting and sending commands to simulator at the begining
@sio.on("connect")
def coonect(sid, environ):
    print("Connected")
    sendControl(0, 0)  #(0, 0) = (steering, speed)



# 4. Control
def sendControl(steering, throttle):
    sio.emit("steer", data={
        "steering_angle": steering.__str__(),
        "throttle": throttle.__str__()
    })



# 1. Load the model
if __name__ == "__main__":
    # Loading the model with custom objects
    custom_objects = {'mse': MeanSquaredError()}
    model = load_model("self_drive_model.h5", custom_objects=custom_objects)
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(("", 4567)), app)