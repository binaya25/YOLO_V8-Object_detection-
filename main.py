1. Modes - train, val , prediction

2. Tasks - detection, classify, segmentation

3. data = data.yaml, data_sets , data_folder


import ultralytics
from roboflow import Roboflow
from ultralytics import YOLO
from IPython.display import Image


# Downloading labaeled datasets from roboflow (Best option is to download into local directory and skip this step)
from roboflow import Roboflow
rf = Roboflow(api_key="vk50M712VnDwYHzZJ6IX")
project = rf.workspace("leo-ueno").project("people-detection-o4rdr")
version = project.version(8)
dataset = version.download("yolov8")


# Training model (train, val , prediction)
model = YOLO("yolov8s.pt")
model.train(data="./people/data.yaml", epochs = 5)


# Object detection model to make prdeictions on live video
model = YOLO('yolov8n.pt')
model.predict(source=0, imgsz= 640, conf= 0.6, show= True)

