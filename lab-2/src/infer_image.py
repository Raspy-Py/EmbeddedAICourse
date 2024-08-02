import os
import cv2
import time
import math
import argparse
from utils import Streamer, ImageLoader
from yolo import *

import matplotlib.pyplot as plt

image_path = "../data/images/baseball.jpg"


# Loading model and image
model = Yolo("../data/models/yolov8.rknn")
image = cv2.imread(image_path)

preprocessed_image = model.preprocess(image)
outputs = model.detect(preprocessed_image)      
boxes, classes, scores = model.postprocess(outputs)

image = annotate_image(image, boxes, classes, scores)  

plt.imshow(image)
plt.axis("off")
plt.tight_layout()
plt.savefig(f"../data/images/annotated-baseball.jpg")
plt.close()
