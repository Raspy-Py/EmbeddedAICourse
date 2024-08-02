import os
import cv2
import time
import math
import argparse
from utils import ImageLoader, Streamer
from yolo import *

frame_count = 0
start_time = time.time()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple script with a 'source' argument")
    parser.add_argument("source", type=str, help="Video source. Either video file or input device.")
    parser.add_argument("--model", type=str, default="../data/models/yolov8n.rknn", help="Path to YOLO model.")
    parser.add_argument("--host", type=str, default="10.42.0.1", help="Address of the host device to stream ouput to.")
    args = parser.parse_args()


    image_loader = ImageLoader(args.source)
    w_, h_ = image_loader.get_res()
    fps = image_loader.get_fps()
    streamer = Streamer(args.host, w=w_, h=h_, fps=fps)
    model = Yolo(args.model)

    for _ in range(len(image_loader)):
        image = image_loader.read()
        
        # Run inference and pre\postprocessing
        preprocessed_image = model.preprocess(image)
        outputs = model.detect(preprocessed_image)      
        boxes, classes, scores = model.postprocess(outputs)

        # Plot detection on the source image
        image = annotate_image(image, boxes, classes, scores)   

        frame_count += 1
        elapsed_time = time.time() - start_time
        start_time = time.time()
        fps = 1.0 / elapsed_time
        fps_str = f'FPS: {fps:.2f}'
        cv2.putText(image, fps_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        streamer.write(image)
