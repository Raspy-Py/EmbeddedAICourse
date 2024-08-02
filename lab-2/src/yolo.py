import os
import cv2
import torch
import numpy as np
# ONNX
import onnxruntime as ort
# RKNN
from rknnlite.api import RKNNLite
# TFLITE
import tflite_runtime.interpreter as tflite

from utils import COCO_test_helper

# Bbox conofidence score threshold for TFLite model postprocessing
ABS_THRESH = 0.6

# Threshold for ONNX and RKNN models
OBJ_THRESH = 0.25
NMS_THRESH = 0.45

INPUT_SIZE = 640
IMG_SIZE = (INPUT_SIZE, INPUT_SIZE) 

CLASSES = ("person", "bicycle", "car","motorbike ","aeroplane ","bus ","train","truck ","boat","traffic light",
           "fire hydrant","stop sign ","parking meter","bench","bird","cat","dog ","horse ","sheep","cow","elephant",
           "bear","zebra ","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
           "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife ",
           "spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza ","donut","cake","chair","sofa",
           "pottedplant","bed","diningtable","toilet ","tvmonitor","laptop	","mouse","remote ","keyboard ","cell phone","microwave ",
           "oven ","toaster","sink","refrigerator ","book","clock","vase","scissors ","teddy bear ","hair drier", "toothbrush ")

# Plot detections 
def annotate_image(image, boxes, classes, scores):
    h_, w_, _= image.shape
    if boxes is not None:
        for box, class_id, score in zip(boxes, classes, scores):
            x, y, x1, y1 = box.astype(np.int64)
            cv2.rectangle(image, (x, y), (x1, y1), (0, 255, 0), 2)
            text = f'{CLASSES[int(class_id)]}: {score:.2f}'
            cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return image

#========================================================================================
# [PRE|POST]PROCESSING TOOLS 
# from Rockchip's repo 'rknn_model_zoo:
# https://github.com/airockchip/rknn_model_zoo/blob/main/examples/yolov8/python/yolov8.py
#========================================================================================

def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with object threshold.
    """
    box_confidences = box_confidences.reshape(-1)
    candidate, class_num = box_class_probs.shape

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score* box_confidences >= OBJ_THRESH)
    scores = (class_max_score* box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]

    return boxes, classes, scores

def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.
    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep

def dfl(position):
    # Distribution Focal Loss (DFL)
    x = torch.tensor(position)
    n,c,h,w = x.shape
    p_num = 4
    mc = c//p_num
    y = x.reshape(n,p_num,mc,h,w)
    y = y.softmax(2)
    acc_metrix = torch.tensor(range(mc)).float().reshape(1,1,mc,1,1)
    y = (y*acc_metrix).sum(2)
    return y.numpy()


def box_process(position):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1]//grid_h, IMG_SIZE[0]//grid_w]).reshape(1,2,1,1)

    position = dfl(position)
    box_xy  = grid +0.5 -position[:,0:2,:,:]
    box_xy2 = grid +0.5 +position[:,2:4,:,:]
    xyxy = np.concatenate((box_xy*stride, box_xy2*stride), axis=1)

    return xyxy

def post_process(input_data):
    boxes, scores, classes_conf = [], [], []
    defualt_branch=3
    pair_per_branch = len(input_data)//defualt_branch
    for i in range(defualt_branch):
        boxes.append(box_process(input_data[pair_per_branch*i]))
        classes_conf.append(input_data[pair_per_branch*i+1])
        scores.append(np.ones_like(input_data[pair_per_branch*i+1][:,:1,:,:], dtype=np.float32))

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0,2,3,1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]

    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)

    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)

        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores


#========================================================================================
#
# DIFFERENT YOLO MODEL WRAPPERS - these I've developed myself
#
#========================================================================================

class TFLiteBackend(object):
    def __init__(self, path="../data/models/yolov5s-fp16.tflite"):
        self.interpreter = tflite.Interpreter(model_path=path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.coco_helper = COCO_test_helper(enable_letter_box=True)

        expected_input_shape = [1, 640, 640, 3]
        if not np.all(self.input_details[0]['shape'] == expected_input_shape):
            raise ValueError(f"Expected input shape {expected_input_shape}, but got {self.input_details[0]['shape']}")

        expected_output_shape = [1, 25200, 85]
        if not np.all(self.output_details[0]['shape'] == expected_output_shape):
            raise ValueError(f"Expected output shape {expected_output_shape}, but got {self.output_details[0]['shape']}")


    def preprocess(self, image):
        pad_color = (0,0,0)
        img = self.coco_helper.letter_box(im=image.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0,0,0))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.reshape(1, *img.shape).astype(np.float32)
        img = img / 255.0
        return img

    def postprocess(self, outputs):
        boxes, scores, classes = [], [], []
        for output in outputs:
            if output[4] >= ABS_THRESH:
                x_center, y_center, width, height = output[:4]  * INPUT_SIZE
                x_min = float((x_center - width / 2)) 
                y_min = float((y_center - height / 2)) 
                x_max = float(x_min + width) 
                y_max = float(y_min + height)
                boxes.append([x_min, y_min, x_max, y_max])
                scores.append(output[4])
                classes.append(np.argmax(output[5:]))
                
        if len(boxes) > 0:
            boxes = self.coco_helper.get_real_box(np.array(boxes))

        return boxes, np.array(classes), np.array(scores)

    def detect(self, input):
        if input.shape != (1, 640, 640, 3):
            raise ValueError(f"Expected input shape (1, 640, 640, 3), but got {input.shape}")
        
        input = input.astype(np.float32)

        self.interpreter.set_tensor(self.input_details[0]['index'], input)
        self.interpreter.invoke()

        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output[0]

class ONNXBackend(object):
    def __init__(self, path="../data/models/yolov8s.onnx"):
        self.session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.coco_helper = COCO_test_helper(enable_letter_box=True)

    def preprocess(self, image):
        pad_color = (0,0,0)
        img = self.coco_helper.letter_box(im=image.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0,0,0))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2,0,1))
        img = img.reshape(1, *img.shape).astype(np.float32)
        img = img / 255.0
        return img

    def postprocess(self, outputs):
        boxes, classes, scores = post_process(outputs)
        if boxes is not None:
            boxes = self.coco_helper.get_real_box(boxes)
        return boxes, classes, scores


    def detect(self, input):
        outputs = self.session.run(self.output_names, {self.input_name: input})
        return outputs

class RKNNBackend(object):
    def __init__(self, path="../data/models/yolov5s.rknn"):
        self.rknn = RKNNLite(verbose=False)
        self.coco_helper = COCO_test_helper(enable_letter_box=True)

        if self.rknn.load_rknn(path) < 0:
            print(f"ERROR: failed to load RKNN model: {path}")

        if self.rknn.init_runtime() < 0:
            print(f"ERROR: runtime initialization for RKNNLite failed.")


    def preprocess(self, image):
        pad_color = (0,0,0)
        img = self.coco_helper.letter_box(im=image.copy(), new_shape=(IMG_SIZE[1], IMG_SIZE[0]), pad_color=(0,0,0))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.reshape(1, *img.shape)
        return img
    
    def postprocess(self, outputs):
        boxes, classes, scores = post_process(outputs)
        if boxes is not None:
            boxes = self.coco_helper.get_real_box(boxes)
        return boxes, classes, scores

    def detect(self, input):
        outputs = self.rknn.inference(inputs=[input])
        return outputs


class Yolo(object):
    def __init__(self, path="../data/models/yolov5s.onnx", shape=(640, 640)):
        extension = os.path.splitext(path)[1]

        if extension == ".onnx":
            self.backend = ONNXBackend(path)
        elif extension == ".rknn":
            self.backend = RKNNBackend(path)
        elif extension == ".tflite":
            self.backend = TFLiteBackend(path)
        else:
            print(f"ERROR: [{extension}] model format is not supported!")
            exit(-1)
    
    def detect(self, input):
        return self.backend.detect(input)

    def preprocess(self, image):
        return self.backend.preprocess(image)

    def postprocess(self, outputs):
        return self.backend.postprocess(outputs)
