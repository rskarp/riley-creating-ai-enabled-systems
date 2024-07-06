

import cv2
import os
import numpy as np

from src.inference import object_detection
from src.inference import non_maximal_suppression
from src.inference import video_processing


class InferenceService:
    def __init__(self, stream, detector, nms):
        self.stream = stream
        self.detector = detector
        self.nms = nms

    def _save(self, frame, filename, objects):
        path = 'storages/prediction'
        file = f'{path}/{filename}'
        # Create folder if it doesn't exist
        os.makedirs(path, exist_ok=True)
        # Save image
        cv2.imwrite(f'{file}.jpg', frame)
        # Save objects
        with open(f'{file}.txt', "w") as objs:
            objs.write('\n'.join(objects))

    def detect(self):
        # TODO: Implement this method for your Module 5 Assignment
        pass


if __name__ == "__main__":
    cfg_path = "yolo_resources/yolov4-tiny-logistics_size_416_1.cfg"
    weights_path = "yolo_resources/models/yolov4-tiny-logistics_size_416_1.weights"
    names_path = "yolo_resources/logistics.names"
    detector = object_detection.YOLOObjectDetector(
        cfg_path, weights_path, names_path)

    score_threshold = .5
    iou_threshold = .4
    nms = non_maximal_suppression.NMS(score_threshold, iou_threshold)

    video_source = 'udp://127.0.0.1:23000'
    stream = video_processing.VideoProcessing(video_source)

    pipeline = InferenceService(stream, detector, nms)

    for detections in pipeline.detect():
        print(detections)
