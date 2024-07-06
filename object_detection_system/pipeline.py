

import cv2

from src.inference import object_detection
from src.inference import non_maximal_suppression
from src.inference import video_processing


class InferenceService:
    def __init__(self, stream, detector, nms):
        self.stream = stream
        self.detector = detector
        self.nms = nms

    def detect(self):
        # TODO: Implement this method for your Module 5 Assignment

if __name__ == "__main__":
    cfg_path = "yolo_resources/yolov4-tiny-logistics_size_416_1.cfg"
    weights_path = "yolo_resources/models/yolov4-tiny-logistics_size_416_1.weights"
    names_path = "yolo_resources/logistics.names"
    detector = object_detection.YOLOObjectDetector(cfg_path, weights_path, names_path)

    score_threshold = .5
    iou_threshold = .4
    nms = non_maximal_suppression.NMS(score_threshold,iou_threshold)

    video_source = 'udp://127.0.0.1:23000'
    stream = video_processing.VideoProcessing(video_source)

    pipeline = InferenceService(stream, detector, nms)    

    for detections in pipeline.detect():
        print(detections)