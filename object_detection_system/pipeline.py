

import cv2
import os
import numpy as np

from src.inference import object_detection
from src.inference import non_maximal_suppression
from src.inference import video_processing
from src.rectification import hard_negative_mining
import src.metrics as metrics


class InferenceService:
    def __init__(self, stream: video_processing.VideoProcessing, detector: object_detection.YOLOObjectDetector, nms: non_maximal_suppression.NMS):
        self.stream = stream
        self.detector = detector
        self.nms = nms

    def _save(self, frame, filename, detections):
        path = 'storages/prediction'
        file = f'{path}/{filename}'
        # Create folder if it doesn't exist
        os.makedirs(path, exist_ok=True)
        # Save image
        cv2.imwrite(f'{file}.jpg', frame)
        # Save objects
        objects = []
        # Format detections into YOLO format
        frameHeight, frameWidth = frame.shape[:2]
        for class_id, confidence, box in zip(*detections):
            x, y, w, h = box
            x_center = x+w/2
            y_center = y+h/2
            objects.append(
                f'{class_id} {x_center/frameWidth} {y_center/frameHeight} {w/frameWidth} {h/frameHeight}')
        with open(f'{file}.txt', "w") as objs:
            objs.write('\n'.join(objects))

    def detect(self):
        ctr = 0
        for frame in self.stream.capture_udp_stream():
            resized = self.stream.resize_image(frame)
            output = self.detector.predict(resized)
            detections = self.detector.process_output(output)
            filteredDetections = nms.filter(detections)
            self._save(frame, f'frame_{ctr}', filteredDetections)
            ctr += 1
            yield filteredDetections


if __name__ == "__main__":
    cfg_path = "yolo_resources/yolov4-tiny-logistics_size_416_1.cfg"
    weights_path = "yolo_resources/models/yolov4-tiny-logistics_size_416_1.weights"
    names_path = "yolo_resources/logistics.names"

    score_threshold = .5
    iou_threshold = .4

    detector = object_detection.YOLOObjectDetector(
        cfg_path, weights_path, names_path)

    nms = non_maximal_suppression.NMS(score_threshold, iou_threshold)

    # video_source = 'udp://127.0.0.1:23000'
    # stream = video_processing.VideoProcessing(video_source)
    # pipeline = InferenceService(stream, detector, nms)

    # for detections in pipeline.detect():
    #     print(detections)

    dataset_dir = 'storages/training'
    num_hard_negatives = 5
    loss_parameters = {'num_classes': 20,
                       'lambda_coord': 5., 'lambda_noobj': 1.}
    miner = hard_negative_mining.HardNegativeMiner(
        model=detector, nms=nms, measure=metrics.Loss(**loss_parameters), dataset_dir=dataset_dir)
    print(miner.sample_hard_negatives(num_hard_negatives, criteria='total_loss'))
