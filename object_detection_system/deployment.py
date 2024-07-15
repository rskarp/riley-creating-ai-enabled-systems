from src.inference.object_detection import YOLOObjectDetector
from src.inference.non_maximal_suppression import NMS
from src.inference.video_processing import VideoProcessing
from src.rectification.hard_negative_mining import HardNegativeMiner
from src.metrics import Loss, Metrics
from pipeline import InferenceService
import os
from glob import glob
from io import BytesIO
from zipfile import ZipFile


class Deployment:
    """
    Deployment class handles the deployment logic for ML models,
    including rectification and inference pipelines.
    """

    def __init__(self):
        """
        Initialize the Deployment with necessary components.
        """
        self.dataset_dir = 'storages/prediction'

        cfg_path = "yolo_resources/yolov4-tiny-logistics_size_416_1.cfg"
        weights_path = "yolo_resources/models/yolov4-tiny-logistics_size_416_1.weights"
        names_path = "yolo_resources/logistics.names"
        self.detector = YOLOObjectDetector(cfg_path, weights_path, names_path)

        score_threshold = .5
        iou_threshold = .4
        self.nms = NMS(score_threshold, iou_threshold)

        video_source = 'udp://127.0.0.1:23000'
        self.skip_every_frame = 30
        self.stream = VideoProcessing(
            video_source, skip_every_frame=self.skip_every_frame)

        self.init_hard_negative_miner()
        print('Pipeline initilaized')

    def initialize_inference_service(self):
        inference = InferenceService(self.stream, self.detector, self.nms)
        for detections in inference.detect():
            print(detections)

    def init_hard_negative_miner(self):
        loss_parameters = {'num_classes': 20,
                           'lambda_coord': 5., 'lambda_noobj': 1.}
        self.miner = HardNegativeMiner(
            model=self.detector, nms=self.nms, measure=Loss(**loss_parameters), dataset_dir=self.dataset_dir)

    def get_detections_list(self, start_frame: int, end_frame: int):
        detections = []
        filenames = [
            f'{self.dataset_dir}/frame_{i}.txt' for i in range(start_frame, end_frame+1) if i % self.skip_every_frame == 0]

        for fname in filenames:
            if os.path.isfile(fname):
                with open(fname, 'r') as f:
                    detections.extend([line.strip() for line in f.readlines()])

        return detections

    def get_hard_negatives(self, N: int):
        hard_negatives = self.miner.sample_hard_negatives(
            N, criteria='total_loss')
        baseNames = [name[len(self.dataset_dir)+1:-4]
                     for name in hard_negatives['annotation_file']]
        return baseNames

    def get_predictions(self, start_frame: int, end_frame: int):
        baseNames = [
            f'{self.dataset_dir}/frame_{i}' for i in range(start_frame, end_frame+1) if i % self.skip_every_frame == 0]

        stream = BytesIO()
        with ZipFile(stream, 'w') as zf:
            for name in baseNames:
                imgPath = name+'.jpg'
                annotationPath = name+'.txt'
                if os.path.isfile(imgPath):
                    zf.write(imgPath, os.path.basename(imgPath))
                if os.path.isfile(annotationPath):
                    zf.write(annotationPath, os.path.basename(annotationPath))
        stream.seek(0)
        return stream
