from typing import List
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
        cfg_path = "yolo_resources/yolov4-tiny-logistics_size_416_2.cfg"
        weights_path = "yolo_resources/models/yolov4-tiny-logistics_size_416_2.weights"
        names_path = "yolo_resources/logistics.names"
        self.detector = YOLOObjectDetector(
            cfg_path, weights_path, names_path)

        score_threshold = 0.3
        iou_threshold = 0.3
        self.nms = NMS(score_threshold, iou_threshold)

        video_source = 'udp://127.0.0.1:23000'
        self.skip_every_frame = 2
        frame_size = 624
        self.stream = VideoProcessing(
            video_source, output_size=frame_size, skip_every_frame=self.skip_every_frame)

        self.init_hard_negative_miner()
        print('Pipeline initilaized')

    def initialize_inference_service(self):
        '''
        Initialize running the inference service so that objects in any streamed video will be detected.
        '''
        inference = InferenceService(self.stream, self.detector, self.nms)
        for detections in inference.detect():
            print(detections)

    def init_hard_negative_miner(self):
        '''
        Initialize the hard negative mining component.
        '''
        loss_parameters = {'num_classes': 20,
                           'lambda_coord': 5., 'lambda_noobj': 1.}
        self.miner = HardNegativeMiner(
            model=self.detector, nms=self.nms, measure=Loss(**loss_parameters), dataset_dir=self.dataset_dir)

    def get_detections_list(self, start_frame: int, end_frame: int) -> List:
        '''
        Get the list of detections identified by the system within the given frame range.

        Args:
            start_frame (int): Starting index of frame range.
            end_frame (int): Ending index of frame range.

        Returns:
            List: The list of detections.
        '''
        detections = []
        filenames = [
            f'{self.dataset_dir}/frame_{i}.txt' for i in range(start_frame, end_frame+1) if i % self.skip_every_frame == 0]

        for fname in filenames:
            if os.path.isfile(fname):
                with open(fname, 'r') as f:
                    detections.extend([line.strip() for line in f.readlines()])

        return detections

    def get_hard_negatives(self, N: int, sample_size: int = 1000) -> List:
        """
        Get the list of top N hard negatives.

        Args:
            N (int): Number of top hard negatives to return.
            sample_size (int, optional. Default=1000): The number of files to randomly sample from.

        Returns:
            List: The list of hard negatives base filenames.
        """
        hard_negatives = self.miner.sample_hard_negatives(
            N, criteria='total_loss', sample_size=sample_size)
        baseNames = [name[len(self.dataset_dir)+1:-4]
                     for name in hard_negatives['annotation_file']]
        return baseNames

    def get_predictions(self, start_frame: int, end_frame: int):
        """
        Get the image files with detections and associated prediction files.

        Args:
            start_frame (int): Starting index of frame range.
            end_frame(int): Ending index of frame range.

        Returns:
            stream: Stream of a zip file containing the detection images and annotations.
        """
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
