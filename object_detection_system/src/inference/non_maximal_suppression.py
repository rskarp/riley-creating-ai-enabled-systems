import numpy as np
import cv2


class NMS:
    """
    A class to perform Non-Maximal Suppression (NMS) on bounding boxes.

    Attributes:
        score_threshold (float): The threshold for confidence scores.
        nms_iou_threshold (float): The threshold for Intersection over Union (IoU) for NMS.
    """

    def __init__(self, score_threshold, nms_iou_threshold):
        """
        Initialize the NMS object with the given thresholds.

        Args:
            score_threshold (float): The threshold for confidence scores.
            nms_iou_threshold (float): The threshold for Intersection over Union (IoU) for NMS.
        """
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold

    def filter(self, outputs, for_evaluation=False):
        """
        Perform Non-Maximal Suppression (NMS) on bounding boxes, with class IDs.

        Args:
            outputs (tuple): A tuple containing class IDs, scores, and bounding boxes. If for_evaluation
                             is True, also includes class scores.
            for_evaluation (bool, optional): If True, includes class scores in the output. Defaults to False.

        Returns:
            tuple: The filtered list of bounding boxes, confidence scores, and class IDs after applying NMS.
                   If for_evaluation is True, also includes class scores.
        """
        if for_evaluation:
            class_ids, scores, bboxes, class_scores = outputs
        else:
            class_ids, scores, bboxes = outputs
            class_scores = []

        indices = cv2.dnn.NMSBoxes(
            bboxes, scores, self.score_threshold, self.nms_iou_threshold)
        class_ids = [class_ids[i] for i in indices]
        bboxes = [bboxes[i] for i in indices]
        scores = [scores[i] for i in indices]

        if for_evaluation:
            class_scores = [class_scores[i] for i in indices]
            return class_ids, scores, bboxes, class_scores
        else:
            return class_ids, scores, bboxes


if __name__ == "__main__":
    from object_detection import YOLOObjectDetector

    # Model arguments
    cfg_path = "yolo_resources/yolov4-tiny-logistics_size_416_1.cfg"
    weights_path = "yolo_resources/models/yolov4-tiny-logistics_size_416_1.weights"
    names_path = "yolo_resources/logistics.names"
    score_threshold = 0.5

    yolo_detector = YOLOObjectDetector(cfg_path, weights_path, names_path)

    iou_threshold = 0.4

    nms = NMS(score_threshold, iou_threshold)
    frame = cv2.imread("yolo_resources/test_images/test_images.jpg")

    output = yolo_detector.predict(frame)
    output = yolo_detector.process_output(output)
    yolo_detector.draw_labels(frame, output)

    output = nms.filter(output)

    yolo_detector.draw_labels(frame, output)
