from typing import List
import numpy as np
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import itertools
import cv2


class Loss:
    """
    Modified YOLO Loss for Hard Negative Mining.

    Attributes:
        num_classes (int): Number of classes.
        iou_threshold (float): Intersection over Union (IoU) threshold.
        lambda_coord (float): Weighting factor for localization loss.
        lambda_noobj (float): Weighting factor for no object confidence loss.
    """

    def __init__(self, num_classes=20, iou_threshold=0.5, lambda_coord=0.5, lambda_noobj=0.5):
        """
        Initialize the Loss object with the given parameters.

        Args:
            num_classes (int): Number of classes.
            lambda_coord (float): Weighting factor for localization loss.
            lambda_noobj (float): Weighting factor for no object confidence loss.
        """
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.columns = [
            'total_loss',
            f'loc_loss (lambda={self.lambda_coord})',
            'conf_loss_obj',
            f'conf_loss_noobj (lambda={self.lambda_noobj})',
            'class_loss'
        ]
        self.iou_threshold = iou_threshold

    def compute(self, predictions, annotations):
        """
        Compute the YOLO loss components.

        Args:
            predictions (list): List of predictions.
            annotations (list): List of ground truth annotations.

        Returns:
            dict: Dictionary containing the computed loss components.
        """
        loc_loss = 0
        class_loss = 0
        conf_loss_obj = 0
        conf_loss_noobj = 0

        pred_box = np.array([preds[:4]
                            for preds in itertools.chain(*predictions)])
        pred_score = np.array([preds[5:]
                              for preds in itertools.chain(*predictions)])
        objectness_score = np.array(
            [preds[4] for preds in itertools.chain(*predictions)])

        gt_box = np.array([anns[1:] for anns in annotations])
        gt_class_id = np.array([anns[0] for anns in annotations])

        boxes = np.vstack([gt_box, pred_box]) if len(gt_box) > 0 else pred_box
        ious = Metrics.calculate_ious(boxes)[
            :len(annotations), len(annotations):]

        obj_masks = ious > self.iou_threshold
        noobj_masks = (ious > self.iou_threshold) & (objectness_score != 0.0)

        for idx, (obj_mask, noobj_mask) in enumerate(zip(obj_masks, noobj_masks)):
            loc_loss += self.lambda_coord * \
                np.sum((pred_box[obj_mask] - gt_box[idx][0:4]) ** 2)
            conf_loss_obj += np.sum((objectness_score[obj_mask] - 1) ** 2)
            conf_loss_noobj += self.lambda_noobj * \
                np.sum((objectness_score[noobj_mask] - 0) ** 2)
            class_loss += self.__cross_entropy_loss(
                (np.arange(self.num_classes) == gt_class_id[idx]).astype(
                    float), pred_score[obj_mask]
            )

        total_loss = loc_loss + conf_loss_obj + conf_loss_noobj + class_loss

        return {
            "total_loss": total_loss,
            "loc_loss": loc_loss,
            "conf_loss_obj": conf_loss_obj,
            "conf_loss_noobj": conf_loss_noobj,
            "class_loss": class_loss
        }

    def __cross_entropy_loss(self, y_true, y_pred, epsilon=1e-12):
        """
        Compute the cross entropy loss between true labels and predicted probabilities.

        Args:
            y_true (numpy array): True labels, one-hot encoded or binary labels.
            y_pred (numpy array): Predicted probabilities, same shape as y_true.
            epsilon (float): Small value to avoid log(0). Default is 1e-12.

        Returns:
            float: Cross entropy loss.
        """
        # Clip y_pred to avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)

        # Binary classification case
        if y_true.ndim == 1 or y_true.shape[1] == 1:
            loss = -np.mean(y_true * np.log(y_pred) +
                            (1 - y_true) * np.log(1 - y_pred))
        # Multi-class classification case
        else:
            loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        return loss

    def get_annotations(self, file_path):
        """
        Read annotations from a text file.

        Args:
            file_path (str): Path to the annotation file.

        Returns:
            list: List of annotations in the format (class_label, x_center, y_center, width, height).
        """
        annotations = []
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                class_label = int(parts[0])
                bbox = list(map(float, parts[1:]))
                annotations.append((class_label, *bbox))
        return annotations


class Metrics:
    """
    A class to compute various metrics for object detection.
    """

    @staticmethod
    def calculate_ious(boxes, coco_format=False):
        """
        Compute the Intersection over Union (IoU) of each pair of bounding boxes using vectorized operations.

        Args:
            boxes (numpy array): NumPy array of bounding boxes with shape (N, 4),
                                 where N is the number of boxes. Each box is represented by
                                 a list of coordinates [x1, y1, x2, y2].

        Returns:
            numpy array: A NumPy matrix where the element at [i][j] is the IoU between boxes[i] and boxes[j].
        """

        # This is here because during my experimentation while working on desing_considerations.ipynb,
        #   I plotted the predicted and annotated bounding boxes on the images and discovered that they
        #   are actually provided in this format.
        if coco_format:
            x1 = boxes[:, 0]
            x2 = boxes[:, 0] + boxes[:, 2]
            y1 = boxes[:, 1]
            y2 = boxes[:, 1] + boxes[:, 3]
        else:
            x1 = boxes[:, 0]
            y1 = boxes[:, 1]
            x2 = boxes[:, 2]
            y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)

        inter_x1 = np.maximum(x1[:, None], x1)
        inter_y1 = np.maximum(y1[:, None], y1)
        inter_x2 = np.minimum(x2[:, None], x2)
        inter_y2 = np.minimum(y2[:, None], y2)

        inter_w = np.maximum(0, inter_x2 - inter_x1 + 1)
        inter_h = np.maximum(0, inter_y2 - inter_y1 + 1)
        inter_area = inter_w * inter_h

        union_area = area[:, None] + area - inter_area

        iou_matrix = inter_area / union_area

        return iou_matrix

    def get_annotations(self, file_path):
        """
        Read annotations from a text file.

        Args:
            file_path (str): Path to the annotation file.

        Returns:
            list: List of annotations in the format (class_label, x_center, y_center, width, height).
        """
        annotations = []
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                class_label = int(parts[0])
                bbox = list(map(float, parts[1:]))
                annotations.append((class_label, *bbox))
        return annotations

    @staticmethod
    def get_matches(detections, annotations, iou_threshold, height, width):
        """
        Compute the precision, recall, thresholds, and mean avergae precision (mAP).

        Args:
            detections (list): List of detected objects.
            annotations (list): List of ground truth annotations.
            iou_threshold (float): The threshold to use when calculationg IOUs.
            height (int): Image frame height.
            width (int): Image frame width.

        Returns:
            tuple (list,list): List containing the true class ids and prediction scores for detections that match ground truth objects.
        """
        # Get ground truth bounding boxes
        gt_box = np.array([anns[1:] for anns in annotations])
        # Convert bounding boxes into same format as detections
        gt_box = np.vstack([width*(gt_box[:, 0] - gt_box[:, 2]/2), height*(gt_box[:, 1] - gt_box[:, 2]/2),
                            width*gt_box[:, 2],  height*gt_box[:, 2]]).astype(int).T
        gt_class_id = np.array([anns[0] for anns in annotations])
        # Get prediction scores and boxes
        pred_box = np.array(detections[2])
        pred_score = np.array(detections[3])
        # Combine ground truth and predicted boxes into one array
        boxes = np.vstack([gt_box, pred_box]) if len(
            pred_box) > 0 else gt_box
        # Get IOU between predicted and true objects
        ious = Metrics.calculate_ious(boxes, True)
        ious = ious[:len(annotations), len(annotations):]
        # Apply IOU threshold
        obj_masks = ious > iou_threshold
        rowMask = np.any(obj_masks, axis=0)
        # Determine matched objects based on IOU threshold
        match_scores = pred_score[rowMask]
        match_ids = gt_class_id
        return match_ids, match_scores

    @staticmethod
    def calculate_metrics(gt_class_ids, pred_scores, num_classes):
        """
        Compute the precision, recall, thresholds, and mean avergae precision (mAP).

        Args:
            gt_class_ids (list): List of ground truth class ids.
            pred_scores (list): List of prediction scores for each detection.
            num_classes (int): Number of object classes in the data.

        Returns:
            dict: Dictionary containing the computed precision, recall, thresholds, and mAP.
        """
        precision, recall, thresholds = Metrics.calculate_precision_recall_curve(
            gt_class_ids, pred_scores, num_classes)

        precision_recall_points = {
            i: list(zip(recall[i], precision[i])) for i in range(num_classes)} if len(recall) > 0 else {}

        map_value = Metrics.calculate_map_11_point_interpolated(
            precision_recall_points, num_classes)

        return {
            'precision': precision,
            'recall': recall,
            'thresholds': thresholds,
            'map': map_value
        }

    @staticmethod
    def calculate_precision_recall_curve(y_true, scores, num_classes):
        """
        Calculate the precision-recall curve for multi-class tasks.

        Args:
            y_true (array-like): True labels.
            scores (array-like): Target scores, can either be probability estimates of each class,
                                 confidence values, or non-thresholded measure of decisions.
            num_classes (int): Number of classes.

        Returns:
            tuple: Precision values for each class, recall values for each class, and
                   decreasing thresholds on the decision function used to compute precision and recall for each class.
        """

        precision = {}
        recall = {}
        thresholds = {}

        if len(scores) == 0 or scores.shape[0] == 0:
            return precision, recall, thresholds

        y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))

        for i in range(num_classes):
            class_scores = scores[:, i] if len(
                scores.shape) > 1 and scores.shape[1] > 1 else scores
            class_y_true_bin = y_true_bin[:,
                                          i] if len(y_true_bin.shape) > 1 and y_true_bin.shape[1] > 1 else y_true_bin
            sorted_indices = np.argsort(scores[:, i])[
                ::-1] if len(scores.shape) > 1 and scores.shape[1] > 1 else np.argsort(class_scores)[::-1]
            sorted_scores = class_scores[sorted_indices]
            sorted_true = [class_y_true_bin[idx] if len(
                class_y_true_bin) > idx else 0 for idx in sorted_indices]

            # Append an extra threshold to cover all cases
            thresholds[i] = np.append(sorted_scores, [0], axis=0)
            precision[i] = []
            recall[i] = []
            TP = 0
            FP = 0
            # Initial false negatives are all positives
            FN = np.sum(y_true_bin[:, i])

            for j in range(len(sorted_scores)):
                if sorted_true[j] == 1:
                    TP += 1
                    FN -= 1
                else:
                    FP += 1

                prec = TP / (TP + FP) if TP + FP > 0 else 0
                rec = TP / (TP + FN) if TP + FN > 0 else 0

                precision[i].append(prec)
                recall[i].append(rec)

            # Ensure last point is at recall zero
            precision[i].append(1.0)
            recall[i].append(0.0)

        return precision, recall, thresholds

    @staticmethod
    def calculate_map_11_point_interpolated(precision_recall_points, num_classes):
        """
        Calculate the mean average precision (mAP) using 11-point interpolation for multi-class tasks.

        Args:
            precision_recall_points (dict): A dictionary where keys are class indices and values are lists of
                                            tuples representing (recall, precision) points.
            num_classes (int): Number of classes.

        Returns:
            float: The mAP value.
        """
        mean_average_precisions = []

        if len(precision_recall_points) == 0:
            return 0.

        for i in range(num_classes):
            points = precision_recall_points[i]
            # Ensure the list is sorted by confidence in descending order
            points = sorted(points, key=lambda x: x[0], reverse=True)

            interpolated_precisions = []
            for recall_threshold in [j * 0.1 for j in range(11)]:
                # Find all precisions with recall greater than or equal to the threshold
                possible_precisions = [
                    p for r, p in points if r >= recall_threshold]

                # Interpolate precision: take the maximum precision to the right of the current recall level
                if possible_precisions:
                    interpolated_precisions.append(max(possible_precisions))
                else:
                    interpolated_precisions.append(0)

            # Calculate the mean of the interpolated precisions
            mean_average_precision = sum(
                interpolated_precisions) / len(interpolated_precisions)
            mean_average_precisions.append(mean_average_precision)

        # Calculate the overall mean average precision
        overall_map = sum(mean_average_precisions) / num_classes

        return overall_map


if __name__ == "__main__":

    metrics = Metrics()
    # Example usage of calculate_map_11_point_interpolated():
    # Actual labels for 3000 samples with 3 classes (0, 1, 2)
    y_true = np.random.randint(0, 3, 3000)
    # Random predicted scores/probabilities for each class
    scores = np.random.rand(3000, 3)
    num_classes = 3

    precision, recall, thresholds = metrics.calculate_precision_recall_curve(
        y_true, scores, num_classes)
    print("Precision:", precision)
    print("Recall:", recall)
    print("Thresholds:", thresholds)

    precision_recall_points = {
        i: list(zip(recall[i], precision[i])) for i in range(num_classes)}

    map_value = metrics.calculate_map_11_point_interpolated(
        precision_recall_points, num_classes)
    print(f"Mean Average Precision: {map_value:.4f}")
