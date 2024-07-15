import cv2
import numpy as np


class YOLOObjectDetector:
    """
    A class to perform object detection using YOLO (You Only Look Once).

    Attributes:
        cfg_path (str): Path to the YOLO configuration file.
        weights_path (str): Path to the YOLO weights file.
        names_path (str): Path to the file containing class names.
        frame_size (int): Size to which each frame is resized. Default is 416.
    """

    def __init__(self, cfg_path, weights_path, names_path, frame_size=416):
        """
        Initialize the YOLOObjectDetector with configuration, weights, and class names.

        Args:
            cfg_path (str): Path to the YOLO configuration file.
            weights_path (str): Path to the YOLO weights file.
            names_path (str): Path to the file containing class names.
            frame_size (int): Size to which each frame is resized. Default is 416.
        """
        self.cfg_path = cfg_path
        self.weights_path = weights_path
        self.names_path = names_path
        self.frame_size = frame_size

        # Load YOLO model
        self.net = cv2.dnn.readNetFromDarknet(self.cfg_path, self.weights_path)
        self.layer_names = self.net.getLayerNames()
        self.output_layer = [self.layer_names[i - 1]
                             for i in self.net.getUnconnectedOutLayers()]

        # Load class names
        with open(self.names_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

        print("Successfully loaded model...")

    def predict(self, frame):
        """
        Predict the objects in a given frame using the YOLO model.

        Args:
            frame (ndarray): The input image frame.

        Returns:
            list: The output from the YOLO model after forwarding the blob.
        """
        self.height, self.width = frame.shape[:2]

        frame = cv2.resize(frame, (416, 416))
        # Create a blob from the image
        blob = cv2.dnn.blobFromImage(frame, scalefactor=0.00392, size=(
            self.frame_size, self.frame_size), mean=(0, 0, 0), swapRB=True, crop=False)
        self.net.setInput(blob)

        outs = self.net.forward(self.output_layer)

        return outs

    def process_output(self, output):
        """
        Process the output from the YOLO model to extract class IDs, confidences, and bounding boxes.

        Args:
            output (list): The output from the YOLO model.

        Returns:
            tuple: Lists of class IDs, confidences, and bounding boxes.
        """
        class_ids = []
        confidences = []
        boxes = []

        for out in output:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.0:
                    center_x = int(detection[0] * self.width)
                    center_y = int(detection[1] * self.height)
                    w = int(detection[2] * self.width)
                    h = int(detection[3] * self.height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        return class_ids, confidences, boxes

    def draw_labels(self, frame, detections, show_image=True):
        """
        Draw bounding boxes and labels on the image frame.

        Args:
            frame (ndarray): The input image frame.
            detections (tuple): The detections including class IDs, confidences, and bounding boxes.
            show_image (boolean, default=True): Indicates whether to display the image. If false, return image. 
        """
        for class_id, confidence, box in zip(*detections):
            x, y, w, h = box
            label = f"{self.classes[class_id]}: {confidence:.2f}"
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display the frame
        if show_image:
            cv2.imshow("Frame", frame)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
        else:
            # Return frame with the labels
            return frame


if __name__ == "__main__":
    cfg_path = "yolo_resources/yolov4-tiny-logistics_size_416_1.cfg"
    weights_path = "yolo_resources/models/yolov4-tiny-logistics_size_416_1.weights"
    names_path = "yolo_resources/logistics.names"

    yolo_detector = YOLOObjectDetector(cfg_path, weights_path, names_path)
    frame = cv2.imread(
        "storages/training/znak0241-_jpg.rf.e42a126be0dcbafdc79bb9cfb5df5a1d.jpg")
    # frame = cv2.imread("yolo_resources/test_images/test_images.jpg")

    output = yolo_detector.predict(frame)
    output = yolo_detector.process_output(output)
    yolo_detector.draw_labels(frame, output)
