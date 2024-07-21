import cv2
import numpy as np
import os
import random


class VideoProcessing:
    """
    A class to process video streams from a UDP source.

    Attributes:
        stream_capture: The VideoCapture object for the UDP stream.
    """

    def __init__(self, udp_url, output_size=416, skip_every_frame=30):
        """
        Initialize the VideoProcessing with the UDP URL and frame skip interval.

        Args:
            udp_url (str): The URL of the UDP stream.
            skip_every_frame (int, optional): Number of frames to skip. Defaults to 30.
        """
        self.udp_url = udp_url
        self.stream_capture = cv2.VideoCapture(udp_url, cv2.CAP_FFMPEG)
        self.skip_every_frame = skip_every_frame
        self.output_size = output_size

    def resize_image(self, image):
        """
        Resize the input image to a specified size if it does not match the output size.

        Args:
            image (ndarray): The image to resize.

        Returns:
            ndarray: The resized image.
        """
        height, width, _ = image.shape
        if height != self.output_size and width != self.output_size:
            image = cv2.resize(image, (self.output_size, self.output_size))
        return image

    def scale_image(self, frame):
        """
        Scale the image by normalizing and standardizing pixel values.

        Args:
            image (ndarray): The image to scale.

        Returns:
            ndarray: The scaled image.
        """
        scaled = frame.copy()
        for i in range(3):
            channel = scaled[:, :, i]
            mean = np.mean(channel)
            std = np.std(channel)
            channel = (channel-mean)/std
            max = np.max(channel)
            min = np.min(channel)
            scaled[:, :, i] = (channel-min)/(max-min)
        return scaled

    def capture_udp_stream(self):
        """
        Capture video from a UDP stream and yield frames.

        Yields:
            ndarray: The next frame from the UDP stream.
        """
        if not self.stream_capture.isOpened():
            print(f"Error: Unable to open UDP stream {self.udp_url}")
            return

        frame_count = 0
        while True:
            ret, frame = self.stream_capture.read()

            if not ret:
                print("Error: Unable to read frame from UDP stream")
                break

            # Only process every nth frame
            if frame_count % self.skip_every_frame == 0:
                yield frame, frame_count

            # Increment frame counter
            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.stream_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    udp_url = 'udp://127.0.0.1:23000'  # Replace with your UDP stream URL
    stream = VideoProcessing(udp_url)
    for frame, _ in stream.capture_udp_stream():
        cv2.imshow('UDP Stream', frame)
