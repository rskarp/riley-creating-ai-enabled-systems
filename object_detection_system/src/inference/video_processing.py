import cv2

class VideoProcessing:

    def __init__(self, udp_url, skip_every_frame=30):
        self.stream_capture = cv2.VideoCapture(udp_url, cv2.CAP_FFMPEG)

    def resize_image(self, image):
        height, width, _ = image.shape
        if height != self.output_size and width != self.output_size:
            image = cv2.resize(image, (self.output_size, self.output_size))

    def scale_image(self):
        # TODO: implement this function to scale the image 
        # (i.e. pixel-wise normalize and standardize).
        pass

    def capture_udp_stream(self):
        """
        Captures video from a UDP stream and displays it in a window.

        Parameters:
        - udp_url: URL of the UDP stream.

        """
        # Open a connection to the UDP stream
        if not self.stream_capture.isOpened():
            print(f"Error: Unable to open UDP stream {udp_url}")
            return

        while True:
            # Read a frame from the UDP stream
            ret, frame = self.stream_capture.read()


            if not ret:
                print("Error: Unable to read frame from UDP stream")
                break

            yield frame

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
        self.stream_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    udp_url = 'udp://127.0.0.1:23000'  # Replace with your UDP stream URL
    stream = VideoProcessing(udp_url)
    for frame in stream.capture_udp_stream():
        cv2.imshow('UDP Stream', frame)