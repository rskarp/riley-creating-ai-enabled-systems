import cv2
import numpy as np
import os
import random


class VideoProcessing:

    def __init__(self, udp_url, output_size=416, skip_every_frame=30):
        self.udp_url = udp_url
        self.stream_capture = cv2.VideoCapture(udp_url, cv2.CAP_FFMPEG)
        self.skip_every_frame = skip_every_frame
        self.output_size = output_size
        # self.scale_stats = self.get_scale_stats()
        # print(self.scale_stats)

    def get_scale_stats(self):
        trainImgs = [
            f'logistics/{f}' for f in os.listdir('logistics') if f[-4:] == '.jpg']
        trainImgs = random.sample(trainImgs, 100)
        img = cv2.imread(trainImgs[0])
        h, w = img.shape[:2]
        rgbs = np.zeros((h*len(trainImgs), w, 3))
        for i in range(len(trainImgs)):
            if i % 100 == 0:
                print(f'{i}/{len(trainImgs)}')
            img = cv2.imread(trainImgs[i])
            rgbs[i*h:i*h+h, :, :] = img
        print('calculating stats')
        rgbs = np.reshape(rgbs, (h*w*len(trainImgs), 3))
        print(rgbs.shape)
        print(np.mean(rgbs, axis=0))
        print(np.std(rgbs, axis=0))
        stats = {
            'means': np.mean(rgbs, axis=0),
            'stds': np.std(rgbs, axis=0),
            'mins': np.min(rgbs, axis=0),
            'maxs': np.max(rgbs, axis=0)
        }
        print(stats)
        return stats

    def resize_image(self, image):
        height, width, _ = image.shape
        if height != self.output_size and width != self.output_size:
            image = cv2.resize(image, (self.output_size, self.output_size))
        return image

    def scale_image(self, frame):
        # TODO: implement this function to scale the image
        # (i.e. pixel-wise normalize and standardize per color channel).
        h, w = frame.shape[:2]
        # Calculate RGB stats

        scaled = frame.copy()
        # for i in range(3):
        #     channel = scaled[:, :, i]
        #     mean = np.mean(channel)
        #     std = np.std(channel)
        #     channel = (channel-mean)/std
        #     max = np.max(channel)
        #     min = np.min(channel)
        #     scaled[:, :, i] = (channel-min)/(max-min)

        rs = scaled.reshape(h*w, 3)

        # # Standardize = X-mean/std
        rs = (rs-self.scale_stats['means'])/self.scale_stats['stds']
        # Normalize = X-Xmin / Xmax-Xmin
        self.scale_stats['mins'] = np.min(rs, axis=0)
        self.scale_stats['maxs'] = np.max(rs, axis=0)
        rs = (rs-np.min(rs, axis=0)) / \
            (self.scale_stats['maxs']-self.scale_stats['mins'])
        print(
            f'standardized mean: {np.mean(scaled.reshape(h*w,3),axis=0)}, std: {np.std(scaled.reshape(h*w,3),axis=0)}')

        return rs.reshape(h, w, 3)
        # return scaled

    def capture_udp_stream(self):
        """
        Captures video from a UDP stream and displays it in a window.

        Parameters:
        - udp_url: URL of the UDP stream.

        """
        # Open a connection to the UDP stream
        if not self.stream_capture.isOpened():
            print(f"Error: Unable to open UDP stream {self.udp_url}")
            return

        frame_count = 0
        while True:
            # Read a frame from the UDP stream
            ret, frame = self.stream_capture.read()

            if not ret:
                print("Error: Unable to read frame from UDP stream")
                break

            # Only process every nth frame
            if frame_count % self.skip_every_frame == 0:
                yield frame

            # Increment frame counter
            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.stream_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    udp_url = 'udp://127.0.0.1:23000'  # Replace with your UDP stream URL
    stream = VideoProcessing(udp_url)
    for frame in stream.capture_udp_stream():
        print(f'original: {frame.shape}')
        r = stream.resize_image(frame)
        print(f'resize: {r.shape}')
        s = stream.scale_image(r)
        print(f'scale: {s.shape}')
        cv2.imshow('UDP Stream', s)
