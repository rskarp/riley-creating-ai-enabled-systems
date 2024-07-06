# Object Detection System

```sh
ffmpeg -re -i ./yolo_resources/test_videos/worker-zone-detection.mp4 -r 30 -vcodec mpeg4 -f mpegts udp://127.0.0.1:23000
```

```sh
docker build -t object_detection_system:latest .
```

```sh
docker run -it --rm -p 23000:23000/udp object_detection_system
```
