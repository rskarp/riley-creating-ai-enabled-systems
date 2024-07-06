FROM python:3.11.7

RUN apt update
RUN apt install -y ffmpeg 

RUN pip install ffmpeg-python opencv-python-headless

COPY deployment_udp_source.py .

CMD ["pipeline.py"]
ENTRYPOINT [ "python" ]