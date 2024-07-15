import json
import datetime
import threading
import cv2
from flask import Flask, request, jsonify, send_file
from deployment import Deployment

app = Flask(__name__)


@app.route('/')
def index():
    """
    Index route that returns a welcome message.

    Returns:
    JSON response with a welcome message.
    """
    return "Welcome to Riley's Object Detection System!"


@app.route('/detections_list', methods=['GET'])
def get_detections_list():
    """
    Route to get the list of detections identified by the system within the given frame range.

    URL Parameters:
    start_frame (int): Starting index of frame range.
    end_frame(int): Ending index of frame range.

    Returns:
    JSON response with a list of detections.
    """
    start_frame = int(request.args.get('start_frame', -1))
    end_frame = int(request.args.get('end_frame', -1))
    if start_frame < 0 or end_frame < start_frame:
        return jsonify({"error": "start_frame must be >= 0 and end_frame must be > start_frame"}), 400

    detections = deployment.get_detections_list(start_frame, end_frame)
    return jsonify({"detections": detections}), 200


@app.route('/hard_negatives', methods=['GET'])
def get_hard_negatives():
    """
    Route to get the list of top N hard negatives.

    URL Parameters:
    N (int): Number of top hard negatives to return.

    Returns:
    JSON response with a list of hard negatives base filenames.
    """
    N = int(request.args.get('N', -1))
    if N < 0:
        return jsonify({"error": "N must be greater than 0"}), 400

    hard_negatives = deployment.get_hard_negatives(N)
    return jsonify({"hard_negatives": hard_negatives}), 200


@app.route('/predictions', methods=['GET'])
def get_predictions():
    """
    Route to get the image files with detections and associated prediction files.

    URL Parameters:
    start_frame (int): Starting index of frame range.
    end_frame(int): Ending index of frame range.

    Returns:
    JSON response with a list of detections.
    """
    start_frame = int(request.args.get('start_frame', -1))
    end_frame = int(request.args.get('end_frame', -1))
    if start_frame < 0 or end_frame < start_frame:
        return jsonify({"error": "start_frame must be > 0 and end_frame must be > start_frame"}), 400
    stream = deployment.get_predictions(start_frame, end_frame)
    return send_file(
        stream,
        as_attachment=True,
        download_name='predictions.zip'
    )


if __name__ == '__main__':
    deployment = Deployment()
    # Run video capture and Flask API in different threads
    t0 = threading.Thread(target=deployment.initialize_inference_service)
    t1 = threading.Thread(target=lambda: app.run(
        host='0.0.0.0', port=8000, debug=False))
    # Stard the threads
    t0.start()
    t1.start()
    # Stop both threads
    if cv2.waitKey(1) & 0xFF == ord('q'):
        t0.join()
        t1.join()
