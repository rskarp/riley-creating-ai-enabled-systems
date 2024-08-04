import datetime
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
    return "Welcome to Riley's Visual Search System!"


@app.route('/authenticate', methods=['POST'])
def authenticate():
    """
    Route to trigger the authentication process for the given probe image.

    Request Body:
    Probe image.

    Returns:
    JSON response with a list of predicted identities.
    """
    data = request.get_json()
    print(request.files)
    if not data:
        return jsonify({"error": "No data provided"}), 400

    predictions = deployment.authenticate(data)
    return jsonify({"predictions": predictions}), 200


@app.route('/identity', methods=['GET'])
def get_identity():
    """
    Route to get list of image files for an identity in the gallery.

    URL Parameters:
    full_name (str): Name of the identity.

    Returns:
    JSON response with a list of filenames.
    """
    full_name = request.args.get('full_name', '')
    if full_name == '':
        return jsonify({"error": "full_name must be provided"}), 400

    files = deployment.get_identity(full_name)
    return jsonify({"files": files}), 200


@app.route('/add_identity', methods=['PUT'])
def add_identity():
    """
    Route to add identity to the gallery.

    URL Parameters:
    full_name (str): Name of the identity.

    Request Body:
    Image of the person.

    Returns:
    JSON response with a success message and description.
    """
    full_name = request.args.get('full_name', '')
    if full_name == '':
        return jsonify({"error": "full_name must be provided"}), 400
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    description = deployment.add_identity(full_name, data)
    return jsonify({"description": description}), 200


@app.route('/remove_identity', methods=['PUT'])
def remove_identity():
    """
    Route to remove identity from the gallery.

    URL Parameters:
    full_name (str): Name of the identity to remove.

    Returns:
    JSON response with a success message and description.
    """
    full_name = request.args.get('full_name', '')
    if full_name == '':
        return jsonify({"error": "full_name must be provided"}), 400

    description = deployment.remove_identity(full_name)
    return jsonify({"description": description}), 200


@app.route('/access_logs', methods=['GET'])
def get_access_logs():
    """
    Route to get the access log history of a specific time period.

    URL Parameters:
    start_time (str): Starting time of time range.
    end_time (str): Ending time of timerange.

    Returns:
    JSON response with list of access logs.
    """
    start_time = int(request.args.get('start_time', ''))
    end_time = int(request.args.get('end_time', ''))

    if start_time == '':
        return jsonify({"error": "start_time must provided"}), 400
    if end_time == '':
        return jsonify({"error": "end_time must provided"}), 400
    if start_time > datetime.datetime.now() or end_time < start_time:
        return jsonify({"error": "start_time must be =< now and end_time must be > start_time"}), 400

    access_logs = deployment.get_access_logs(start_time, end_time)
    return jsonify({"access_logs": access_logs}), 200


@app.route('/images', methods=['POST'])
def get_image_files():
    """
    Route to get the image files and the names of the predicted identities.

    Request body:
    JSON object containing list of image filenames.

    Returns:
    zip file containing the images of the predicted identities.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    stream = deployment.get_image_files(data)
    return send_file(
        stream,
        as_attachment=True,
        download_name='images.zip'
    )


if __name__ == '__main__':
    deployment = Deployment()
    app.run(host='0.0.0.0', port=8000, debug=False)
