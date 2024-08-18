import datetime
from flask import Flask, request, jsonify, send_file
from deployment import Deployment
from dateutil import parser

app = Flask(__name__)


@app.route('/')
def index():
    """
    Index route that returns a welcome message.

    Returns:
    JSON response with a welcome message.
    """
    return "Welcome to Riley's Retrieval Augmented Generation System!"


@app.route('/question', methods=['POST'])
def answer_question():
    """
    Route to trigger the question answering process for the given question.

    Request Body:
    JSON request with question string.

    Returns:
    JSON response with answer and a list of context documents used.
    """
    data = request.get_json()
    if not data or data.get("question", None) is None:
        return jsonify({"error": "No question provided"}), 400

    response = deployment.answer_question(data.get('question', ''))
    return jsonify({"response": response}), 200


@app.route('/documents', methods=['GET'])
def get_documents():
    """
    Route to get list of documents in the corpus.

    Returns:
    JSON response with a list of filenames.
    """
    files = deployment.get_documents()
    return jsonify({"files": files}), 200


@app.route('/document', methods=['PUT'])
def add_document():
    """
    Route to add document to the corpus.

    Request Body:
    Document to add.

    Returns:
    JSON response with the name given to the uploaded file.
    """
    file = request.files['document']
    document = file.stream.read()
    if not document:
        return jsonify({"error": "No document provided"}), 400

    description = deployment.add_document(file.filename, document)
    return jsonify({"filename": description}), 200


@app.route('/document', methods=['DELETE'])
def remove_document():
    """
    Route to remove document from the corpus.

    URL Parameters:
    filename (str): Name of the document to remove.

    Returns:
    JSON response with filename of the deleted file.
    """
    filename = request.args.get('filename', '')
    if filename == '':
        return jsonify({"error": "filename must be provided"}), 400

    description = deployment.remove_document(filename)
    return jsonify({"removed": description}), 200


@app.route('/logs', methods=['GET'])
def get_logs():
    """
    Route to get the quesiton log history of a specific time period.

    URL Parameters:
    start_time (str): Starting time of time range.
    end_time (str): Ending time of time range.

    Returns:
    JSON response with list of question logs.
    """
    start_time = request.args.get('start_time', '')
    end_time = request.args.get('end_time', '')

    if start_time == '':
        return jsonify({"error": "start_time must provided"}), 400
    if end_time == '':
        return jsonify({"error": "end_time must provided"}), 400

    start = parser.parse(start_time)
    end = parser.parse(end_time)

    if start > datetime.datetime.now() or end < start:
        return jsonify({"error": "start_time must be =< now and end_time must be > start_time"}), 400

    logs = deployment.get_logs(start, end)
    return jsonify({"logs": logs}), 200


@app.route('/document_files', methods=['POST'])
def get_document_files():
    """
    Route to get the document files associated with the given filenames.

    Request body:
    JSON object containing list of document filenames.

    Returns:
    zip file containing the files of the corpus documents.
    """
    data = request.get_json()
    if not data or data.get("filenames", None) is None:
        return jsonify({"error": "No filenames provided"}), 400
    stream = deployment.get_document_files(data.get("filenames", []))
    return send_file(
        stream,
        as_attachment=True,
        download_name='documents.zip'
    )


if __name__ == '__main__':
    deployment = Deployment()
    app.run(host='0.0.0.0', port=8000, debug=False)
