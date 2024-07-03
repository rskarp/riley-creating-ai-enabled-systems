import json
import datetime
from flask import Flask, request, jsonify
from deployment import DeploymentPipeline, Endpoint

app = Flask(__name__)


@app.route('/')
def index():
    """
    Index route that returns a welcome message.

    Returns:
    JSON response with a welcome message.
    """
    return "Welcome to Riley's Fraud Detection System!"


@app.route('/generate_new_dataset', methods=['PUT'])
def generate_new_dataset():
    """
    Route to generate a new dataset.

    URL Params:
    version : str
        The version name to save the data to.
    type : str (Optional. Default 'train')
        The type of datset to create: train or test.
    n_samples : int (Optional. Default 1000)
        The number of samples (if random) or samples per class (if stratified) to be sampled.
    sampling_type : str (Optional. Default 'stratified')
        The sampling method to use: random or stratified.
    random_state : int (Optional. Default 1)
        The random_state value to use for reproducibility.
    generate_features : boolean (Optional. Default True)
        Boolean indicating whether to extract features from the sampled dataset that is generated.

    Returns:
    JSON response with a success message and description.
    """
    pipeline.track_endpoint_usage(Endpoint.generate_new_dataset)
    dataset_version = request.args.get('version')
    dataset_type = request.args.get('type', 'train').lower()
    n_samples = int(request.args.get('n_samples', 1000))
    sampling_type = request.args.get('sampling_type', 'stratified')
    random_state = int(request.args.get('random_state', 1))
    generate_features = False if request.args.get(
        'generate_features', '').lower() == 'false' else True

    if not dataset_version:
        return jsonify({"error": "No version specified"}), 400
    if dataset_type not in ['train', 'test']:
        return jsonify({"error": "'type' parameter must have value 'train' or 'test'."}), 400
    description = pipeline.generate_new_dataset(
        dataset_version, dataset_type, n_samples, sampling_type, random_state)
    pipeline.log(
        'datasets',
        log_entry={
            "version": dataset_version,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": description
        },
        log_file=f'{dataset_version}.json'
    )

    # Generate features file from newly created dataset
    if generate_features == True:
        feature_description = pipeline.generate_new_features(
            dataset_version, random_state=random_state)
        pipeline.log(
            'features',
            log_entry={
                "version": dataset_version,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "description": feature_description
            },
            log_file=f'{dataset_version}.json'
        )

    return jsonify({"message": f"Generated new dataset {dataset_version}", "description": description}), 200


@app.route('/generate_new_features', methods=['PUT'])
def generate_new_features():
    """
    Route to generate new features from a dataset.

    URL Params:
    version : str
        The version name to use for the data source.
    run_smote : bool (Optional. Default False)
        Boolean flag indicating whether or not to run SMOTE if is_fraud classes are imbalanced.
    random_state : int (Optional. Default 1)
        The random_state value to use for reproducibility.

    Returns:
    JSON response with a success message and description.
    """
    pipeline.track_endpoint_usage(Endpoint.generate_new_features)
    dataset_version = request.args.get('version')
    run_smote = request.args.get('run_smote', '').lower() == 'true'
    random_state = int(request.args.get('random_state', 1))

    if not dataset_version:
        return jsonify({"error": "No version specified"}), 400
    description = pipeline.generate_new_features(
        dataset_version, run_smote, random_state)
    pipeline.log(
        'features',
        log_entry={
            "version": dataset_version,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": description
        },
        log_file=f'{dataset_version}.json'
    )
    return jsonify({"message": f"Generated new features for dataset {dataset_version}", "description": description}), 200


@app.route('/dataset_description', methods=['GET'])
def get_dataset_description():
    """
    Route to get the description of a dataset version.

    URL Params:
    version (str): The version of the dataset to describe.

    Returns:
    JSON response with the dataset description.
    """
    pipeline.track_endpoint_usage(Endpoint.dataset_description)
    dataset_version = request.args.get('version')
    if not dataset_version:
        return jsonify({"error": "No version specified"}), 400
    try:
        description = pipeline.get_log('datasets', dataset_version)
    except FileNotFoundError:
        return jsonify({"error": "Description not found"}), 404

    return jsonify({"description": description}), 200


@app.route('/dataset_features_description', methods=['GET'])
def get_dataset_features_description():
    """
    Route to get the description of features for a dataset version.

    URL Params:
    version (str): The version of the dataset features to describe.

    Returns:
    JSON response with the dataset features description.
    """
    pipeline.track_endpoint_usage(Endpoint.dataset_features_description)
    dataset_version = request.args.get('version')
    if not dataset_version:
        return jsonify({"error": "No version specified"}), 400
    try:
        description = pipeline.get_log('features', dataset_version)
    except FileNotFoundError:
        return jsonify({"error": "Description not found"}), 404

    return jsonify({"description": description}), 200


@app.route('/model_description', methods=['GET'])
def get_model_description():
    """
    Route to get the description of a model version.

    URL Params:
    version (str): The version of the model to describe.

    Returns:
    JSON response with the model description.
    """
    pipeline.track_endpoint_usage(Endpoint.model_description)
    model_version = request.args.get('version')
    if not model_version:
        return jsonify({"error": "No version specified"}), 400
    try:
        description = pipeline.get_log('models', model_version)
    except FileNotFoundError:
        return jsonify({"error": "Description not found"}), 404

    return jsonify({"description": description}), 200


@app.route('/dataset_list', methods=['GET'])
def get_dataset_list():
    """
    Route to get the list of available dataset versions.

    Returns:
    JSON response with the datasets list.
    """
    pipeline.track_endpoint_usage(Endpoint.dataset_list)
    datasets = pipeline.get_resource_list('datasets')

    return jsonify({"datasets": datasets}), 200


@app.route('/feature_list', methods=['GET'])
def get_feature_list():
    """
    Route to get the list of available feature versions.

    Returns:
    JSON response with the features list.
    """
    pipeline.track_endpoint_usage(Endpoint.feature_list)
    features = pipeline.get_resource_list('features')

    return jsonify({"features": features}), 200


@app.route('/model_list', methods=['GET'])
def get_model_list():
    """
    Route to get the list of available trained model versions.

    Returns:
    JSON response with the models list.
    """
    pipeline.track_endpoint_usage(Endpoint.model_list)
    datasets = pipeline.get_resource_list('models')

    return jsonify({"models": datasets}), 200


@app.route('/model_metrics', methods=['GET'])
def get_model_metrics():
    """
    Route to get the model metrics for a given model and dataset version.

    URL Params:
    dataset_version (str): The version of the dataset to use for inference.
    model_version (str): The version of the model to use for predictions.

    Returns:
    JSON response with the model metrics.
    """
    pipeline.track_endpoint_usage(Endpoint.model_metrics)
    timestamp = datetime.datetime.now()
    dataset_version = request.args.get('dataset_version')
    model_version = request.args.get('model_version')
    random_state = int(request.args.get('random_state', 1))
    if not dataset_version:
        return jsonify({"error": "No dataset_version specified"}), 400
    if not model_version:
        return jsonify({"error": "No model_version specified"}), 400

    metrics = pipeline.get_model_metrics(
        model_version, dataset_version, random_state)
    pipeline.log(
        'metrics',
        log_entry={
            'time': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            'model_version': model_version,
            'dataset_version': dataset_version,
            'metrics': metrics
        },
        log_file=f'{timestamp.strftime("%Y%m%d_%H%M%S")}.json'
    )

    return jsonify({"metrics": metrics}), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Route to make predictions based on input data.

    URL Params:
    version (str): model version name to use for inference.
    random_state (int, optional): random-state value for reproducibility.

    Request Body:
    JSON object containing input data for prediction.

    Returns:
    JSON response with the prediction result.
    """
    pipeline.track_endpoint_usage(Endpoint.predict)
    timestamp = datetime.datetime.now()
    model_version = request.args.get('version')
    random_state = int(request.args.get('random_state', 1))
    if not model_version:
        return jsonify({"error": "No version specified"}), 400
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    model_prediction = pipeline.predict(
        model_version, data, random_state).tolist()
    pipeline.log(
        'predictions',
        log_entry={
            'input_data': data,
            'time': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            'model_version': model_version,
            'prediction': model_prediction
        },
        log_file=f'{timestamp.strftime("%Y%m%d_%H%M%S")}.json'
    )

    return jsonify({"prediction": model_prediction}), 200


@app.route('/train', methods=['PUT'])
def train():
    """
    Route to train a model on given dataset.

    URL Parameters:
    model_version : str
        The name of the model version for this model.
    model_type : str
        The type of model to train: random_forest, stocahstic_gradient_descent, or logistic_regression.
    dataset_version : str
        The dataset to use for training data.
    random_state : int (Optoinal. Default 1)
        The random_state value to use for reproducibility.

    Request Body:
    JSON object containing the hyperparameters used for tuning. The value of each key in the object should be a list, even if it contains one value.

    Returns:
    JSON response with the model description.
    """
    pipeline.track_endpoint_usage(Endpoint.train)
    timestamp = datetime.datetime.now()
    dataset_version = request.args.get('dataset_version')
    model_version = request.args.get('model_version')
    model_type = request.args.get('model_type')
    random_state = int(request.args.get('random_state', 1))
    hyperparameters = request.get_json()

    if not dataset_version:
        return jsonify({"error": "No dataset_version provided"}), 400
    if not model_type:
        return jsonify({"error": "No model_type provided"}), 400
    if not model_version:
        return jsonify({"error": "No model_version provided"}), 400
    if model_type.lower() not in ['random_forest', 'stochastic_gradient_descent', 'logistic_regression']:
        return jsonify({"error": f"model_type must be in: 'random_forest', 'stochastic_gradient_descent', 'logistic_regression'"}), 400
    if not pipeline.dataset_has_features(dataset_version):
        return jsonify({"error": f"No features found for dataset_version {dataset_version}. Call PUT /generate_new_features?version={dataset_version}"}), 400

    description = pipeline.train(
        model_version, model_type, dataset_version, hyperparameters, random_state)

    pipeline.log(
        'models',
        log_entry={
            'model_version': model_version,
            'time': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            'description': description
        },
        log_file=f'{model_version}.json'
    )

    return jsonify({"description": description}), 200


@app.route('/system_metrics', methods=['GET'])
def get_system_metrics():
    """
    Route to get the metrics of endpoint usage of the system.

    Returns:
    JSON response with the system endpoint usgae.
    """
    pipeline.track_endpoint_usage(Endpoint.system_metrics)
    system_metrics = pipeline.get_system_metrics()

    return jsonify({"metrics": system_metrics}), 200


if __name__ == '__main__':
    pipeline = DeploymentPipeline()
    app.run(host='0.0.0.0', port=8000, debug=True)
