from glob import glob
from pipeline import MULTI_IMAGE_GALLERY_STORAGE, Pipeline
from src.search.search import Measure
from src.metrics import RankingMetrics
import os
from datetime import datetime
import json
from PIL import Image
from io import BytesIO
from zipfile import ZipFile
from dateutil import parser


class Deployment:
    """
    Deployment class handles the deployment logic for the system components,
    including extraction and search services.
    """

    def __init__(self):
        """
        Initialize the Deployment with necessary components.
        """
        self.pipeline = Pipeline(image_size=224, architecture='resnet_018',
                                 dimension=256, measure=Measure.euclidean, gallery_folder=MULTI_IMAGE_GALLERY_STORAGE)
        self.k = 3
        self.metrics = RankingMetrics(k=self.k)
        self.log_date_format = "%Y%m%d_%H%M%S"

    def _save_access_log(self, image, predictions):
        timestamp = datetime.now()
        filename = f'{timestamp.strftime(self.log_date_format)}'
        path = 'storage/logs'
        imageFile = f'{path}/{filename}.jpg'
        # Create folder if it doesn't exist
        os.makedirs(path, exist_ok=True)
        # Save image file
        Image.fromarray(image).save(imageFile)
        # Save log information in json file
        log_entry = {
            'image_file': imageFile,
            'timestamp': timestamp,
            'predictions': predictions
        }
        with open(f'{path}/{filename}.json', "w") as log:
            json.dump(log_entry, log, indent=4)

    def authenticate(self, image):
        """
        Get predicted identities for the given probe image.

        Parameters:
            image (ndarray): Image of the person.

        Returns:
            List: list of predicted identities.
        """
        neighbors = self.pipeline.search_gallery(image, self.k)
        self._save_access_log(image, neighbors)
        return neighbors

    def get_identity(self, fullName):
        """
        Add identity to the gallery.

        Parameters:
            full_name (str): Name of the identity.

        Returns:
            List: description.
        """
        cleanName = fullName.strip().replace(" ", "_")
        folder = MULTI_IMAGE_GALLERY_STORAGE + '/' + cleanName
        pattern = f'{folder}/*.jpg'
        identity_files = glob(pattern)
        return identity_files

    def add_identity(self, fullName, image):
        """
        Add identity to the gallery.

        Parameters:
            full_name (str): Name of the identity.
            image (ndarray): Image of the person.

        Returns:
            Dict: description.
        """
        cleanName = fullName.strip().replace(" ", "_")
        folder = MULTI_IMAGE_GALLERY_STORAGE + '/' + cleanName
        # Create folder if it doesn't exist
        os.makedirs(folder, exist_ok=True)
        # Determine new filename
        pattern = f'{folder}/*.jpg'
        identity_files = glob(pattern)
        filename = f'{folder}/{cleanName}_{len(identity_files)+1:04}.jpg'
        # Save image file
        Image.fromarray(image).save(filename)
        # Add new image to KD Tree
        self.pipeline.add_identity(filename)
        return filename

    def remove_identity(self, imageFilename):
        """
        Remove identity from the gallery.

        Parameters:
            full_name (str): Name of the identity to remove.

        Returns:
            Dict: description.
        """
        # Remvoe from KD tree and gallery
        self.pipeline.remove_identity(imageFilename, True)

    def get_access_logs(self, start_time, end_time):
        """
        Get the access log history of a specific time period.

        Parameters:
            start_time (str): Starting time of time range.
            end_time (str): Ending time of timerange.

        Returns:
            List: access logs.
        """
        logs = []
        start = parser.parse(start_time)
        end = parser.parse(end_time)
        pattern = 'storage/logs/*.json'
        logFiles = glob(pattern)
        for f in logFiles:
            if start <= datetime.strptime(f[:-5], self.log_date_format) <= end:
                with open(f, 'r') as file:
                    data = json.load(file)
                logs.append(data)
        return logs

    def get_image_files(self, files):
        """
        Get the image files and the names of the given identities.

        Parameters:
            filenames (List): list of image filenames.

        Returns:
            stream: Stream of a zip file containing the images of the given identities.
        """
        stream = BytesIO()
        with ZipFile(stream, 'w') as zf:
            for imgPath in files:
                if os.path.isfile(imgPath):
                    zf.write(imgPath, os.path.basename(imgPath))
        stream.seek(0)
        return stream
