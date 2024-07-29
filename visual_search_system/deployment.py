from pipeline import MULTI_IMAGE_GALLERY_STORAGE, Pipeline
from src.search.search import Measure


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

    def authenticate(self):
        """
        Get predicted identities for the given probe image.

        Parameters:
            image (ndarray): Image of the person.

        Returns:
            List: list of predicted identities.
        """
        pass

    def add_identity(self):
        """
        Add identity to the gallery.

        Parameters:
            full_name (str): Name of the identity.
            image (ndarray): Image of the person.

        Returns:
            Dict: description.
        """
        pass

    def remove_identity(self):
        """
        Remove identity from the gallery.

        Parameters:
            full_name (str): Name of the identity to remove.

        Returns:
            Dict: description.
        """
        pass

    def get_access_logs(self):
        """
        Get the access log history of a specific time period.

        Parameters:
            start_time (str): Starting time of time range.
            end_time (str): Ending time of timerange.

        Returns:
            List: access logs.
        """
        pass

    def get_predictions(self):
        """
        Get the image files and the names of the predicted identities.

        Parameters:
            filenames (List): list of image filenames.

        Returns:
            stream: Stream of a zip file containing the images of the predicted identities.
        """
        pass
