import os
from glob import glob
import numpy as np
import cv2
import random
import pandas as pd


class HardNegativeMiner:
    """
    A class to mine hard negative examples for a given model.

    Attributes:
        model: The model used for prediction.
        nms: Non-maximum suppression object.
        measure: Measure to evaluate predictions.
        dataset_dir: Directory containing the dataset.
        table: DataFrame to store the results.
    """

    def __init__(self, model, nms, measure, dataset_dir):
        """
        Initialize the HardNegativeMiner with model, nms, measure, and dataset directory.

        Args:
            model: The model used for prediction.
            nms: Non-maximum suppression object.
            measure: Measure to evaluate predictions.
            dataset_dir: Directory containing the dataset.
        """
        self.model = model
        self.nms = nms
        self.measure = measure
        self.dataset_dir = dataset_dir
        self.table = pd.DataFrame(
            columns=['annotation_file', 'image_file'] + self.measure.columns)

    def __read_annotations(self, file_path):
        """
        Read annotations from a text file.

        Args:
            file_path (str): Path to the annotation file.

        Returns:
            list: List of annotations in the format (class_label, x_center, y_center, width, height).
        """
        annotations = []
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                class_label = int(parts[0])
                bbox = list(map(float, parts[1:]))
                annotations.append((class_label, *bbox))
        return annotations

    def __predict(self, image):
        """
        Make a prediction on the provided image using the model.

        Args:
            image (ndarray): The image to predict on.

        Returns:
            output: The model's prediction output.
        """
        output = self.model.predict(image)
        return output

    def __construct_table(self, sample_size=None):
        """
        Construct a table with image files, annotation files, and measures.

        This method reads images and annotations, makes predictions,
        computes measures, and appends the results to the table.

        Args:
            sample_size (int, optional): The number of files to randomly sample and use as the dataset to sample from.
        """
        allFiles = list(zip(
            sorted(glob(os.path.join(self.dataset_dir, "*.jpg"))),
            sorted(glob(os.path.join(self.dataset_dir, "*.txt")))))
        if sample_size != None:
            sampleSet = random.sample(
                allFiles, np.min([sample_size, len(allFiles)]))
        else:
            sampleSet = allFiles
        for image_file, annotation_file in sampleSet:

            image = cv2.imread(image_file)
            annotation = self.__read_annotations(annotation_file)
            prediction = self.__predict(image)

            measures = self.measure.compute(prediction, annotation)

            self.table = pd.concat([self.table,
                                    pd.DataFrame([{'annotation_file': annotation_file,
                                                   'image_file': image_file, **measures}])],
                                   ignore_index=True
                                   )

    def sample_hard_negatives(self, num_hard_negatives, criteria, sample_size=None):
        """
        Sample hard negative examples based on the specified criteria.

        Args:
            num_hard_negatives (int): The number of hard negatives to sample.
            criteria (str): The criteria to sort and sample the hard negatives.
            sample_size (int, optional): The number of files to randomly sample from.

        Returns:
            DataFrame: A DataFrame containing the sampled hard negative examples.
        """
        self.__construct_table(sample_size)
        self.table.sort_values(by=criteria, inplace=True,
                               ascending=False, ignore_index=True)
        return self.table.head(num_hard_negatives)
