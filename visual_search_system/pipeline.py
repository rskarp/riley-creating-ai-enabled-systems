from enum import Enum
import os
from glob import glob

from src.extraction.model import Model
from src.extraction.preprocess import Preprocessing
from src.search.indexing import KDTree
from src.search.search import KDTreeSearch, Measure

from pathlib import Path
from PIL import Image
from typing import List
import numpy as np
import re


# Location of storage
GALLERY_STORAGE = "storage/gallery"
MULTI_IMAGE_GALLERY_STORAGE = "storage/multi_image_gallery"
EMBEDDINGS_STORAGE = "storage/embeddings"
ACCESS_LOGS_STORAGE = "storage/access_logs"

parent_folder = str(Path(__file__).parent) + '/'

ImageSize = Enum('ImageSize', ['064', '224'])
Architecture = Enum('Architecture', ['resnet_018', 'resnet_034'])


class Pipeline:

    def __init__(self, image_size: ImageSize, architecture: Architecture, measure: Measure, dimension: int = 256, gallery_folder: str = GALLERY_STORAGE):
        '''
        Initialize the Pipeline class with all components and precompute gallery embeddings.

        Parameters:
            image_size (64 or 224): The input image size.
            architecture (resnet_018 or resnet_034): The model architecture.
            measure (Measure method): The distance measure to use for nearest neighbor search.
            dimension (int, default=256): The dimension of the KD tree.
            gallery_folder (str, default="storage/gallery"): The folder containing the gallery images.

        Returns:
            None
        '''
        self.preprocessing = Preprocessing(image_size=int(image_size))
        self.model_name = f'model_size_{image_size:03}_{architecture}'
        self.model = Model(
            parent_folder + f"simclr_resources/{self.model_name}.pth")
        self.index = None
        self.search = None
        self.gallery_folder = gallery_folder
        self.dimension = dimension
        self.measure = measure
        self.__precompute()

    def __predict(self, probe: np.ndarray) -> np.ndarray:
        '''
        Extract the embedding vector output from a preprocessed image.

        Parameters:
            probe (ndarray): The image matrix.

        Returns:
            ndarray: The embedding vector.
        '''
        return self.model.extract(probe)

    def _process_image(self, filename):
        probe = Image.open(filename)
        processed = self.preprocessing.process(probe)
        embedding = self.__predict(processed)
        self.__save_embeddings(filename, embedding)
        fullName = filename.split('/')[-2]
        metadata = {'filename': filename.removeprefix(parent_folder), 'firstName': fullName.split(
            '_')[0], 'lastName': '_'.join(fullName.split('_')[1:])}
        return embedding, metadata

    def __precompute(self):
        '''
        Precomputes the embeddings for all images in self.gallery_folder and construct a K-D Tree to organize the embedding.

        Parameters:
            None

        Returns:
            None
        '''
        pattern = parent_folder + self.gallery_folder + '/*/*.jpg'
        jpg_files = glob(pattern)
        points = np.zeros([len(jpg_files), self.dimension])
        metadata = []
        print('Calculating embeddings...')
        for i in range(len(jpg_files)):
            filename = jpg_files[i]
            embedding, meta = self._process_image(filename)
            points[i, :] = embedding.T
            metadata.append(meta)

        print('Indexing embeddings...')
        self.index = KDTree(k=self.dimension, points=points,
                            metadata_list=metadata)
        self.search = KDTreeSearch(self.index, self.measure)

    def __save_embeddings(self, filename: str, embedding: np.ndarray):
        '''
        Extract the embedding vector output from a preprocessed image.

        Parameters:
            filename (string): The name of the image file.
            embedding (ndarray): The embedding vector of the image.

        Returns:
            None
        '''
        outputFile = filename.replace(
            self.gallery_folder, f'{EMBEDDINGS_STORAGE}/{self.model_name}')[:-3] + 'npy'
        directory = re.search(
            f'({parent_folder}storage/embeddings/model_size_\d+_resnet_\d+/[^/]+)', outputFile).group(1)
        Path(directory).mkdir(parents=True, exist_ok=True)
        with open(outputFile, 'wb') as f:
            np.save(f, embedding)

    def search_gallery(self, probe: np.ndarray, k: int) -> List:
        '''
        Returns the nearest neighbors of a probe.

        Parameters:
            probe (ndarray): The probe image matrix.
            k (int): The number of nearest neighbors to return.

        Returns:
            List: The nearest neighbors of the probe including first name, last name, and image filename.
        '''
        processed = self.preprocessing.process(probe)
        embedding = self.__predict(processed)
        return self.search.find_nearest_neighbors(embedding, k)

    def set_search_measure(self, measure: Measure):
        '''
        Sets the search measure used in the KNN KD search to the given measure.

        Parameters:
            measure (function): The measurement function to calculate the similarity between 2 points.

        Returns:
            None
        '''
        self.measure = measure
        self.search = KDTreeSearch(self.index, self.measure)

    def get_search_measure(self):
        '''
        Gets the search measure that's currently being used in the pipeline's KNN KD search.

        Parameters:
            None.

        Returns:
            str: Name of the similarity measure function.
        '''
        return self.measure.__name__

    def add_identity(self, filename: str):
        embedding, metadata = self._process_image(filename)
        self.index.insert(embedding.T, metadata)
        self.search = KDTreeSearch(self.index, self.measure)

    def remove_identity(self, imageFilename: str, delete_file: bool = False):
        if os.path.isfile(imageFilename):
            fullName = imageFilename.split('/')[-2]
            baseName = os.path.basename(imageFilename)
            embeddingFilename = f'{parent_folder}{EMBEDDINGS_STORAGE}/{self.model_name}/{fullName}/{baseName[:-4]}.npy'
            with open(embeddingFilename, 'rb') as f:
                embedding = np.load(f).T
            self.index.remove(embedding.T)
            self.search = KDTreeSearch(self.index, self.measure)

            if delete_file:
                os.remove(imageFilename)
                os.remove(embeddingFilename)


if __name__ == "__main__":
    pipeline = Pipeline(image_size=64, architecture='resnet_018',
                        dimension=256, measure=Measure.euclidean, gallery_folder=GALLERY_STORAGE)
    probe = Image.open('storage/gallery/Jim_Edmonds/Jim_Edmonds_0001.jpg')
    neighbors = pipeline.search_gallery(probe, 2)
    for n in neighbors:
        print(n[1])
