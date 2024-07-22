import os
from glob import glob

from src.extraction.model import Model
from src.extraction.preprocess import Preprocessing
from src.search.indexing import KDTree
from src.search.search import KDTreeSearch, Measure

from PIL import Image
import numpy as np


# Location of storage
GALLERY_STORAGE = "storage/gallery"
EMBEDDINGS_STORAGE = "storage/embeddings"
ACCESS_LOGS_STORAGE = "storage/access_logs"


class Pipeline:

    def __init__(self, preprocessing, model, index, search):
        self.preprocessing = preprocessing
        self.model = model
        self.index = index
        self.search = search

    def __predict(self, probe):
        # TODO: IMPLEMENT THIS...
        pass
        
    def __precompute(self):
        # TODO: IMPLEMENT THIS...
        pass

    def __save_embeddings(self, filename, embedding):
        # TODO: IMPLEMENT THIS...
        pass

    def search_gallery(self, probe):
        # TODO: IMPLEMENT THIS...
        pass
    

if __name__ == "__main__":
    image_size = 224
    architecture = 'resnet_034'

    preprocessing = Preprocessing(image_size=image_size)
    model = Model(f"simclr_resources/model_size_{image_size:03}_{architecture}.pth")
    index = KDTree(k=256)
    search_euclidean = KDTreeSearch(index, Measure.euclidean)
    pipeline = Pipeline(preprocessing=preprocessing,
                        model=model,
                        index=index,
                        search = search_euclidean
                        )

    pipeline.precompute()

