import pandas as pd
import h5py
from joblib import Parallel, delayed
from tqdm import tqdm
from iflai.utils import list_of_dict_to_dict

def get_image_mask(f):
    """
    f: full file path to the .h5 file
    """
    h5_file = h5py.File(f, "r")      
    image = h5_file.get("image")[()]*1.
    mask  = h5_file.get("mask")[()]
    h5_file.close()
    return image, mask


class FeatureExtractor(object):
    """
    Feature extractor based on metadata dataframe.
    it expects to get the metadata dataframe including file paths. 
    """
    def __init__(self, feature_unions):
        """
        feature_unions is a series of sklearn transforms, using 
        sklearn.pipeline.FeatureUnion
        """
        self.feature_unions = feature_unions

    def extract_(self, f):
        """
        loads the images and masks based on the metadata["file"]
        and pass the feature_unions on the data
        """
        try:
            image, mask = get_image_mask(f)
            features = self.feature_unions.transform([image,mask]).copy()
            features = list_of_dict_to_dict(features)
        except:
            print("corrupted file", f)
            features = []
        return features

    def extract_features(self, metadata, n_jobs = -1):
        """
        parallelization of feature extraction. It uses tqdm for progress
        """
        file_list = metadata["file"].tolist()
        results = Parallel(n_jobs=n_jobs)(delayed(self.extract_)(f) \
            for f in tqdm(file_list, position=0, leave=True) )
        return results