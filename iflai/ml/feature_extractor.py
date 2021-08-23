import os
import glob
import numpy as np
import pandas as pd
import h5py
import random
from joblib import Parallel, delayed
from tqdm import tqdm
from iflai.utils import list_of_dict_to_dict


def get_label(h5_file_path):
    h5_file = h5py.File(h5_file_path, "r")  
    ## label
    results = dict()
    try:
        results["label"] = h5_file.get("label")[()]
        results["set"] = "labeled"
    except TypeError:
        results["label"] = "-1"
        results["set"] = "unlabeled"
    try:
        results["object_number"] = os.path.split(h5_file_path)[-1].replace(".h5","").split("_")[-1]
    except TypeError:
        results["object_number"] = None
    h5_file.close()
    return results


def metadata_generator_(data_dir,sample_size=None ,n_jobs=-1):
    
    metadata_columns = ["file",
                        "experiment",
                        "donor", 
                        "condition",
                        "object_number",
                        "set",
                        "label"]
    metadata = pd.DataFrame(columns=metadata_columns)
    
    experiments_list = sorted(os.listdir(data_dir))
    print("Metadata prepration starts...")
    for exp in experiments_list:
        experiments_path = os.path.join(data_dir, exp)
        donors_list = sorted(os.listdir(experiments_path))
        for donor in donors_list:
            donors_path = os.path.join(data_dir, exp, donor)
            conditions_list = sorted(os.listdir(donors_path))
            for cond in conditions_list:
                print(exp, donor, cond)
                conditions_path = os.path.join(data_dir, exp, donor, cond + "/*.h5" )
                files = glob.glob(conditions_path)
                
                if sample_size:
                    files = random.choices(files, k=min(sample_size, len(files)) )
                
                metadata_temp = pd.DataFrame(columns=metadata_columns)
                metadata_temp["file"] = files
                metadata_temp["experiment"] = exp
                metadata_temp["donor"] = donor
                metadata_temp["condition"] = cond

                index_list = metadata_temp.file.tolist()
                results = Parallel(n_jobs=n_jobs)(delayed(get_label)(f) for f in tqdm(index_list, position=0, leave=True) )
                results = pd.DataFrame(results)
                
                if results.shape[0] > 0:
                    metadata_temp["label"] = results["label"]
                    metadata_temp["set"] = results["set"]
                    metadata_temp["object_number"] = results["object_number"]

                    metadata = metadata.append(metadata_temp, ignore_index = True)
                results = None
                metadata_temp = None
    print("...metadata prepration ended.")
    return metadata


class AmnisData(object): 
    def __init__(self,data_dir, feature_extractor, sample_size = None,n_jobs=-1):
        self.metadata = metadata_generator_(data_dir,sample_size,n_jobs)
        self.sample_size = sample_size
        self.feature_extractor = feature_extractor
        self.n_jobs = n_jobs
    
    def metadata_generator(self,data_dir):
        return metadata_generator_(data_dir)

    def extract(self,f):
        print(f)
        h5_file = h5py.File(f, "r")      
        image = h5_file.get("image")[()]*1.
        mask  = h5_file.get("mask")[()]
        h5_file.close()
        features = self.feature_extractor.transform([image,mask]).copy()
        features = list_of_dict_to_dict(features)
        return features
    
    def extract_features_for_all_samples(self):
        file_list = self.metadata["file"].tolist()
        results = Parallel(n_jobs=self.n_jobs)(delayed(self.extract)(f) for f in tqdm(file_list, position=0, leave=True) )
        self.df_features = pd.DataFrame(results)

    def get_image_mask(self, i = 0):
        h5_file = h5py.File(self.metadata.loc[i,"file"], "r")      
        image = h5_file.get("image")[()]*1.
        mask  = h5_file.get("mask")[()]
        h5_file.close()
        return image, mask