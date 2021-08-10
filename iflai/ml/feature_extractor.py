import os
import glob
import numpy as np
import pandas as pd
import h5py
from joblib import Parallel, delayed
from iflai.utils import list_of_dict_to_dict

def metadata_generator_(data_dir):
    
    metadata_columns = ["file",
                        "experiment",
                        "donor", 
                        "condition",
                        "object_number",
                        "set",
                        "label"]
    metadata = pd.DataFrame(columns=metadata_columns)
    
    experiments_list = sorted(os.listdir(data_dir))
    for exp in experiments_list:
        experiments_path = os.path.join(data_dir, exp)
        donors_list = sorted(os.listdir(experiments_path))
        for donor in donors_list:
            donors_path = os.path.join(data_dir, exp, donor)
            conditions_list = sorted(os.listdir(donors_path))
            for cond in conditions_list:
                conditions_path = os.path.join(data_dir, exp, donor, cond + "/*.h5" )
                files = glob.glob(conditions_path)
                
                metadata_temp = pd.DataFrame(columns=metadata_columns)
                metadata_temp["file"] = files
                metadata_temp["experiment"] = exp
                metadata_temp["donor"] = donor
                metadata_temp["condition"] = cond
                metadata_temp["set"] = "unlabeled"
                metadata_temp["label"] = "-1"

                for i in range(metadata_temp.shape[0]):
                    h5_file = h5py.File(metadata_temp.loc[i,"file"], "r")  
                    ## label
                    try:
                       metadata_temp.loc[i,"label"] = h5_file.get("label")[()]
                       metadata_temp.loc[i,"set"] = "labeled"
                    except TypeError:
                        pass
                    ## object number
                    try:
                       metadata_temp.loc[i,"object_number"] = h5_file.get("object_number")[()]
                    except TypeError:
                        pass
                metadata = metadata.append(metadata_temp, ignore_index = True)
    return metadata


class AmnisData(object): 
    def __init__(self,data_dir, feature_extractor):
        self.metadata = metadata_generator_(data_dir)
        self.feature_extractor = feature_extractor
    
    def metadata_generator(self,data_dir):
        return metadata_generator_(data_dir)

    def extract(self,i):
        h5_file = h5py.File(self.metadata.loc[i,"file"], "r")      
        image = h5_file.get("image")[()]*1.
        mask  = h5_file.get("mask")[()]
        h5_file.close()
        features = self.feature_extractor.transform([image,mask]).copy()
        features = list_of_dict_to_dict(features)
        return features
    
    def extract_features_for_all_samples(self, n_jobs=1):
        results = Parallel(n_jobs=n_jobs)(delayed(self.extract)(i) for i in self.metadata.index.tolist() )
        self.df_features = pd.DataFrame(results)

    def get_image_mask(self, i = 0):
        h5_file = h5py.File(self.metadata.loc[i,"file"], "r")      
        image = h5_file.get("image")[()]*1.
        mask  = h5_file.get("mask")[()]
        h5_file.close()
        return image, mask