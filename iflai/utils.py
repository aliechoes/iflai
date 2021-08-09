import os
import numpy as np
import pandas as pd
import glob
import logging
from sklearn.model_selection import train_test_split


def list_of_dict_to_dict(list_of_dicts):
    new_dict = dict()
    for one_dict in list_of_dicts:
        new_dict.update(one_dict)
    return new_dict

def finding_classes(data_dir):
    """
    this function finds the folders in the root path and considers them
    as classes
    """
    classes = sorted(os.listdir(data_dir))
    logging.info("Classes: %s \n" % classes)
    return classes
    
def data_frame_creator(data_dir, classes, validation_split):
    """
    datafame including every file, which later will be used for reading the images.
    the structure is like this:
    file	        label	class	    prediction  set
    file1_Chx.ext	0	    class_0 	0	        test
    file2_Chx.ext	1	    class_1 	0	        train
    file3_Chx.ext	0	    class_0 	0	        validation
    .               .       .           .           .
    .               .       .           .           .
    .               .       .           .           .
    """
    df = input_dataframe_generator( data_dir, classes)
    df = train_validation_test_split(  df, validation_split)

    number_of_files_per_class(  self.df )
        
def number_of_files_per_class(df ):
    """
    this function finds the number of files in each folder. It is important to
    consider that we consider all the channels togethr as on single image
    output: dictionary with keys as classes and values as number of separate images
    """

    logging.info("detected independent images per classes") 
    logging.info(df.groupby(["class", "set"])["class"].agg("count")) 
    
    return None
    
    
    
def input_dataframe_generator(data_dir, classes):
    """
    This functions gets the dictionary with the classes and number of files 
    per class and gives back a dataframe with these columns
    ["file"  ,"label", "class", "prediction",  
            "class0_probability" ... "classN_probability"]
    """
    
    df = pd.DataFrame(columns= ["file"  ,
                                "label", 
                                "class", 
                                "set",
                                "uncertainty" ,
                                "prediction"] )
    
    data_directory = {"train" : data_dir}
    
    for dd in data_directory:
        train_data_path = data_directory[dd]
        for tdp in train_data_path:
            label = 0
            for cl in classes:
                df_dummy = pd.DataFrame(columns= ["file" ,"label", "class", "set","prediction"]  )
                df_dummy["file"] = glob.glob(os.path.join(tdp , cl,"*") ) 
                df_dummy["label"] = label
                df_dummy["class"] = cl
                df_dummy["uncertainty"] = -1.
                df_dummy["prediction"] = -1
                df_dummy["set"] = dd
                df = df.append(df_dummy, ignore_index=True)
                label = label + 1
    for cl in classes:
            df[cl+"_probability"] = -1.
    df_dummy["prediction"] = df_dummy["prediction"].astype(int)
    return df

    
def train_validation_test_split(df, validation_size= 0.2 , test_size= 0.2 ):
    """
    This functions gets the dataframe and creates train, validation and test 
    split. it adds a new column: "set"
    """
    assert validation_size <= 1.
    assert validation_size >= 0.
    assert test_size <= 1.
    assert test_size >= 0.
    
    _, X_test = train_test_split( df.loc[df["set"]=="train" , "set"], 
                                                stratify = df.loc[df["set"]=="train" , "set"], 
                                                test_size=test_size, 
                                                random_state=314)
    df.loc[X_test.index,"set"] = "test"
    
    _, X_validation = train_test_split( df.loc[df["set"]=="train" , "set"], 
                                                stratify = df.loc[df["set"]=="train" , "set"], 
                                                test_size=validation_size, 
                                                random_state=314)
    df.loc[X_validation.index,"set"] = "validation"

    return df