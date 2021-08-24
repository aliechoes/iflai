import numpy as np
import pandas as pd
import logging
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.feature_selection import  SelectFromModel 
from sklearn.feature_selection import mutual_info_classif
from scipy.cluster import hierarchy
from collections import defaultdict
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calc_vif(X):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)

class AutoFeatureSelection(BaseEstimator, TransformerMixin):
    """
    Auto feature selection transformer
    """
    def __init__(self, top_k = 20, correlation_threshold = 0.95, distance_threshold = 2., verbose =False):
        self.top_k = top_k 
        self.correlation_threshold = correlation_threshold
        self.distance_threshold = distance_threshold
        self.verbose = verbose
        self.selected_features = []
    
    def fit(self, X,y):
        X_ = X.copy()
        
        if self.verbose:
            print("Step 1: drop highly correlated features")
        cor_matrix = pd.DataFrame(X_).corr().abs()
        upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))
        to_keep = [column for column in upper_tri.columns if any(upper_tri[column] < self.correlation_threshold)]
        
        if self.verbose: 
            print(  "From", 
                    X.shape[1] ,
                    "initial features Selected (correlated):", 
                    len(to_keep))
        X_ = X_[:,to_keep]


        if self.verbose: 
            print("Step 2: wrapper methods")
        selector1 = mutual_info_classif(X_, y) 
        selector1 = selector1 > selector1.mean() + 1.*selector1.std()
        
        selector2 = SelectFromModel(    SVC(kernel="linear"),
                                        max_features = self.top_k)
        selector2 = selector2.fit(X_, y)
        if self.verbose:
            print('Calculating SVC')

        selector3 = SelectFromModel(    RandomForestClassifier(),  
                                        max_features = self.top_k)
        selector3 = selector3.fit(X_, y) 
        if self.verbose:
            print('Calculating random forest')
        
        selector4 = SelectFromModel(    LogisticRegression(penalty="l1", 
                                            solver="liblinear"),
                                        max_features = self.top_k)
        selector4 = selector4.fit(X_, y) 
        if self.verbose:
            print('Calculating l1 logistic regression')
        
        selector5 = SelectFromModel(    LogisticRegression(penalty="l2"),
                                        max_features = self.top_k)
        selector5 = selector5.fit(X_, y)
        if self.verbose:
            print('Calculating l2 logistic regression')
        
        selector6 = SelectFromModel(    XGBClassifier(n_jobs = -1, 
                                            eval_metric='logloss'),
                                        max_features = 20)
        selector6 = selector6.fit(X_, y)
        if self.verbose:
            print('Calculating xgb')


        importances = [ selector1, 
                        selector2.get_support(), 
                        selector3.get_support() ,  
                        selector4.get_support(),
                        selector5.get_support() ]

        for i, im in enumerate(importances):
            self.selected_features = self.selected_features + np.where(im)[0].tolist()
            self.selected_features = sorted(list(set(self.selected_features)))
        
        if self.verbose:
            print(  "From", 
                X.shape[1] ,
                "initial features Selected (multicolinear):", 
                len(self.selected_features))

        if self.verbose:
            print("Step 3: clustering over correlation of features")
        
        corr_spearman = pd.DataFrame(X[:,self.selected_features]).corr("spearman").to_numpy()
        corr_spearman = hierarchy.ward(corr_spearman)    

        cluster_ids = hierarchy.fcluster(corr_spearman,
                                        self.distance_threshold, 
                                        criterion='distance')

        cluster_id_to_feature_ids = defaultdict(list)
        for idx, cluster_id in enumerate(cluster_ids):
            cluster_id_to_feature_ids[cluster_id].append(idx)
        selected_features_spearman = [v[0] for v in cluster_id_to_feature_ids.values()]

        self.selected_features = np.array(self.selected_features)[selected_features_spearman]
        if self.verbose:
            print(  "From", 
                X.shape[1] ,
                "initial features Selected (uncorrelated):", 
                len(self.selected_features))
        return self
    
    def transform(self,X):
        X_ = X.copy()
        X_ = X_[:,self.selected_features]
        return X_