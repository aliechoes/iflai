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
    def __init__(self, top_k = 20, multicolinearity_threshold = 10.):
        self.top_k = top_k 
        self.multicolinearity_threshold = multicolinearity_threshold
        self.selected_features = []
    
    def fit(self, X,y):

        selector1 = mutual_info_classif(X, y) 
        selector1 = selector1 > selector1.mean() + 1.*selector1.std()
        print('Calculating mutual info')

        selector2 = SelectFromModel(    SVC(kernel="linear"),
                                        max_features = self.top_k)
        selector2 = selector2.fit(X, y)
        print('Calculating SVC')

        selector3 = SelectFromModel(    RandomForestClassifier(),  
                                        max_features = self.top_k)
        selector3 = selector3.fit(X, y) 
        print('Calculating random forest')
        
        selector4 = SelectFromModel(    LogisticRegression(penalty="l1", 
                                            solver="liblinear"),
                                        max_features = self.top_k)
        selector4 = selector4.fit(X, y) 
        print('Calculating l1 logistic regression')
        
        selector5 = SelectFromModel(    LogisticRegression(penalty="l2"),
                                        max_features = self.top_k)
        selector5 = selector5.fit(X, y)
        print('Calculating l2 logistic')
        
        selector6 = SelectFromModel(    XGBClassifier(n_jobs = -1, 
                                            eval_metric='logloss'),
                                        max_features = 20)
        selector6 = selector6.fit(X, y)
        print('Calculating xgb')


        importances = [ selector1, 
                        selector2.get_support(), 
                        selector3.get_support() ,  
                        selector4.get_support(),
                        selector5.get_support() ]

        for i, im in enumerate(importances):
            self.selected_features = self.selected_features + np.where(im)[0].tolist()
            self.selected_features = sorted(list(set(self.selected_features)))
        
        print(  "From", 
                X.shape[1] ,
                "initial features Selected (correlated):", 
                len(self.selected_features))

        while True:
            indx = calc_vif(pd.DataFrame(X).iloc[:,self.selected_features])["VIF"]
            if len(self.selected_features) <= self.top_k :
                break
            elif (indx > self.multicolinearity_threshold).sum() > 0.:
                self.selected_features.pop(indx.argmax())
            else:
                break

        print(  "From", 
                X.shape[1] ,
                "initial features Selected (uncorrelated):", 
                len(self.selected_features))        
        return self
    
    def transform(self,X):
        X_ = X.copy()
        X_ = X_[:,self.selected_features]
        return X_