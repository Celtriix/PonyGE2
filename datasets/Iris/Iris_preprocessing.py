# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 14:20:24 2022

@author: Dominik
"""

import numpy as np
import pandas as pd
import sklearn as skl

def main():
    df = pd.read_csv("Iris.csv", header = None)
    df.columns = ["x0", "x1", "x2", "x3", "y"]
    df["y_Cat"] = pd.Categorical(df["y"]).codes
    df_out = df.iloc[:,[0,1,2,3,-1]]
    #df_out.to_csv("Iris_clean.csv", header = True, index = False)
    split_dataset(df_out)
    
def split_dataset(pd_data):
    pd_data = skl.utils.shuffle(pd_data)
    n_obs = pd_data.shape[0]
    ind_train = int(n_obs*0.7)
    train_data = pd_data.iloc[:ind_train,:]
    test_data = pd_data.iloc[ind_train:,:]
    header = [("x"+str(i)) for i in range(1,pd_data.shape[1]+1)]
    header[0] = "# " + header[0]
    header[-1] = "y"
    # train_data.to_csv("Train.csv", index = False, header = header)
    # test_data.to_csv("Test.csv", index = False, header = header)
    return

if __name__ == '__main__':
    main()