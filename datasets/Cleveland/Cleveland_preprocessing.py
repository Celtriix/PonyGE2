# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 14:20:24 2022

@author: Dominik
"""

import numpy as np
import pandas as pd
import sklearn as skl

def main():
    df = pd.read_csv("Cleveland.csv", header = None)
    df = df[(df.iloc[:,11] != " <null>") & (df.iloc[:,12] != " <null>")]
    cols = ["x"+str(i) for i in range(df.shape[1])]
    cols[-1] = "y"
    df.columns = cols
    df_out = df
    df_out.to_csv("Cleveland_clean_unseeded.csv", header = True, index = False)
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
    train_data.to_csv("Train_unseeded.csv", index = False, header = header)
    test_data.to_csv("Test_unseeded.csv", index = False, header = header)
    return

if __name__ == '__main__':
    main()