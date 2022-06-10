# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 17:29:59 2022

@author: Dominik
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("data_banknote_authentication.csv", header = None)
df.columns = ["x0", "x1", "x2", "x3", "y"]
print(np.unique(df["y"], return_counts=True))
df["y"][df["y"] == 0] = -1.0
df["y"][df["y"] == 1] = 1.0
df.to_csv("data_banknote_authentication.csv")