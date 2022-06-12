# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 19:38:55 2022

@author: Dominik
"""

import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

# Directory of the experiment
experiment_name = "Iris_250Gen_Submission"
# File containing the ensemble output
file_name = "Ensemble_out.csv"
# Whether to use weights or not
weighted = True

# %%
results_path = os.path.dirname(__file__)[:-3] + "results/" + experiment_name
figure_path = results_path + "/Figures"
tables_path = results_path + "/Tables"
data_path = results_path + "/" + file_name
whitespace = False
np.random.seed(81197)

if experiment_name.find("500"):
    calib = 1
elif experiment_name.find("250"):
    calib = 2
else:
    calib = 3
    
data_set_name = experiment_name.split("_")[0]

try:
    os.mkdir(results_path + "/Figures")
except(FileExistsError):
    pass
try:
    os.mkdir(results_path + "/Tables")
except:
    pass


#%% Analyse the ensemble output

# Open the ensemble output
df = pd.read_csv(data_path)
df["Size_Multiplier"] = df["Ensemble_Size_post"]/df["Ensemble_Size_pre"]
df["Training_Fitness"] = df["Training_Fitness"]*100
df["Test_Fitness"] = df["Test_Fitness"]*100

# Create the plot for each percentage of kept correlated trees in the ensemble
for i in df["Similarity_Threshold"].unique()[:-1]:
    fig, axs = plt.subplots(1,2, figsize = (8,6), sharey = True)
    ens = df.loc[~df["Threshold"].isin(["RF", "Best_Ensemble", "Best_All", "DT"])]
    ens = ens.loc[(ens["Similarity_Threshold"] == i)& (df["Weighted"] == weighted),]
    ens.loc[:,"Threshold"] = ens.loc[:,"Threshold"]
    
    # Plot the performance of the ensembles
    ens = pd.concat([ens, df.loc[df["Threshold"] == "RF"]])
    ens["Similarity_Threshold"] = i
    positions = None
    positions = [1, 2, 3, 0, 4, 5]
    ens.boxplot("Training_Fitness", by = "Threshold", ax = axs[0], positions = positions)
    axs[0].set(title = "Training Fitness")
    #axs[1,0].set_xlabel("Initial Ensemble Size")
    #axs[0].set_ylim(ymin, ymax)
    # axs[1,0].hlines(df[df["Threshold"] == "Best_All"]["Training_Fitness"],
    #                 xmin = -0.5, xmax = len(np.unique(ens["Threshold"]))-0.5,
    #                 color = "g", linestyles = "dashed")
    # axs[1,0].text(len(np.unique(ens["Threshold"]))-0.4,
    #               df[df["Threshold"] == "Best_All"]["Training_Fitness"], "Best\nEnsemble")
    
    
    
    ens.boxplot("Test_Fitness", by = "Threshold", ax = axs[1], positions = positions)
    axs[1].set(title = "Testing Fitness")
    #axs[1,1].set_xlabel("Initial Ensemble Size")
    #axs[1].set_ylim(ymin, ymax)
    # axs[1,1].hlines(df[df["Threshold"] == "Best_All"]["Test_Fitness"],
    #                 xmin = -0.5, xmax = len(np.unique(ens["Threshold"]))-0.5,
    #                 color = "g", linestyles = "dashed")
    # axs[1,1].text(len(np.unique(ens["Threshold"]))-0.4,
    #               df[df["Threshold"] == "Best_All"]["Test_Fitness"], "Best\nEnsemble")
    
    plt.suptitle(f"Ensemble Analysis on the {data_set_name} data set")
    plt.savefig(figure_path+f'/{experiment_name}_Q{i}{weighted}.png')



# Close all open figures
plt.close("all")

pops = df.loc[~df["Threshold"].isin(["RF", "Best_Ensemble", "Best_All", "DT"])]
fig, axs = plt.subplots(len(pops["Similarity_Threshold"].unique()), len(pops["Threshold"].unique()), sharex = True)
pops.hist(column = "Ensemble_Size_post", by = ["Threshold", "Similarity_Threshold"], ax = axs)
fig.suptitle("Comparison of post ensemble sizes")

#%% Helper function
# Taken from https://stackoverflow.com/questions/25812255/row-and-column-headers-in-matplotlibs-subplots
def add_headers(
    fig,
    *,
    row_headers=None,
    col_headers=None,
    bottom_header = None,
    row_pad=1,
    col_pad=5,
    rotate_row_headers=True,
    **text_kwargs
):
    # Based on https://stackoverflow.com/a/25814386

    axes = fig.get_axes()

    for ax in axes:
        sbs = ax.get_subplotspec()

        # Putting headers on cols
        if (col_headers is not None) and sbs.is_first_row():
            ax.annotate(
                col_headers[sbs.colspan.start],
                xy=(0.5, 1),
                xytext=(0, col_pad),
                xycoords="axes fraction",
                textcoords="offset points",
                ha="center",
                va="baseline",
                **text_kwargs,
            )

        # Putting headers on rows
        if (row_headers is not None) and sbs.is_first_col():
            ax.annotate(
                row_headers[sbs.rowspan.start],
                xy=(0, 0.5),
                xytext=(-ax.yaxis.labelpad - row_pad, 0),
                xycoords=ax.yaxis.label,
                textcoords="offset points",
                ha="right",
                va="center",
                rotation=rotate_row_headers * 90,
                **text_kwargs,
            )
            
        if (bottom_header is not None) and sbs.is_last_row():
            ax.set_xlabel(bottom_header)
     
# Open the ensemble output
df = pd.read_csv(data_path)
df["Size_Multiplier"] = df["Ensemble_Size_post"]/df["Ensemble_Size_pre"]
df["Training_Fitness"] = df["Training_Fitness"]*100
df["Test_Fitness"] = df["Test_Fitness"]*100


# Create plot that show the training and testing accuracy in histograms
sims = df["Similarity_Threshold"].unique()[:-1]
thresholds = df["Threshold"].unique()[:5]
fig, axs = plt.subplots(len(sims), len(thresholds), sharex = True, figsize = (10, 10))
plt.subplots_adjust(top=0.91, bottom=0.1, left=0.1, right=0.9, hspace=0.1, wspace=0.1)
row_headers = ["Maximum average\n Q-statistic : -0.25", "Maximum average\n Q-statistic : 0.0", "Maximum average\n Q-statistic : 0.25","Maximum average\n Q-statistic : 0.5", "Maximum average\n Q-statistic : 0.75"]
column_headers = ["50 Individuals", "100 Individuals", "200 Individuals", "250 Individuals", "500 Individuals"]
add_headers(fig, row_headers = row_headers, col_headers = column_headers, bottom_header = "Accuracy in %")
for i in range(len(sims)):
    for j in range(len(thresholds)):
        sub_df = df[(df["Similarity_Threshold"] == sims[i]) & (df["Threshold"] == thresholds[j])&(df["Weighted"] == weighted)][["Training_Fitness", "Test_Fitness"]]
        train = axs[i,j].hist(sub_df["Training_Fitness"], alpha=0.3, color = "b", bins = 10)
        test = axs[i,j].hist(sub_df["Test_Fitness"], alpha=0.3, color = "g", bins = 10)
        highest_count = max(test[0].max(), train[0].max())
        test_median = round(sub_df["Test_Fitness"].median(),1)
        train_median = round(sub_df["Training_Fitness"].median(),1)
        axs[i,j].vlines(train_median, ymin = 0, ymax = highest_count, color = "b")
        axs[i,j].vlines(test_median, ymin = 0, ymax = highest_count, color = "g")
        if train_median > 40 or test_median > 40:
            axs[i,j].annotate(f'Train: {round(train_median, 1)}', xy = (0.01, 0.9), xycoords = "axes fraction", color = "darkblue")
            axs[i,j].annotate(f'Test: {round(test_median, 1)}', xy = (0.01, 0.8),xycoords = "axes fraction" , color = "darkgreen")
        else:
            axs[i,j].annotate(f'Train: {round(train_median, 1)}', xy = (0.5, 0.9), xycoords = "axes fraction", color = "darkblue")
            axs[i,j].annotate(f'Test: {round(test_median, 1)}', xy = (0.5, 0.8),xycoords = "axes fraction" , color = "darkgreen")
        
        axs[i,j].tick_params(axis='y', which='both', labelleft = False, left = False)

if weighted:   
    plt.suptitle(f"Training and test fitness of the population ensembles on the\n{data_set_name} data set using a weighted majority vote", x = 0.1, y = 0.975, ha = "left")
else:
    plt.suptitle(f"Training and test fitness of the population ensembles on the\n{data_set_name} data set using an unweighted majority vote", x = 0.1, y = 0.975, ha = "left")
fig.legend(labels = ["Median Training Fitness", "Median Test Fitness"], loc = (0.7,0.9375))
plt.savefig(figure_path+f'/{experiment_name}_Histograms_{weighted}_median.png', bbox_inches='tight')

#%% Compare the best ensembles to RF

df = pd.read_csv(data_path)
best_individuals = pd.read_csv(results_path+"/Best_Individuals.csv")
best_individuals = best_individuals.drop(columns = ["Phenotype"])
df = pd.concat([df, best_individuals])
df["Threshold"][~df["Threshold"].isin(["RF", "Best_Ensemble", "DT", "Single_Individual"])] = "Population_Ensemble"
df = df[df["Threshold"].isin(["RF", "Best_Ensemble", "DT", "Single_Individual", "Population_Ensemble"])]
df["Training_Fitness"] = df["Training_Fitness"]*100
df["Test_Fitness"] = df["Test_Fitness"]*100


fig, axs = plt.subplots(1,2, figsize = (9,5), sharey = True)
plt.subplots_adjust(top=0.89, bottom=0.08, left=0.085,right=0.995, hspace=0.2,wspace=0.02)
axs[0].boxplot([df[(df["Threshold"]=="Best_Ensemble")&(df["Weighted"] == False)]["Training_Fitness"],
                df[(df["Threshold"]=="Best_Ensemble")&(df["Weighted"] == True)]["Training_Fitness"],
                df[(df["Threshold"]=="Population_Ensemble")&(df["Weighted"] == False)]["Training_Fitness"],
                df[(df["Threshold"]=="Population_Ensemble")&(df["Weighted"] == True)]["Training_Fitness"],
                best_individuals["Training_Fitness"]*100,
                df[df["Threshold"]=="RF"]["Training_Fitness"],
                df[df["Threshold"]== "DT"]["Training_Fitness"]])
axs[0].set_title("Training accuracy")
axs[0].set_ylabel("Accuracy in %")
axs[0].set_xticklabels(["EnsBest\n", "EnsBest\nweighted", "EnsPop\n","EnsPop\nweighted", "GEDT", "RF", "DT"])

axs[1].boxplot([df[(df["Threshold"]=="Best_Ensemble")&(df["Weighted"] == False)]["Test_Fitness"],
                df[(df["Threshold"]=="Best_Ensemble")&(df["Weighted"] == True)]["Test_Fitness"],
                df[(df["Threshold"]=="Population_Ensemble")&(df["Weighted"] == False)]["Test_Fitness"],
                df[(df["Threshold"]=="Population_Ensemble")&(df["Weighted"] == True)]["Test_Fitness"],
                best_individuals["Test_Fitness"]*100,
                df[df["Threshold"]=="RF"]["Test_Fitness"],
                df[df["Threshold"]== "DT"]["Test_Fitness"]])

axs[1].set_title("Test accuracy")
axs[1].set_xticklabels(["EnsBest\n", "EnsBest\nweighted", "EnsPop\n","EnsPop\nweighted", "GEDT", "RF", "DT"])
plt.suptitle(f"Accuracy comparison for the {data_set_name} data set using calibration {calib}")
plt.savefig(figure_path+f'/{experiment_name}_Individual_Comparison.png')


# Create the comparison of methods for export
cols = [["Best_Ensemble", False], ["Best_Ensemble", True], ["Population_Ensemble", False], ["Population_Ensemble", True], ["Single_Individual", False], ["RF", False], ["DT", False]]
df_pivot_train = df.pivot_table(columns = ["Threshold", "Weighted"], values = "Training_Fitness", aggfunc = np.median)
df_pivot_train = df_pivot_train[cols]
df_pivot_train_STD = df.pivot_table(columns = ["Threshold", "Weighted"], values = "Training_Fitness", aggfunc = np.std).round(decimals = 1)
df_pivot_train_STD = df_pivot_train_STD[cols]
df_pivot_test = df.pivot_table(columns = ["Threshold", "Weighted"], values = "Test_Fitness", aggfunc = np.median)
df_pivot_test = df_pivot_test[cols]
df_pivot_test_STD = df.pivot_table(columns = ["Threshold", "Weighted"], values = "Test_Fitness", aggfunc = np.std).round(decimals = 1)
df_pivot_test_STD = df_pivot_test_STD[cols]
df_pivot_merge = pd.concat([df_pivot_train, df_pivot_train_STD, df_pivot_test, df_pivot_test_STD])
df_pivot_merge.to_latex(tables_path + "/Fitness_Comparison_median.tex", float_format="{:0.1f}".format)

#%% Create the cross tables for export
df = pd.read_csv(data_path)
df = df[~df["Threshold"].isin(["RF", "Best_Ensemble", "DT"])]
df["Threshold"] = df["Threshold"].astype(int)
df["Training_Fitness"] = df["Training_Fitness"]*100
df["Test_Fitness"] = df["Test_Fitness"]*100

df_pivot_train = df.pivot_table(index = ["Similarity_Threshold", "Weighted"], columns = "Threshold", values = "Training_Fitness", aggfunc = np.median).round(decimals = 3)
df_pivot_train_STD = df.pivot_table(index = ["Similarity_Threshold", "Weighted"], columns = "Threshold", values = "Training_Fitness", aggfunc = np.std).round(decimals = 3)
df_pivot_test = df.pivot_table(index = ["Similarity_Threshold", "Weighted"], columns = "Threshold", values = "Test_Fitness", aggfunc = np.median).round(decimals = 3) 
df_pivot_test_STD = df.pivot_table(index = ["Similarity_Threshold", "Weighted"], columns = "Threshold", values = "Test_Fitness", aggfunc = np.std).round(decimals = 3) 

df_pivot_merge = pd.concat([df_pivot_train, df_pivot_train_STD, df_pivot_test, df_pivot_test_STD])
df_pivot_merge.to_latex(tables_path + "/Fitness_popEnsemble_median.tex", float_format="{:0.1f}".format)


plt.close("all")


# End of File