# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 16:31:14 2022

@author: Dominik
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import tree
import os
import itertools

# Data source
data_name = "Iris"

# Input information
nGen = 250
suffix = "Submission" # "Moo"
file_name = "Ensemble_out_Q_combined.csv"

# Create the name of the input/output folder
experiment_name = f"{data_name}_{nGen}Gen_{suffix}"

# Parameters
np.random.seed(81197)
n_Runs = 50
use_weights = True
both_weights = True
riffle = True

# Threshold of one tree to be too correlated with another
max_corr = [-0.25, 0.0, 0.25, 0.5, 0.75]
# Percentage of correlated trees to keep
flip_pct = np.array([0.0])
# Number of initial candidates in an ensemble
thresholds_abs = np.array([50, 100, 200, 250, 500])


# Data input
standardize = True
out_path = os.path.dirname(__file__)[:-3] + "results/" + experiment_name
data_path = os.path.dirname(__file__)[:-3] + "datasets/" +  data_name
test_file_name = "Test.csv"
train_file_name = "Train.csv"
test_file= data_path + "/" + test_file_name
train_file = data_path + "/" + train_file_name
# Set separator
whitespace= False
if data_name == "Banknote":
    whitespace = True

# Infer the number of runs to be mixed
n_mix = 1 if nGen == 500 else 2

#%%
def main():
    pd_pop_out = pd.DataFrame()
    best_ensemble = pd.DataFrame()
    for run in range(n_Runs):
        pd_pop, best_individual = mix_runs(run, n_mix, n_Runs)
        best_ensemble = pd.concat([best_ensemble, best_individual])
        pd_pop_run = create_pop_ensemble_max(thresholds_abs, pd_pop, max_corr, riffle, True)
        pd_pop_out  = pd.concat([pd_pop_out, pd_pop_run])
        print(f"Done with run {run}")
    best_individuals = evaluate_best_individuals(best_ensemble.drop_duplicates(["Phenotype", "Run_ID"]))
    best_individuals.to_csv(out_path + "/Best_Individuals.csv", index = False)
    pd_pop_out = pd.concat([pd_pop_out,
                            evaluate_best_ensemble(best_individuals, use_weights=True),
                            evaluate_RF(n_Runs),
                            evaluate_DT(n_Runs)])
    pd_pop_out.to_csv(out_path + "/" + file_name, index = False)



def create_pop_ensemble_max(thresholds, df, max_corr, riffle = False, use_weights = False):
    # Import the training data set
    data_train = pd.read_csv(train_file, delim_whitespace = whitespace).to_numpy()
    y_train = data_train[:,-1]
    x_train = data_train[:,:-1]
    # Import the test data set
    data_test = pd.read_csv(test_file, delim_whitespace = whitespace).to_numpy()
    y_test = data_test[:,-1]
    x_test = data_test[:,:-1]
    if riffle and n_mix > 1:
        df = riffle_shuffle(df, thresholds[-1])
    else:
        df = df.sort_values("Fitness", ascending = False)
        df = df.iloc[0:thresholds[-1],]
    # Get the predictions on the training set for all candidate individuals
    yhat_train = evaluate_trees(df["Phenotype"].to_numpy(), y_train, x_train)
    # Get the predictions on the test set for all candidate individuals
    yhat_test = evaluate_trees(df["Phenotype"].to_numpy(), y_test, x_test)
    # Create the output dict
    df_out = {}
    for t, corr in list(itertools.product(thresholds, max_corr)):
        ens_out = {}
        # Extract the ensemble votes
        ensemble_votes_train = yhat_train[0:t]
        ens_out["Ensemble_Size_pre"] = ensemble_votes_train.shape[0]
        ensemble_votes_test = yhat_test[0:t]
        # Treat the similarity in the predictions
        too_similar = treat_similarity_Q(ensemble_votes_train, flip_pct, corr, y_train)
        if np.sum(~too_similar) == 0:
            n_best = df[df["Fitness"] == df["Fitness"].max()].shape[0]
            too_similar[np.random.randint(0, min(n_best, t))] = False
        if both_weights:
            ensemble_votes_test = ensemble_votes_test[~too_similar]
            ensemble_votes_train = ensemble_votes_train[~too_similar]
            # Get the majority vote for the remaining individuals in the ensemble
            majority_votes_train, weights = create_majority_vote(ensemble_votes_train, True, y_train, None)
            majority_votes_test = create_majority_vote(ensemble_votes_test, True, y_test, weights)
            # Logging
            ens_out["Threshold"] = t
            ens_out["Weighted"] = True
            ens_out["Similarity_Threshold"] = corr
            ens_out["Ensemble_Size_post"] = np.sum(~too_similar)
            ens_out["Training_Fitness"] = accuracy(y_train, majority_votes_train)
            ens_out["Test_Fitness"] = accuracy(y_test, majority_votes_test)
            df_out[f'{t}_{int(corr*100)}_weighted'] = ens_out.copy()

            majority_votes_train = create_majority_vote(ensemble_votes_train, False, y_train, None)
            majority_votes_test = create_majority_vote(ensemble_votes_test, False, y_test, None)
            # Logging
            ens_out["Weighted"] = False
            ens_out["Training_Fitness"] = accuracy(y_train, majority_votes_train)
            ens_out["Test_Fitness"] = accuracy(y_test, majority_votes_test)
            df_out[f'{t}_{int(corr*100)}_unweighted'] = ens_out.copy()
        else:
            ensemble_votes_test = ensemble_votes_test[~too_similar]
            ensemble_votes_train = ensemble_votes_train[~too_similar]
            # Get the majority vote for the remaining individuals in the ensemble
            majority_votes_train = create_majority_vote(ensemble_votes_train, True, y_train)
            majority_votes_test = create_majority_vote(ensemble_votes_test, True, y_test)
            # Logging
            ens_out["Threshold"] = t
            ens_out["Weighted"] = use_weights
            ens_out["Similarity_Threshold"] = corr
            ens_out["Ensemble_Size_post"] = np.sum(~too_similar)
            ens_out["Training_Fitness"] = accuracy(y_train, majority_votes_train)
            ens_out["Test_Fitness"] = accuracy(y_test, majority_votes_test)
            df_out[f'{t}_{int(corr*100)}_{use_weights}'] = ens_out.copy()

    df_out = pd.DataFrame.from_dict(df_out, orient = "index")
    return df_out
    
#%% Evaluation of the best ensemble
def evaluate_single_best(best_individuals, data_train, data_test, use_weights, n_Replications = 50):
    y_train = data_train[:,-1]
    x_train = data_train[:,:-1]
    y_test = data_test[:,-1]
    x_test = data_test[:,:-1]
    df_best_out = {}
    for rep in range(n_Replications):
        ens = []
        for i in range(n_Runs):
            run = best_individuals[best_individuals["Run_ID"] == i]
            ind = np.random.randint(low = 0, high = run.shape[0])
            ens.append(run.iloc[ind,])
        ensemble = pd.DataFrame(ens)
        if both_weights:
            # Get the predictions on the training set for all individuals
            yhat_train = evaluate_trees(ensemble["Phenotype"].to_numpy(), y_train, x_train)
            # Get the predictions on the test set for all individuals
            yhat_test = evaluate_trees(ensemble["Phenotype"].to_numpy(), y_test, x_test)
            df_best_run = {}
            majority_votes_train, weights = create_majority_vote(yhat_train, True, y_train, None)
            majority_votes_test = create_majority_vote(yhat_test, True, None, weights)
            df_best_run["Training_Fitness"] = accuracy(y_train, majority_votes_train)
            df_best_run["Test_Fitness"] = accuracy(y_test, majority_votes_test)
            df_best_run["Ensemble_Size_pre"] = ensemble.shape[0]
            df_best_run["Ensemble_Size_post"] = yhat_train.shape[0]
            df_best_run["Threshold"] = "Best_Ensemble"
            df_best_run["Weighted"] = True
            df_best_out[f'Run_{rep}_weighted'] = df_best_run.copy()

            majority_votes_train = create_majority_vote(yhat_train, False, None, None)
            majority_votes_test = create_majority_vote(yhat_test, False, None, None)
            df_best_run["Training_Fitness"] = accuracy(y_train, majority_votes_train)
            df_best_run["Test_Fitness"] = accuracy(y_test, majority_votes_test)
            df_best_run["Threshold"] = "Best_Ensemble"
            df_best_run["Weighted"] = False
            df_best_out[f'Run_{rep}_unweighted'] = df_best_run.copy()
        else:
            # Get the predictions on the training set for all individuals
            yhat_train = evaluate_trees(ensemble["Phenotype"].to_numpy(), y_train, x_train)
            # Get the predictions on the test set for all individuals
            yhat_test = evaluate_trees(ensemble["Phenotype"].to_numpy(), y_test, x_test)
            df_best_run = {}
            majority_votes_train = create_majority_vote(yhat_train, use_weights, y_train)
            majority_votes_test = create_majority_vote(yhat_test, use_weights, y_test)
            df_best_run["Training_Fitness"] = accuracy(y_train, majority_votes_train)
            df_best_run["Test_Fitness"] = accuracy(y_test, majority_votes_test)
            df_best_run["Ensemble_Size_pre"] = ensemble.shape[0]
            df_best_run["Ensemble_Size_post"] = yhat_train.shape[0]
            df_best_run["Threshold"] = "Best_Ensemble"
            df_best_out[f'Run-{rep}'] = df_best_run
    df_best_out = pd.DataFrame.from_dict(df_best_out, orient = "index")
    return df_best_out


def evaluate_best_ensemble(ensemble, use_weights = False):
    ensemble = ensemble.drop_duplicates("Phenotype")
    # Import the training and testing data set
    data_train = pd.read_csv(train_file, delim_whitespace = whitespace).to_numpy()
    data_test = pd.read_csv(test_file, delim_whitespace = whitespace).to_numpy()
    df_best_out = evaluate_single_best(ensemble, data_train, data_test, use_weights)
    print("Done with single best ensembles")
    return df_best_out

#%% Evaluation of the best individuals of the runs
def evaluate_best_individuals(best_ensemble):
    best_individuals = best_ensemble
    test = pd.read_csv(test_file, delim_whitespace = whitespace).to_numpy()
    x_test = test[:,:-1]
    y_test = test[:,-1]
    yhat_test = evaluate_trees(best_individuals["Phenotype"].to_numpy(), y_test, x_test)
    test_acc = np.zeros(best_individuals.shape[0])
    for i in range(best_individuals.shape[0]):
         test_acc[i] = accuracy(y_test, yhat_test[i])

    best_individuals["Training_Fitness"] = best_individuals["Fitness"]
    best_individuals.drop("Fitness", axis = "columns")
    best_individuals["Test_Fitness"] = test_acc
    best_individuals = best_individuals[["Run_ID", "Training_Fitness", "Test_Fitness", "Phenotype"]]
    best_individuals["Threshold"] = "Single_Individual"
    best_individuals["Weighted"] = False
    return best_individuals

#%% Evaluation of comparison methods

def evaluate_RF(n_Runs):
    np.random.seed(1)
    train = pd.read_csv(train_file, delim_whitespace = whitespace).to_numpy()
    test = pd.read_csv(test_file, delim_whitespace = whitespace).to_numpy()
    x_train = train[:,:-1]
    y_train = train[:,-1]
    x_test = test[:,:-1]
    y_test = test[:,-1]
    RF_out = {}
    for i in range(n_Runs):
        # Create a classifier
        clf = RandomForestClassifier(random_state=None, n_estimators = 50, max_depth = 10)
        # Fit the model
        clf.fit(x_train,y_train)
        # Evaluate on the training-set
        RF_run = {}
        RF_run["Training_Fitness"] = clf.score(x_train, y_train)
        # Evaluate on the test-set
        RF_run["Test_Fitness"] = clf.score(x_test, y_test)
        RF_run["Threshold"] ="RF"
        RF_run["Weighted"] = False
        RF_run["Ensemble_Size_pre"] = clf.n_estimators
        RF_run["Ensemble_Size_post"] = clf.n_estimators
        RF_out[f'{i}'] = RF_run
    df_out = pd.DataFrame.from_dict(RF_out, orient = "index")
    print("Done with Random Forest")
    return df_out

def evaluate_DT(n_Runs):
    np.random.seed(1)
    out = {}
    # Import the training data set
    data_train = pd.read_csv(train_file, delim_whitespace = whitespace).to_numpy()
    y_train = data_train[:,-1]
    x_train = data_train[:,:-1]
    # Import the test data set
    data_test = pd.read_csv(test_file, delim_whitespace = whitespace).to_numpy()
    y_test = data_test[:,-1]
    x_test = data_test[:,:-1]
    for i in range(n_Runs):
        ind = {}
        clf = tree.DecisionTreeClassifier(random_state = None, max_depth = 10)
        clf= clf.fit(x_train, y_train)
        clf_train = clf.score(x_train, y_train)
        clf_test = clf.score(x_test, y_test)
        ind["Training_Fitness"] = clf_train
        ind["Test_Fitness"] = clf_test
        ind["Threshold"] = "DT"
        ind["Weighted"] = False
        ind["Ensemble_Size_pre"] = None
        ind["Ensemble_Size_post"] = None
        out[f'{i}'] = ind
    df_out = pd.DataFrame.from_dict(out, orient = "index")
    print("Done with Decision Trees")
    return df_out

df_DT = evaluate_DT(50)
print(df_DT[["Training_Fitness", "Test_Fitness"]].mean())
print(df_DT[["Training_Fitness", "Test_Fitness"]].std())

df_RF = evaluate_RF(50)
print(df_RF[["Training_Fitness", "Test_Fitness"]].mean())
print(df_RF[["Training_Fitness", "Test_Fitness"]].std())



# %% Helper functions
def treat_similarity_Q(yhat_train, flip_pct, max_corr, y):
    Q_out = np.zeros(yhat_train.shape[0])
    yhat_correct = yhat_train == y
    for i in range(yhat_correct.shape[0]):
        yhat_ind = yhat_correct[i]
        N00 = np.sum(np.logical_and(yhat_correct, yhat_ind), axis = 1)
        N11 = np.sum(np.logical_and(yhat_correct == False, yhat_ind == False), axis = 1)
        N10 = np.sum(np.logical_and(yhat_correct == True, yhat_ind == False), axis = 1)
        N01 = np.sum(np.logical_and(yhat_correct == False, yhat_ind == True), axis = 1)
        Q_out[i] = (np.mean((N00*N11 - N01*N10) /(N00*N11 + N01*N10)) - 1/yhat_correct.shape[0])
    too_similar = Q_out > max_corr
    if flip_pct == 0:
        return too_similar
    for i in range(too_similar.shape[0]):
        if too_similar[i] and np.random.rand() < flip_pct:
            too_similar[i] = ~too_similar[i]
    return too_similar



def mix_runs(run_ID, n_mix, n_others):
    # Import the individuals from the output file
    results_path = os.path.dirname(__file__)[:-3] + "results/" + experiment_name + "/Runs/"
    filepath = results_path + "/Run" + str(run_ID) + ".csv"
    df = pd.read_csv(filepath_or_buffer = filepath, sep = ";")
    df["Run_ID"] = run_ID
    # Exchange the miss-rate with the accuracy
    df["Fitness"] = 1-df["Fitness"]
    df = df[df["Fitness"]>0]
    # Export the best tree in the run
    best_individual = df.loc[df["Fitness"] == df["Fitness"].max()][["Phenotype", "Fitness"]]
    best_individual["Run_ID"] = run_ID
    # Add other runs to the candidates
    ID_other = None
    for i in range(n_mix-1):
        ID_other = np.random.randint(0, n_others)
        if ID_other == run_ID and run_ID < n_others -1:
            ID_other += 1
        elif ID_other == run_ID and run_ID == n_others-1:
            ID_other -= 1
        filepath_2 = results_path + "/Run" + str(ID_other) + ".csv"
        df_other = pd.read_csv(filepath_or_buffer = filepath_2, sep = ";")
        # Exchange the miss-rate with the accuracy
        df_other["Fitness"] = 1-df_other["Fitness"]
        df_other = df_other[df_other["Fitness"]>0]
        df_other["Run_ID"] = ID_other
        df = pd.concat([df, df_other])
    print(f'Mixed runs {run_ID} and {ID_other}')
    # Remove Duplicates
    df["Fitness"] = df["Fitness"].round(4)
    return (df, best_individual)

def riffle_shuffle(df, threshold):
    df_out = [None]*threshold
    runs = np.unique(df["Run_ID"])
    out1 = df.loc[df["Run_ID"] == runs[0]]
    out1 = out1.sort_values("Fitness", ascending = False).iloc[0:threshold//2]
    out2 = df.loc[df["Run_ID"] == runs[1]]
    out2 = out2.sort_values("Fitness", ascending = False).iloc[0:threshold//2]
    for i in range(0,threshold//2):
        df_out[2*i] = out1.iloc[i,]
        df_out[2*i+1] = out2.iloc[i,]
    return pd.DataFrame(df_out)

def evaluate_trees(trees, y, x):
    if standardize:
        scaler_x = StandardScaler().fit(x)
        x = scaler_x.transform(x)
    n_individuals = trees.shape[0]
    n_observations = y.shape[0]
    yhat = np.zeros((n_individuals,n_observations))
    for i in range(n_individuals):
        ind = trees[i]
        yhat[i] = eval(ind)
    return yhat

def accuracy(y, yhat):
    fit = np.sum(yhat == y) / len(y)
    fit = np.round(fit, 4)
    return fit

def create_majority_vote(yhat, use_weights = False, y = None, weights = None):
    vote = np.zeros(yhat.shape[1])
    if use_weights and weights is None: # Create weights from training set
        accuracies = np.zeros(yhat.shape[0])
        # Create the linear weights in the ensemble
        for i in range(yhat.shape[0]):
            accuracies[i] = accuracy(y, yhat[i])
        weights = accuracies/np.sum(accuracies)
        # Create the weighted votes
        for i in range(yhat.shape[1]):
            unique = np.unique(yhat[:,i])
            weighted_votes = dict(zip(np.unique(y), [0]*len(np.unique(y))))
            for value in unique:
                weighted_votes[value] = np.sum(weights[yhat[:,i] == value])
            vote[i] = max(weighted_votes, key=weighted_votes.get)
        return (vote, weights)
    elif use_weights: # use weights from training set on test set
        for i in range(yhat.shape[1]):
            unique = np.unique(yhat[:,i])
            weighted_votes = dict(zip(np.unique(y), [0]*len(np.unique(y))))
            for value in unique:
                weighted_votes[value] = np.sum(weights[yhat[:,i] == value])
            vote[i] = max(weighted_votes, key=weighted_votes.get)
        return vote
    else: # no weighting applied
        for i in range(yhat.shape[1]):
            unique, counts = np.unique(yhat[:,i], return_counts = True)
            vote[i] = unique[np.argmax(counts)]
        return vote

#%%
if __name__ == '__main__':
    main()

# End of File
