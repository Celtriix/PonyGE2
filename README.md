# Ensemble Creation with Grammatical Evolution Decision Trees

This document goes over the most important changes that have been made to the distribution of the PonyGE2 library. The original distribution used in the scope of this thesis can be found under https://github.com/PonyGE/PonyGE2. The experiments in the thesis were conducted using version 0.2.0 which was the latest release at the time of writing. [1]

## General changes
The most important changes to the original implementation of Grammatical Evolution is the expansion to conduct multiple runs consecutively and the inclusion of the export of all individuals within a run. The original library has a feature in beta that aims to provide the execution of multiple runs of the evolutionary process. There were also changes made that allow for the usage of comparisons between individuals for the evaluation of the fitness of the individuals. This directory contains the needed files to replicate the experiments conducted in the scope of the thesis. Many files that were not needed were removed from the original distribution. The missing files can be included by downloading the original distribution and copying the contents of this repository to them. The changes to the PonyGE2 library were done as additions to it, without hindering the base utility.

This document is structured according to the main directories of the PonyGE2 distribution sorted by their importance.


## Src

The principal files the PonyGE2 library needs to function can be found in this directory.

The main file is called "ponyge.py" and runs all the necessary code for the generation and evolution of individuals. This file reads the necessary parameters from the specified parameter file and creates an output folder for the export of the individuals if this option is used. The runs are exported to a sub-directory to tidy up the output. The first thing written to the new directory is a copy of the parameter file for save-keeping. The runs used in this thesis are not included in this directory due to their large file size. The main method iterates over the specified number of runs and prints to the console when the program finishes with a run and at around what time all the runs will be finished.

The file used in the creation of the ensembles is called "EnsembleCreator.py". This file reads file with the exported individuals and creates the ensembles presented in the thesis. The resulting performance of the ensembles is exported into a .csv file with the name specified in the input-information section of the program.
For input, only the name of the output folder in the results directory and the name of the corresponding data set folder in the dataset directory are needed. The name of the input/output folder is inferred from the information given for the data set name, number of generations and suffix of the experiment. Those parameters need to be changed in the corresponding input parameter file in the parameter directory as explained below. To replicate the results from the thesis, no changes to the rest of the program are needed.


The "Plotter.py" file was used to create the plots in the thesis. It also creates the output in the tables of the thesis.  This file handles input of data analogously to the EnsembleCreator file. Additional sub-directories are created to hold Figures and Tables to tidy the directory containing the results of an experiment.

An additional parameter called "weighted" needs to be adjusted according to whether the weighted or unweighted versions of the ensembles should be plotted.

The tables are exported into Latex files (.tex). The tables in the thesis were created by hand from the output files. The first table contain information for the comparisons between the different methods. The second table contains the same information for the different combinations of population ensembles.

### Algorithm
There were changes made to the following files: "parameters.py", "search_loop.py" and "step.py". "parameters.py" contains the default values of the variables not specified in the parameter file loaded by the "ponyge.py" file. The file is set up to create a dictionary of the parameters in key-value pairs. Some additional key variables for use in the export of the individuals were created.
- SAVE_POP: is a boolean used to toggle whether all individuals of a run are saved
- USE_DIVERSITY: is a boolean used to chose between the two modes of export. The use of multiple objective functions necessitates the export of larger tables.

The file "search_loop.py" governs the evolutionary process and calls the step method to iterate through the number of generations in a run. If multi-objective optimization is active an additional step of evaluating the individuals in the initial populations is run. This is done to create a boolean matrix of whether the individuals made a right or wrong prediction for the observations of the training set. This matrix is stored in the parameter dictionary as "YHAT_CORRECT". The predictions of the individuals are needed to compute the pairwise Q-statistic. This represents one of the large changes to the original implementation. There only fitness functions that work on the single individuals are possible and comparisons between individuals are not available. However, this comes attached to some redundancy as the program evaluates the population of ensembles twice. Once in this additional step and once during the proper evaluation step of the ensembles. With the size of the matrix depending on population size, the memory demand of the method also increases. Another change to the original code is the export of the individuals of a generation to a pandas DataFrame that is saved to the location specified by the "ponyge.py" file.

The "step.py" file was adjusted similar to the search_loop file to enable the calcualtion of the Q-statistic. No other change was made.

### Fitness
The file "Q-statistic" enables the calculation of the Q-statistic as presented in the thesis. This file was only used in the multi-objective calibration of the experiments. The class within this file inherits from the base fitness function class "base_ff". The Q-statistic inherits the default fitness value which is set to be a missing value.

The file "moo_ff.py" in the base_ff sub-directory offers the multi-objective functionality for the evaluation of individuals. The class creates a list of single-objective classifiers that it sequentially evaluates during the evaluation step. To work with the Q-statistic the class also inputs the training and test data to the parameter dictionary.

The file "supervised_learning_numpy.py" handles the evaluation of an individual. The phenotype is evaluated resulting in the predictions of the labels for the supplied data. The number of false predictions is handled using the error metric specified in the supplied parameter file.

### Operator
An additional mutation version was added to the "mutation.py" file in this directory. The new version is called prob_flip_per_ind. The method calls a random number and checks whether it is smaller than the specified mutation probability. If this is the case a random index of the genotype is exchanged by a random integer in the range specified in the supplied parameter file.$

### Utilities/Fitness
Two additional error metrics were added to the "error_metric.py" file. Those are the "Accuracy" and the inadequately named "Miss_rate". The accuracy calculates the percentage of identical indices of two supplied vectors. The miss-rate does the same but calculates the percentage of differing indices. The experiments in the thesis used the "Miss_rate" error metric as the property of whether to maximize or minimize the error_metric seems to be ignored by the original implementation. The default is to minimize the error_metric-

The file "get_data.py" was adjusted to include the possibility to standardize the data read from the dataset directory. If the flag "STANDARDIZE" is set to be true, the standardization infrastructure of Scikit-learn is used on the training and the test set.

## Data sets

The data set directory for the original distribution of the PonyGE2 library comes with many available data sets. The Banknote data set has been adopted from the original distribution of the library. The other data sets included were removed to decrease the memory demand and facilitate easier sharing of the submission.


For the Banknote [2] data set no changes were made to the training and test set. The test data contains 27.4% of observations, so this represents a slight deviation from the 70/30 split for the other data sets. The combination of the test and training set results in the same data found on the UCI data repository.

The Iris [3] and Vehicle [4] data sets were taken form the UCI repository. For this data set the splits into training and test set were done by shuffling the original data and cutting it into the two parts. This step was not seeded, so a replication with only the base data set is not possible.

The Cleveland [5] data set can be found in the KEEL repository, which itself links back to the bigger heart disease data set on the UCI repository. The splits into training and test set was done using a copy of the code used for the previous data sets and suffer from the same problem of not being seeded.

## Grammars
The grammars used in the experiments have been saved into a separate folder with the name "GEDT_Experiments". The grammars only differ by the labels for the different classes. No other changes were made to this directory.

## Parameters
Similar to the grammar directory a folder with the name "GEDT_Experiments" was added containing the parameter files for the different calibrations used.

Important!
The flag for "DEBUG" has to be set to True. If the debug mode is turned off the original library exports information on the run into the same results folder. This is not desired as the stats logger of the library does not work with multiple runs as it can't be reset after a run.


## Sources
[1] Fenton, M., McDermott, J., Fagan, D., Forstenlechner, S., Hemberg, E., and O'Neill, M. PonyGE2: Grammatical Evolution in Python. arXiv preprint, arXiv:1703.08535, 2017. https://github.com/PonyGE/PonyGE2

[2] https://archive.ics.uci.edu/ml/datasets/banknote+authentication

[3] https://archive.ics.uci.edu/ml/datasets/iris

[4] https://archive.ics.uci.edu/ml/datasets/Statlog+%28Vehicle+Silhouettes%29

[5] https://sci2s.ugr.es/keel/dataset.php?cod=57
