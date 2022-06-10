#! /usr/bin/env python

# PonyGE2
# Copyright (c) 2017 Michael Fenton, James McDermott,
#                    David Fagan, Stefan Forstenlechner,
#                    and Erik Hemberg
# Hereby licensed under the GNU GPL v3.
""" Python GE implementation """

from utilities.algorithm.general import check_python_version

check_python_version()

from stats.stats import get_stats
from algorithm.parameters import params, set_params, load_params
import sys, os
import shutil
import datetime


# Directory of the parameter file that is to be used
param_file = "GEDT_Experiments/GEDT_Vehicle_500Gen.txt"

def mane():
    """ Run program """
    print(f"Beginning with {param_file}")
    # Load from parameter file
    dir = os.path.dirname(__file__)[:-3] + "parameters/"
    load_params(dir+param_file)
    
    # Set up directories for the output of the runs
    if params["SAVE_POP"]:
        # Set output path for individual export
        params["FILE_PATH"] = os.path.dirname(__file__)[:-3] + "results/" + params["EXPERIMENT_NAME"] + "/"
        # Create directory for results if it does not yet exist
        params["SAVE_LOC"] = params["FILE_PATH"] + "Runs/"
        try:
            os.mkdir(params["FILE_PATH"])
        except(FileExistsError):
            pass
        try:
            os.mkdir(params["SAVE_LOC"])
        except(FileExistsError):
            pass
        # Copy the parameter file to the output directory for safekeeping
        shutil.copy2(dir+param_file, params["FILE_PATH"])
    
    # Repeat the evolution for the number of runs
    for run in range(params["RUNS"]):
        tic = datetime.datetime.now()
        # Change the runID parameter and export file
        params["RUN_ID"] = run
        params["FILE_NAME"] = f'Run{str(run)}.csv'
        # Run evolution
        individuals = None
        individuals = params['SEARCH_LOOP']()
        # Print final review
        get_stats(individuals, end=True)
        toc = datetime.datetime.now()-tic
        print(f'done with run {run} in {toc}')
        print(f'Finished at approximately {datetime.datetime.now() + (params["RUNS"]-run-1)*toc}')

if __name__ == "__main__":
    mane()
