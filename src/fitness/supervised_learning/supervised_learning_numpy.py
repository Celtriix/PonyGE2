import numpy as np

np.seterr(all="raise")

from algorithm.parameters import params
from utilities.fitness.get_data import get_data
from utilities.fitness.math_functions import *
from utilities.fitness.optimize_constants import optimize_constants

from fitness.base_ff_classes.base_ff import base_ff


class supervised_learning_numpy(base_ff):
    """
    Fitness function for supervised learning, ie regression and
    classification problems. Given a set of training or test data,
    returns the error between y (true labels) and yhat (estimated
    labels).

    We can pass in the error metric and the dataset via the params
    dictionary. Of error metrics, eg RMSE is suitable for regression,
    while F1-score, hinge-loss and others are suitable for
    classification.

    This is an abstract class which exists just to be subclassed:
    should not be instantiated.
    """

    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()
        
        # Get training and test data
        self.training_in, self.training_exp, self.test_in, self.test_exp = \
            get_data(params['DATASET_TRAIN'], params['DATASET_TEST'])

        # Find number of variables.
        self.n_vars = np.shape(self.training_in)[1] # sklearn convention

        # Regression/classification-style problems use training and test data.
        if params['DATASET_TEST']:
            self.training_test = True

    def evaluate(self, ind, **kwargs):
        """
        Note that math functions used in the solutions are imported from either
        utilities.fitness.math_functions or called from numpy.

        :param ind: An individual to be evaluated.
        :param kwargs: An optional parameter for problems with training/test
        data. Specifies the distribution (i.e. training or test) upon which
        evaluation is to be performed.
        :return: The fitness of the evaluated individual.
        """

        dist = kwargs.get('dist', 'training')

        if dist == "training":
            # Set training datasets.
            x = self.training_in
            y = self.training_exp

        elif dist == "test":
            # Set test datasets.
            x = self.test_in
            y = self.test_exp

        else:
            raise ValueError("Unknown dist: " + dist)

        shape_mismatch_txt = """Shape mismatch between y and yhat. Please check
that your grammar uses the `x[:, 0]` style, not `x[0]`. Please see change
at https://github.com/PonyGE/PonyGE2/issues/130."""


        yhat = np.zeros(y.shape)
        yhat = eval(ind.phenotype)
        if type(yhat) == int:
            yhat = yhat*np.ones(y.shape)
        ind.yhat = yhat
        assert np.isrealobj(yhat)
        # Phenotypes that don't refer to x are constants, ie will
        # return a single value (not an array). That will work
        # fine when we pass it to our error metric, but our shape
        # mismatch check (re x[:, 0] v x[0]) will check the
        # shape. So, only run it when yhat is an array, not when
        # yhat is a single value. Note np.isscalar doesn't work
        # here, see help(np.isscalar).
        if np.ndim(yhat) != 0:
            if y.shape != yhat.shape:
                raise ValueError(shape_mismatch_txt)

        # let's always call the error function with the true
        # values first, the estimate second
        fit = params['ERROR_METRIC'](y, yhat)
        # print(fit)
        fit = fit.round(decimals = 3)
        return fit
            