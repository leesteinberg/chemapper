import pandas as pd
import numpy as np
from sklearn.base import clone
import copy

class ActiveLearner:
    def __init__(self, X_train, X_test, y_train, y_test, model, learning_strategy, chosen_indices=None):
        """
        Create an ActiveLearner instance

        Parameters
        ----------
        X_train: np.ndarray(n_train, n_dim)
            Potential training data for the model
        X_test: np.ndarray(n_test, n_dim)
            The test set for the model
        y_train: np.ndarray(n_train, n_out)
            The output values for the training set
        y_test: np.ndarray(n_test, n_dim)
            The output values for the test set
        model:
            The model which is trying to be learnt
        learning_strategy: func(model, X_chosen, y_chosen, **kwargs)
            The learning strategy for the model. Should take a model, X values, y values as input, and
            return a list of indices to be chosen --- Even a single index should be returned as a list
        chosen_indices: array-like, default=None
            The initial chosen indices. If None, default indices will be chosen randomly, as 10%
            of the training data
        """

        # If X or y are 1 dimensional, do a reshape on them
        if len(X_train.shape) == 1:
            X_train = X_train.reshape(-1, 1)
            X_test = X_test.reshape(-1, 1)
        if len(y_train.shape) == 1:
            y_train = y_train.reshape(-1, 1)
            y_test = y_test.reshape(-1, 1)

        # Check that the dimensions of the training and test sets make sense
        try:
            assert len(X_train[:, 0]) == len(y_train[:, 0])  # training sets should match n_obs
            assert len(X_test[:, 0]) == len(y_test[:, 0])  # test sets should match n_obs
            assert len(X_train[0, :]) == len(X_test[0, :])  # X sets should match n_dim
            assert len(y_train[0, :]) == len(y_test[0, :])  # y sets should match n_dim
        except AssertionError:
            print('Array dimensions do not make sense.')
            print('Please check that X and y are of the correct dimensions')
            raise AssertionError

        # Load data into object
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = copy.deepcopy(model)
        self.learning_strategy = learning_strategy

        # Get useful parameters
        self.n_train = len(X_train[:, 0])
        self.n_test = len(X_test[:, 0])

        # Get initial set of idx_chosen
        self.idx_chosen = np.zeros(self.n_train, dtype=bool)
        if chosen_indices is None:
            print('Picking random indices')
            print(chosen_indices)
            # Randomly pick 10% of the training set to be the chosen indices
            chosen_indices = np.random.choice(np.arange(self.n_train, dtype=int),
                                              size=int(0.1 * self.n_train), replace=False)
        # Set idx_chosen[i] to be True for i in chosen_indices
        self.idx_chosen[chosen_indices] = True

        # Also get unchosen indices
        self.unchosenFromChosen()

        # Set up cycles_known, which counts how many cycles we've known an observation for
        self.cycles_known = np.zeros(self.n_train)
        self.cycles_known[self.idx_chosen] += 1

        # Initial fit
        self.fit()
        self.n_cycles = 0

    def fit(self):
        """
        Fit the model based on the currently chosen training indices
        """
        self.model.fit(self.X_train[self.idx_chosen, :], self.y_train[self.idx_chosen, :])

    def unchosenFromChosen(self):
        """
        Get the unchosen indices from the chosen indices
        """
        self.n_chosen = self.idx_chosen.sum()
        self.idx_train_chosen = np.arange(self.n_train, dtype=int)[self.idx_chosen]
        # The actual chosen indices, rather than True/False for each index

        # Get initial set of idx_unchosen
        self.idx_unchosen = np.logical_not(self.idx_chosen)
        self.idx_train_unchosen = np.arange(self.n_train, dtype=int)[self.idx_unchosen]

    def findNewChosen(self, **kwargs):
        """
        Run the learning strategy, and add the new indices to chosen_indices
        """
        new_indices = self.learning_strategy(self.model, self.X_train[self.idx_unchosen],
                                             self.y_train[self.idx_unchosen], **kwargs)
        new_idx_chosen = self.idx_train_unchosen[new_indices]

        # Add new chosen indices to idx_chosen
        self.idx_chosen[new_idx_chosen] = True

        # Also find new unchosen()
        self.unchosenFromChosen()

    def learning_cycle(self, **kwargs):
        """
        Learn new indices, then refit the model
        """
        # Learn new indices
        self.findNewChosen(**kwargs)
        # Update how many cycles we've known each observation for
        self.cycles_known[self.idx_chosen] += 1

        # Refit the model
        self.fit()
        self.n_cycles += 1

    def compute_full_model(self):
        """
        Creates a model using all of the training data
        """
        self.full_model = clone(self.model)
        self.full_model.fit(self.X_train, self.y_train)


class NetworkActiveLearner(ActiveLearner):
    def __init__(self, X_train, X_test, y_train, y_test, model, learning_strategy, chosen_indices=None):
        """
        Create an ActiveLearner instance

        Parameters
        ----------
        X_train: np.ndarray(n_train, n_dim)
            Potential training data for the model
        X_test: np.ndarray(n_test, n_dim)
            The test set for the model
        y_train: np.ndarray(n_train, n_out)
            The output values for the training set
        y_test: np.ndarray(n_test, n_dim)
            The output values for the test set
        model:
            The model which is trying to be learnt
        learning_strategy: func(model, X_train, y_train, **kwargs)
            The learning strategy for the model. Should take a model, X values, y values as input, and
            return a tuple where:
            tuple[0] is a list of indices to be chosen --- Even a single index should be returned as a list
            tuple[1] is a list containing the name of the new nodes that have been chosen
            ------- NB ---------
            In general, the learning strategy for a network random learner requires the entire training set, not just
            the unchosen points.
        chosen_indices: array-like, default=None
            The initial chosen indices. If None, default indices will be chosen randomly, as 10%
            of the training data
        """
        # Use the init function inherited from the base active learner
        super(NetworkActiveLearner, self).__init__(X_train, X_test, y_train, y_test, model, learning_strategy,
                                                   chosen_indices)
        self.seen_nodes = []

    def findNewChosen(self, **kwargs):
        """
        Run the learning strategy, and add the new indices to chosen_indices
        """
        new_indices, new_nodes = self.learning_strategy(self.model, self.X_train,
                                                        self.y_train, **kwargs)
        new_idx_chosen = new_indices  # We don't need to translate this back from the

        # Add new chosen indices to idx_chosen
        self.idx_chosen[new_idx_chosen] = True

        # Also find new unchosen()
        self.unchosenFromChosen()

        # Add new node to seen_nodes
        self.seen_nodes += new_nodes

    ### ALL OTHER METHODS ARE UNCHANGED FROM INHERITANCE

########################################################################################################################
########################################################################################################################
#                                                                                                                      #
#                                               LEARNING STRATEGIES                                                    #
#                                                                                                                      #
########################################################################################################################
########################################################################################################################

###################
# ON OBSERVATIONS #
###################

def random_learning_strategy(model, X_chosen, y_chosen, n=1):
    # Pick n random points to learn
    return np.random.choice(len(X_chosen[:,0]), size=n, replace=False)


def high_error_learning_strategy(model, X_chosen, y_chosen, n=1):
    # Pick the n highest absolute errors to learn
    absolute_errors = np.abs(model.predict(X_chosen) - y_chosen.flatten())
    return absolute_errors.argsort()[-n:]


def uncertain_learning_strategy(model, X_chosen, y_chosen, n=1):
    # Pick the n most uncertain (i.e. highest stdev between tree predictions of random forest) to learn
    n_estimators = len(model.estimators_)
    predictions = np.zeros((len(X_chosen), n_estimators))
    for estimator_i in range(n_estimators):
        estimator = model.estimators_[estimator_i]
        predictions[:, estimator_i] = estimator.predict(X_chosen)
    stdevs = predictions.std(axis=1)
    return stdevs.argsort()[-n:]

####################
# ON NETWORK NODES #
####################

def random_node_learning_strategy(model, X_chosen, y_chosen, network, seen_nodes, n=1, frac=1):
    # Pick n random nodes to learn
    not_seen_nodes = [node for node in list(network['nodes'].keys()) if node not in seen_nodes]
    nodeIDs_chosen = list(np.random.choice(not_seen_nodes, size=n, replace=False))

    points = []
    for node in nodeIDs_chosen:
        node_points = network['nodes'][node]
        n_node_points = len(node_points)
        n_take = int(frac * n_node_points)
        if n_take == 0:
            n_take = 1
        points += list(np.random.choice(node_points, size=n_take, replace=False))
    return points, nodeIDs_chosen


def worst_node_learning_strategy(model, X_train, y_train, network, seen_nodes, n=1, frac=1):
    # Pick the n highest mean absolute error value nodes to learn

    # Predict the y-values for the entire training set
    y_pred = model.predict(X_train)
    absolute_errors = np.abs(y_pred - y_train)

    nodeIDs = list(network['nodes'].keys())
    not_seen_nodes = [node for node in nodeIDs if node not in seen_nodes]

    summary_df = pd.DataFrame(data=not_seen_nodes, columns=['node_ID'])
    summary_df = summary_df.set_index('node_ID')

    descriptor_names = ['MAE']
    for desc in descriptor_names:
        summary_df[desc] = np.inf  # Add these as columns to the summary df

    # Calculate mean value for the absolute error
    for node_ID in not_seen_nodes:
        mean_val = absolute_errors[network['nodes'][node_ID]].mean()

        summary_df.loc[node_ID] = mean_val

    # Find worst performing nodes
    sorted_clusters = summary_df.sort_values(by='MAE', ascending=False).index.values
    nodeIDs_chosen = list(sorted_clusters[-n:])

    points = []
    for node in nodeIDs_chosen:
        node_points = network['nodes'][node]
        n_node_points = len(node_points)
        n_take = int(frac * n_node_points)
        points += list(np.random.choice(node_points, size=n_take, replace=False))
    return points, nodeIDs_chosen


def largest_node_learning_strategy(model, X_train, y_train, network, seen_nodes, n=1):
    # Pick the n largest nodes to use
    not_seen_nodes = [node for node in list(network['nodes'].keys()) if node not in seen_nodes]
    n_nodes = len(not_seen_nodes)
    node_sizes = np.zeros(n_nodes)

    for node_i, node in enumerate(not_seen_nodes):
        node_sizes[node_i] = len(network['nodes'][node])
    nodeIDs_chosen = list(np.array(not_seen_nodes)[node_sizes.argsort()[-n:]])
    points = []
    for node in nodeIDs_chosen:
        points += network['nodes'][node]
    return points, nodeIDs_chosen


def uncertain_node_learning_strategy(model, X_train, y_train, network, seen_nodes, n=1, frac=1):
    # Pick the n most uncertain nodes

    # Calculate uncertainties
    n_estimators = len(model.estimators_)
    predictions = np.zeros((len(X_train), n_estimators))
    for estimator_i in range(n_estimators):
        estimator = model.estimators_[estimator_i]
        predictions[:, estimator_i] = estimator.predict(X_train)
    stdevs = predictions.std(axis=1)

    nodeIDs = list(network['nodes'].keys())
    not_seen_nodes = [node for node in nodeIDs if node not in seen_nodes]

    summary_df = pd.DataFrame(data=not_seen_nodes, columns=['node_ID'])
    summary_df = summary_df.set_index('node_ID')

    descriptor_names = ['STD']
    for desc in descriptor_names:
        summary_df[desc] = np.inf  # Add these as columns to the summary df

    # Calculate mean value for the absolute error
    for node_ID in not_seen_nodes:
        mean_val = stdevs[network['nodes'][node_ID]].mean()

        summary_df.loc[node_ID] = mean_val

    # Find worst performing nodes
    sorted_clusters = summary_df.sort_values(by='STD', ascending=False).index.values
    nodeIDs_chosen = list(sorted_clusters[-n:])

    points = []
    for node in nodeIDs_chosen:
        node_points = network['nodes'][node]
        n_node_points = len(node_points)
        n_take = int(frac * n_node_points)
        points += list(np.random.choice(node_points, size=n_take, replace=False))
    return points, nodeIDs_chosen
