#!/usr/bin/env python3
import joblib
import pickle
from matplotlib import rcParams
from matplotlib.cm import rainbow
from auto_pip_finder import PipFinder
from dbscan_pp import DBSCANPP
from combined_scoring_metrics import classification_scores


## Some Constants
##*********************##
RANDOM_SEED = 42
RANDOM_SAMPLE_SIZE = 13
NUM_DEC_PLACES = 4
GOLDEN_RATIO = 1.618033989
FIG_WIDTH = 20
FIG_HEIGHT = FIG_WIDTH / GOLDEN_RATIO
FIG_SIZE = (FIG_WIDTH, FIG_HEIGHT)
FIG_DPI = 72


## K Neighbours Classification Function
def k_nearest_neighbours(X_train, y_train, X_test, y_test, max_k=None, num_dp=4, messages=True):
    """
    Evaluate K-Nearest Neighbours classifier accuracy for different values of K.

    Plots accuracy scores for K values ranging from 1 to max_k and returns the scores in list and dictionary forms.

    Parameters
    ----------
    X_train : pandas.DataFrame or numpy.ndarray
        Training data features.
    y_train : pandas.Series or numpy.ndarray
        Training data labels.
    X_test : pandas.DataFrame or numpy.ndarray
        Test data features.
    y_test : pandas.Series or numpy.ndarray
        Test data labels.
    max_k : int
        Maximum number of neighbours to test.
    num_dp : int, optional (default=4)
        Number of decimal places for displaying accuracy scores.

    Returns
    -------
    tuple
        A tuple containing:
        - knn_acc_scores_list : list
            List of accuracy scores for each K value.
        - knn_acc_scores_dict : dict
            Dictionary with K values as keys and corresponding accuracy scores as values.
    Examples
    --------
    k_nearest_neighbours(X_train, y_train, X_test, y_test, 23)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    try:
        import fireducks.pandas as pd
    except ImportError:
        import pandas as pd
    from sklearn.neighbors import KNeighborsClassifier as KNeighboursClassifier

    knn_acc_scores_list = []
    knn_acc_scores_dict = {}
    if not max_k:
        max_k = 7

    for k in range(1, max_k + 1):
        knn_clf = KNeighboursClassifier(n_neighbors=k)
        knn_clf.fit(X_train, y_train)
        y_pred = knn_clf.predict(X_test)
        knn_acc_score = knn_clf.score(X_test, y_test)
        knn_acc_scores_list.append(knn_acc_score)
        knn_acc_scores_dict[f"{k}_neighbours"] = classification_scores(y_test, y_pred, messages=False)

    if messages:
        # Plot the scores on a line plot
        plt.acc_plot(
            [k for k in range(1, max_k + 1)],
            knn_scores_list,
            color="grey",
            marker="o",
            linewidth=0.73,
            markerfacecolor="red",
        )
        for i in range(1, max_k + 1):
            plt.acc_text(
                i,
                knn_scores_list[i - 1],
                f"   ({i}, {round(knn_scores_list[i - 1], num_dp)})",
                rotation=90,
                va="bottom",
                fontsize=8,
            )
        plt.xticks([i for i in range(1, max_k + 2)])
        y_bottom, y_top = plt.ylim()
        plt.ylim(top=y_top * 1.05)
        plt.xlabel("Number of Neighbours (K)")
        plt.ylabel("Accuracy Scores")
        plt.title("K Neighbours Classifier Accuracy Scores for Different K Values")
    return knn_acc_scores_list, knn_acc_scores_dict


## Support Vector Classification Function
def support_vector_classification(
    X_train,
    y_train,
    X_test,
    y_test,
    kernels=["linear", "poly", "rbf", "sigmoid"],
    num_dp=4,
    messages=True
):
    """
    Evaluate Support Vector Classifier accuracy for different kernels.

    Plots accuracy scores for each kernel in list `kernels` and returns the scores in list and dictionary forms.

    Parameters
    ----------
    X_train : pandas.DataFrame or numpy.ndarray
        Training data features.
    y_train : pandas.Series or numpy.ndarray
        Training data labels.
    X_test : pandas.DataFrame or numpy.ndarray
        Test data features.
    y_test : pandas.Series or numpy.ndarray
        Test data labels.
    kernels : list of str, optional (default=["linear", "poly", "rbf", "sigmoid"])
        List of kernels to test.
    num_dp : int, optional (default=4)
        Number of decimal places for displaying accuracy scores.

    Returns
    -------
    tuple
        A tuple containing:
        - svc_acc_scores_list : list
            List of accuracy scores for each kernel in list `kernels`.
        - svc_acc_scores_dict : dict
            Dictionary with kernel names as keys and corresponding accuracy scores as values.
    Examples
    --------
    support_vector_classification(X_train, y_train, X_test, y_test)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    try:
        import fireducks.pandas as pd
    except ImportError:
        import pandas as pd
    from sklearn.svm import SVC

    svc_acc_scores_list = []
    svc_acc_scores_dict = {}
    max_k = len(kernels)

    for k in range(max_k):
        svc_clf = SVC(kernel=kernels[k])
        svc_clf.fit(X_train, y_train)
        y_pred = svc_clf.predict(X_test)
        svc_acc_score = svc_clf.score(X_test, y_test)
        svc_acc_scores_list.append(svc_acc_score)
        svc_acc_scores_dict[kernels[k]] = classification_scores(y_test, y_pred, messages=False)

    if messages:
        # Plot the scores on a barplot
        colours = rainbow(np.linspace(0, 1, max_k))
        plt.bar(kernels, svc_acc_scores_list, color=colours)
        for i in range(max_k):
            plt.text(
                i,
                svc_acc_scores_list[i],
                f"{round(svc_acc_scores_list[i], num_dp)}",
                rotation=45,
                va="bottom",
                fontsize=10,
            )
        y_bottom, y_top = plt.ylim()
        plt.ylim(top=y_top * 1.13)
        plt.xlabel("Kernels")
        plt.ylabel("Accuracy Scores")
        plt.title("Support Vector Classifier Accuracy Scores for Different Kernels")
    return svc_acc_scores_list, svc_acc_scores_dict


## Decision Tree Classification Function
def decision_tree_classification(X_train, y_train, X_test, y_test, num_dp=4, messages=True):
    """
    Plots accuracy scores for DecisionTreeClassifier when using different number of max features.

    Parameters
    ----------
    X_train : pandas.DataFrame
        The training data
    y_train : pandas.Series or numpy.ndarray
        The target values of the training data
    X_test : pandas.DataFrame
        The test data
    y_test : pandas.Series or numpy.ndarray
        The target values of the test data
    num_dp : int, optional
        The number of decimal places to round the accuracy scores to, by default 4

    Returns
    -------
    tuple
        A tuple containing a list of accuracy scores and a dictionary of accuracy scores where the keys are the number of max features used.
    Examples
    --------
    decision_tree_classification(X_train, y_train, X_test, y_test)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    try:
        import fireducks.pandas as pd
    except ImportError:
        import pandas as pd
    from sklearn.tree import DecisionTreeClassifier as DecisionTree

    dt_acc_scores_list = []
    dt_acc_scores_dict = {}
    max_k = len(X_train.columns)

    for k in range(1, max_k + 1):
        dt_clf = DecisionTree(
            max_features=k, random_state=RANDOM_SEED or 42
        )
        dt_clf.fit(X_train, y_train)
        y_pred = dt_clf.predict(X_test)
        dt_acc_score = dt_clf.score(X_test, y_test)
        dt_acc_scores_list.append(dt_acc_score)
        dt_acc_scores_dict[f"{k}_features"] = classification_scores(y_test, y_pred, messages=False)

    if messages:
        # Plot the scores on a line plot
        plt.plot(
            [k for k in range(1, max_k + 1)],
            dt_acc_scores_list,
            color="green",
            marker="o",
            linewidth=0.73,
            markerfacecolor="red",
        )
        for i in range(1, max_k + 1):
            plt.text(
                i,
                dt_acc_scores_list[i - 1],
                f"   ({i}, {round(dt_acc_scores_list[i - 1], num_dp)})",
                rotation=90,
                va="bottom",
                fontsize=8,
            )
        plt.xticks([i for i in range(1, max_k + 2)])
        y_bottom, y_top = plt.ylim()
        plt.ylim(top=y_top * 1.05)
        plt.xlabel("Max Features")
        plt.ylabel("Accuracy Scores")
        plt.title(
            "Decision Tree Classifier Accuracy Scores for Different Number of Max Features"
        )
    return dt_acc_scores_list, dt_acc_scores_dict


## Random Forest Classification Function
def random_forest_classification(
    X_train,
    y_train,
    X_test,
    y_test,
    estimators=[2, 3, 5, 7, 10, 13, 89, 100, 200, 233, 500, 1000, 1597],
    num_dp=4,
    messages=True
):
    """
    Evaluate Random Forest Classifier accuracy for different numbers of estimators.

    Plots accuracy scores for each number of estimators in list `estimators` and returns the scores in list and dictionary forms.

    Parameters
    ----------
    X_train : pandas.DataFrame or numpy.ndarray
        Training data features.
    y_train : pandas.Series or numpy.ndarray
        Training data labels.
    X_test : pandas.DataFrame or numpy.ndarray
        Test data features.
    y_test : pandas.Series or numpy.ndarray
        Test data labels.
    estimators : list of int, optional (default=[2, 3, 5, 7, 10, 13, 89, 100, 200, 233, 500, 1000, 1597])
        List of numbers of estimators to test.
    num_dp : int, optional (default=4)
        Number of decimal places for displaying accuracy scores.

    Returns
    -------
    tuple
        A tuple containing:
        - rf_acc_scores_list : list
            List of accuracy scores for each number of estimators in list `estimators`.
        - rf_acc_scores_dict : dict
            Dictionary with number of estimators as keys and corresponding accuracy scores as values.
    Examples
    --------
    random_forest_classification(X_train, y_train, X_test, y_test)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    try:
        import fireducks.pandas as pd
    except ImportError:
        import pandas as pd
    from matplotlib.cm import rainbow
    from sklearn.ensemble import RandomForestClassifier as RandomForest

    rf_acc_scores_list = []
    rf_acc_scores_dict = {}
    max_k = len(estimators)

    for k in estimators:
        rf_clf = RandomForest(
            n_estimators=k, random_state=RANDOM_SEED or 42
        )
        rf_clf.fit(X_train, y_train)
        y_pred = rf_clf.predict(X_test)
        rf_acc_score = rf_clf.score(X_test, y_test)
        rf_acc_scores_list.append(rf_acc_score)
        rf_acc_scores_dict[f"{k}_estimators"] = classification_scores(y_test, y_pred, messages=False)

    if messages:
        # Plot the scores on a barplot
        colours = rainbow(np.linspace(0, 1, max_k))
        plt.bar([i for i in range(max_k)], rf_acc_scores_list, color=colours, width=0.37)
        for i in range(max_k):
            plt.text(
                i,
                rf_acc_scores_list[i],
                f"{round(rf_acc_scores_list[i], num_dp)}",
                rotation=60,
                va="bottom",
                fontsize=10,
            )
        plt.xticks(
            ticks=[i for i in range(max_k)],
            labels=[str(estimator) for estimator in estimators],
            rotation=45,
        )
        y_bottom, y_top = plt.ylim()
        plt.ylim(top=y_top * 1.13)
        plt.xlabel("Number of Estimators")
        plt.ylabel("Accuracy Scores")
        plt.title(
            "Random Forest Classifier Accuracy Scores for Different Number of Estimators"
        )
    return rf_acc_scores_list, rf_acc_scores_dict


## XGBoost Classification Function
def xgboost_classification(
    X_train,
    y_train,
    X_test,
    y_test,
    X_val=None,
    y_val=None,
    max_num_estimators=103,
    num_dp=4,
    messages=True
):
    """
    Evaluate XGBoost Classifier accuracy for different numbers of estimators.

    Plots accuracy scores for each number of estimators in range 1 to `max_num_estimators` and returns the scores in list and dictionary forms.

    Parameters
    ----------
    X_train : pandas.DataFrame or numpy.ndarray
        Training data features.
    y_train : pandas.Series or numpy.ndarray
        Training data labels.
    X_test : pandas.DataFrame or numpy.ndarray
        Test data features.
    y_test : pandas.Series or numpy.ndarray
        Test data labels.
    X_val : pandas.DataFrame or numpy.ndarray, optional (default=None)
        Validation data features.
    y_val : pandas.Series or numpy.ndarray, optional (default=None)
        Validation data labels.
    max_num_estimators : int, optional (default=103)
        Maximum number of estimators to test.
    num_dp : int, optional (default=4)
        Number of decimal places for displaying accuracy scores.

    Returns
    -------
    tuple
        A tuple containing:
        - xgb_acc_scores_list : list
            List of accuracy scores for each number of estimators in range 1 to `max_num_estimators`.
        - xgb_acc_scores_dict : dict
            Dictionary with number of estimators as keys and corresponding accuracy scores as values.
    Examples
    --------
    xgboost_classification(X_train, y_train, X_test, y_test)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    try:
        import fireducks.pandas as pd
    except ImportError:
        import pandas as pd
    from xgboost import XGBClassifier

    xgb_acc_scores_list = []
    xgb_acc_scores_dict = {}
    max_depth = len(X_train.columns)

    for k in range(1, max_num_estimators + 1):
        xgb_clf = XGBClassifier(
            n_estimators=k,
            objective="binary:logistic",
            tree_method="hist",
            eta=0.1,
            max_depth=max_depth,
            enable_categorical=True,
            seed=RANDOM_SEED or 42,
        )
        xgb_clf.fit(X_train, y_train, verbose=False)
        y_pred = xgb_clf.predict(X_test)
        xgb_acc_score = xgb_clf.score(X_test, y_test)
        xgb_acc_scores_list.append(xgb_acc_score)
        xgb_acc_scores_dict[f"{k}_estimators"] = classification_scores(y_test, y_pred, messages=False)

    if messages:
        # Plot the scores on a line plot
        plt.plot(
            [k for k in range(1, max_num_estimators + 1)],
            xgb_acc_scores_list,
            color="green",
            linewidth=0.73,
        )
        for i in range(0, max_num_estimators + 1, 10):
            plt.text(
                i,
                xgb_acc_scores_list[i - 1],
                f"   ({i}, {round(xgb_acc_scores_list[i - 1], num_dp)})",
                rotation=90,
                va="bottom",
                fontsize=8,
            )
        plt.xticks([i for i in range(0, max_num_estimators + 2, 5)])
        y_bottom, y_top = plt.ylim()
        plt.ylim(top=y_top * 1.05)
        plt.xlabel("Number of Estimators")
        plt.ylabel("Accuracy Scores")
        plt.title("XGBoost Classifier Accuracy Scores for Different Number of Estimators")
    return xgb_acc_scores_list, xgb_acc_scores_dict


## Naive Bayes Classification Function
def naive_bayes_classification(X_train, y_train, X_test, y_test, num_dp=4, messages=True):
    """
    Plots accuracy scores for different Naive Bayes classifiers.

    Parameters
    ----------
    X_train : pandas.DataFrame or numpy.ndarray
        Training data features.
    y_train : pandas.Series or numpy.ndarray
        Training data labels.
    X_test : pandas.DataFrame or numpy.ndarray
        Test data features.
    y_test : pandas.Series or numpy.ndarray
        Test data labels.
    num_dp : int, optional (default=4)
        Number of decimal places for displaying accuracy scores.

    Returns
    -------
    tuple
        A tuple containing:
        - nb_acc_scores_list : list
            List of accuracy scores for each classifier.
        - nb_acc_scores_dict : dict
            Dictionary with classifier names as keys and corresponding accuracy scores as values.
    Examples
    --------
    naive_bayes_classification(X_train, y_train, X_test, y_test)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    try:
        import fireducks.pandas as pd
    except ImportError:
        import pandas as pd
    from matplotlib.cm import rainbow
    from sklearn.naive_bayes import (
        BernoulliNB,
        CategoricalNB,
        ComplementNB,
        GaussianNB,
        MultinomialNB,
    )

    nb_classifiers = {
        "BernoulliNB": BernoulliNB(),
        "CategoricalNB": CategoricalNB(),
        "ComplementNB": ComplementNB(),
        "GaussianNB": GaussianNB(),
        "MultinomialNB": MultinomialNB(),
    }

    nb_acc_scores_list = []
    nb_acc_scores_dict = {}

    for name, nb_clf in nb_classifiers.items():
        try:
            nb_clf.fit(X_train, y_train)
            y_pred = nb_clf.predict(X_test)
            nb_acc_score = nb_clf.score(X_test, y_test)
            nb_acc_scores_list.append(nb_acc_score)
            nb_acc_scores_dict[name] = classification_scores(y_test, y_pred, messages=False)
        except ValueError:
            continue

    if messages:
        # Plot the scores on a barplot
        colours = rainbow(np.linspace(0, 1, len(nb_classifiers)))
        plt.bar(nb_classifiers.keys(), nb_acc_scores_list, color=colours)
        for i in range(len(nb_classifiers)):
            plt.text(
                i,
                nb_acc_scores_list[i],
                f"{round(nb_acc_scores_list[i], num_dp)}",
                rotation=45,
                va="bottom",
                fontsize=10,
            )
        plt.xticks(
            ticks=[i for i in range(len(nb_classifiers))],
            labels=nb_classifiers.keys(),
            rotation=45,
        )
        y_bottom, y_top = plt.ylim()
        plt.ylim(top=y_top * 1.13)
        plt.xlabel("Classifiers")
        plt.ylabel("Accuracy Scores")
        plt.title("Naive Bayes Classifier Accuracy Scores for Different Classifiers")
    return nb_acc_scores_list, nb_acc_scores_dict


