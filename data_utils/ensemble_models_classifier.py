#!/usr/bin/env python3
import numpy as np
try:
    import fireducks.pandas as pd
except ImportError:
    import pandas as pd
from os.path import realpath as realpath
from sklearn.ensemble import RandomForestClassifier as RandomForest
from sklearn.ensemble import VotingClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

RANDOM_SEED = 42

# Helper Functions
def generator_nested_dict(dict_obj):
    """
    Generator to iterate over nested dictionaries.

    Args:
        dict_obj (dict): A dictionary which may contain nested dictionaries.

    Yields:
        tuple: A tuple containing a key and a value. The value may be a nested dictionary.

    Notes:
        This generator will recursively iterate over nested dictionaries.
    """
    for key, value in dict_obj.items():
        yield from (
            generator_nested_dict(value) if isinstance(value, dict) else ((key, value),)
        )


def get_data(file_path):
    real_path_to_file_path = realpath(file_path)
    df = pd.read_csv(real_path_to_file_path)
    return df


def preprocess_dataframe(df, file_path=None, target="target", use_robust_scaler=True):
    """
    Preprocess a given DataFrame by dropping duplicates, scaling/transforming the
    numerical columns using RobustScaler() or PowerTransformer() (if specified),
    and then adding the target column back to the DataFrame at the end.

    Parameters
    ----------
    df : pandas DataFrame
        The DataFrame to preprocess. If empty, the file at `file_path` is read in.
    file_path : str, optional
        The path to the file to read in if `df` is empty. Default None.
    target : str, optional
        The column name of the target variable. Default "target".
    use_robust_scaler : bool, optional
        If True, RobustScaler() is used. Otherwise, PowerTransformer() is used.
        Default True.

    Returns
    -------
    df : pandas DataFrame
        The preprocessed DataFrame.
    """
    try:
        import fireducks.pandas as pd
    except ImportError:
        import pandas as pd
    from sklearn.preprocessing import PowerTransformer
    from sklearn.preprocessing import RobustScaler

    if df.empty:
        df = get_data(file_path)

    df = df.drop_duplicates()
    df_target = df[target]
    df = df.drop(target, axis=1)
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if use_robust_scaler:
                df[col] = RobustScaler().fit_transform(df[col].values.reshape(-1, 1))
            else:
                df[col] = PowerTransformer().fit_transform(df[col].values.reshape(-1, 1))

    df["target"] = df_target
    return df


def create_train_test(
    df, file_path=None, target="target", test_prop=0.23, use_robust_scaler=True
):
    """
    Split a given DataFrame into features and target, preprocess the features, and split them into
    training and test sets.

    Parameters
    ----------
    df : pandas DataFrame
        The DataFrame to split. If empty, the file at `file_path` is read in.
    file_path : str, optional
        The path to the file to read in if `df` is empty. Default None.
    target : str, optional
        The column name of the target variable. Default "target".
    test_prop : float, optional
        The proportion of the input dataframe to use for the test set. Default 0.23.
    use_robust_scaler : bool, optional
        If True, RobustScaler() is used. Otherwise, PowerTransformer() is used. Default True.

    Returns
    -------
    X_train : pandas DataFrame
        The subset of the input dataframe to use for training.
    X_test : pandas DataFrame
        The subset of the input dataframe to use for testing.
    y_train : pandas Series or numpy.ndarray
        The target values of the training set.
    y_test : pandas Series or numpy.ndarray
        The target values of the test set.
    """
    df = preprocess_dataframe(df, file_path, target, use_robust_scaler)
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_prop, random_state= RANDOM_SEED or 42
    )
    return X_train, X_test, y_train, y_test


## Random Forest Classification Function
def random_forest_classification_halving_search(
    X_train,
    y_train,
    X_test,
    y_test,
    estimators=[2, 3, 5, 7, 10, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 100, 151, 200, 233, 347, 500, 757, 1000, 1231, 1597, 1777, 2003, 2531, 3001, 3583,],
    num_cv = 3,
    num_dp=4,
    messages=True,
):
    import matplotlib.pyplot as plt
    import numpy as np
    try:
        import fireducks.pandas as pd
    except ImportError:
        import pandas as pd
    from matplotlib.cm import rainbow
    from sklearn.ensemble import RandomForestClassifier as RandomForest
    from sklearn.experimental import enable_halving_search_cv
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        ConfusionMatrixDisplay,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
        precision_recall_fscore_support,
    )
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import HalvingGridSearchCV

    RANDOM_SEED = 42
    rf_scores_dict = {}
    max_k = len(estimators)

    for k in estimators:
        rf_clf = RandomForest(
            n_estimators=k, random_state=RANDOM_SEED or 42, class_weight="balanced"
        )
        rf_clf.fit(X_train, y_train)
        rf_pred = rf_clf.predict(X_test)
        rf_acc = accuracy_score(y_test, rf_pred)
        rf_f1 = f1_score(y_test, rf_pred, average="weighted")
        rf_mcc = matthews_corrcoef(y_test, rf_pred)

        rf_scores_dict[f"{k}_estimators"] = {}
        rf_scores_dict[f"{k}_estimators"]["matthews_corrcoef"] = rf_mcc
        rf_scores_dict[f"{k}_estimators"]["f1_score"] = rf_f1
        rf_scores_dict[f"{k}_estimators"]["acc_score"] = rf_acc

    # Parameters for Halving-Grid Search
    rf_param_grid = {
        "n_estimators": [2, 3, 5, 7, 10, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 100, 151, 200, 233, 347, 500, 757, 1000, 1231, 1597, 1777, 2003, 2531, 3001, 3583,],
        "max_depth": [None, 1, 2, 3, 5, 7, 10, 11, 13, 17, 19, 23],
        "min_samples_leaf": [1, 2, 3, 4, 5, 7, 10, 11, 13, 17, 19, 23],
    }

    base_rf_clf = RandomForest(random_state=RANDOM_SEED or 42, class_weight="balanced")

    halving_rf_clf = HalvingGridSearchCV(base_rf_clf, rf_param_grid, factor=3, aggressive_elimination=True, cv=num_cv, scoring="f1_macro", random_state=RANDOM_SEED or 42, verbose=3)
    halving_rf_clf.fit(X_train, y_train)

    rf_scores_dict["halving"] = {}
    rf_scores_dict["halving"]["cv_scores"] = cross_val_score(halving_rf_clf, X_test, y_test)
    rf_scores_dict["halving"]["rf_best_params"] = halving_rf_clf.best_params_

    if messages:
        print("Best Halving Grid Search Parameters:", halving_rf_clf.best_params_)
        titles_options = [
            ("RFC Confusion Matrix Without Normalisation", None),
            ("Normalised RFC Confusion Matrix", "true"),
        ]
        for title, normalise in titles_options:
            disp = ConfusionMatrixDisplay.from_estimator(
                rf_clf,
                X_test,
                y_test,
                cmap=plt.cm.Blues,
                normalize=normalise,
            )
            print(title)
            print(disp.confusion_matrix)
        plt.show()
        print("All RFC Scores:")
        for key01, val01 in rf_scores_dict.items():
            if not isinstance(val01, dict):
                if isinstance(val01, (int, float)):
                    print(f"{key01}: {np.round(val01, num_dp)}")
                else:
                    print(f"{key01}: {val01}")
                continue
            else:
                print(f"{key01}")
                for key02, val02 in val01.items():
                    print(f"{key02}: {np.round(val02, num_dp)}")
                print("*********************")
            print()

    return rf_scores_dict, halving_rf_clf


## Support Vector Classification Function
def support_vector_classification_halving_search(
    X_train,
    y_train,
    X_test,
    y_test,
    kernels=["linear", "poly", "rbf", "sigmoid",],
    num_dp=4,
    messages=True,
):
    import matplotlib.pyplot as plt
    import numpy as np

    try:
        import fireducks.pandas as pd
    except ImportError:
        import pandas as pd
    from sklearn.experimental import enable_halving_search_cv
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        ConfusionMatrixDisplay,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
        precision_recall_fscore_support,
    )
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import HalvingGridSearchCV
    from sklearn.svm import SVC

    svc_scores_dict = {}
    max_k = len(kernels)

    for k in range(max_k):
        svc_clf = SVC(kernel=kernels[k], class_weight="balanced")
        svc_clf.fit(X_train, y_train)
        svc_pred = svc_clf.predict(X_test)
        svc_acc = accuracy_score(y_test, svc_pred)
        svc_f1 = f1_score(y_test, svc_pred, average="weighted")
        svc_mcc = matthews_corrcoef(y_test, svc_pred)

        svc_scores_dict[f"{kernels[k]}_kernel"] = {}
        svc_scores_dict[f"{kernels[k]}_kernel"]["matthews_corrcoef"] = svc_mcc
        svc_scores_dict[f"{kernels[k]}_kernel"]["f1_score"] = svc_f1
        svc_scores_dict[f"{kernels[k]}_kernel"]["acc_score"] = svc_acc

   # Parameters for Halving-Grid Search
    svm_param_grid = {
        "C": [0.1, 0.3, 0.5, 0.7, 1, 2, 3, 5, 7, 10, 11, 13, 17, 19, 23],
        "gamma": [0.1, 0.05, 0.01, 0.001],
        "kernel": ["linear", "poly", "rbf", "sigmoid",],
    }

    base_svm_clf = SVC(class_weight="balanced")

    svc_scores_dict["halving"] = {}

    halving_svm_clf = HalvingGridSearchCV(base_svm_clf, svm_param_grid, factor=3, aggressive_elimination=True, cv=3, scoring="f1_macro", random_state=RANDOM_SEED or 42, verbose=3)
    halving_svm_clf.fit(X_train, y_train)

    svc_scores_dict["halving"] = {}
    svc_scores_dict["halving"]["cv_scores"] = cross_val_score(halving_svm_clf, X_test, y_test)
    svc_scores_dict["halving"]["rf_best_params"] = halving_svm_clf.best_params_

    if messages:
        print("Best Halving Grid Search Parameters:", halving_svm_clf.best_params_)
        titles_options = [
            ("SVC Confusion Matrix Without Normalisation", None),
            ("Normalised SVC Confusion Matrix", "true"),
        ]
        for title, normalise in titles_options:
            disp = ConfusionMatrixDisplay.from_estimator(
                svc_clf,
                X_test,
                y_test,
                cmap=plt.cm.Blues,
                normalize=normalise,
            )
            print(title)
            print(disp.confusion_matrix)
        plt.show()
        print("All SVC Scores:")
        for key01, val01 in svc_scores_dict.items():
            if not isinstance(val01, dict):
                if isinstance(val01, (int, float)):
                    print(f"{key01}: {np.round(val01, num_dp)}")
                else:
                    print(f"{key01}: {val01}")
                continue
            else:
                print(f"{key01}")
                for key02, val02 in val01.items():
                    print(f"{key02}: {np.round(val02, num_dp)}")
                print("*********************")
            print()

    return svc_scores_dict, halving_svm_clf


def ensemble_classifier(
    X_train,
    y_train,
    X_test,
    y_test,
    num_dp=4,
    messages=True,
):
    """
    Evaluate ensemble classifier accuracy using base models, halving-grid tuned models, and optimized ensemble models.

    This function creates an ensemble classifier using a VotingClassifier with Support Vector Machine (SVM) and Random Forest
    classifiers as base models. It evaluates the performance of the base ensemble, halving-grid tuned ensemble, and optimized
    ensemble models by calculating accuracy, F1 score, and Matthews correlation coefficient. Optionally, it displays confusion
    matrices for each model.

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
    messages : bool, optional (default=False)
        Whether or not to print detailed accuracy scores and display confusion matrices.

    Returns
    -------
    dict
        Dictionary containing accuracy scores, F1 scores, and Matthews correlation coefficients for the base, halving-grid
        tuned, and optimized ensemble models.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    try:
        import fireducks.pandas as pd
    except ImportError:
        import pandas as pd
    from sklearn.ensemble import RandomForestClassifier as RandomForest
    from sklearn.ensemble import VotingClassifier
    from sklearn.experimental import enable_halving_search_cv
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        ConfusionMatrixDisplay,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
        precision_recall_fscore_support,
    )
    from sklearn.model_selection import HalvingGridSearchCV
    from sklearn.svm import SVC

    RANDOM_SEED = 42

    ensemble_scores_dict = {}

    # Using Base Models

    # Instantiate base models
    base_svm_clf = SVC(class_weight="balanced")
    base_rf_clf = RandomForest(random_state=RANDOM_SEED or 42, class_weight="balanced")

    # Create base ensemble model using VotingClassifier
    if messages:
        print("Creating Base Ensemble Model...")
    base_ensemble_classifier = VotingClassifier(
        estimators=[("svm", base_svm_clf), ("random_forest", base_rf_clf)],
        voting="soft",
    )
    # Train the base ensemble model
    base_ensemble_classifier.fit(X_train, y_train)
    # Predictions from the base ensemble model
    base_ensemble_pred = base_ensemble_classifier.predict(X_test)
    base_ensemble_acc = accuracy_score(y_test, base_ensemble_pred)
    base_ensemble_f1 = f1_score(y_test, base_ensemble_pred, average="weighted")
    base_ensemble_mcc = matthews_corrcoef(y_test, base_ensemble_pred)
    ensemble_scores_dict["base"] = {}
    ensemble_scores_dict["base"]["matthews_corrcoef"] = base_ensemble_mcc
    ensemble_scores_dict["base"]["f1_score"] = base_ensemble_f1
    ensemble_scores_dict["base"]["acc_score"] = base_ensemble_acc
    if messages:
        print(classification_report(y_test, base_ensemble_pred, digits=num_dp))
        titles_options01 = [
            ("Base Confusion Matrix Without Normalisation", None),
            ("Normalised Base Confusion Matrix", "true"),
        ]
        for title, normalise in titles_options01:
            disp01 = ConfusionMatrixDisplay.from_estimator(
                base_ensemble_classifier,
                X_test,
                y_test,
                cmap=plt.cm.Blues,
                normalize=normalise,
            )
            print(title)
            print(disp01.confusion_matrix)
        plt.show()

    # Using Halving-Grid Tuned Models
    if messages:
        print("Creating Halving-Grid Tuned Ensemble Model...")



    # Initialise record object for initial halving grid search outside ensemble
    _, halving_svm_clf = support_vector_classification_halving_search(X_train, y_train, X_test, y_test)
    _, halving_rf_clf = random_forest_classification_halving_search(X_train, y_train, X_test, y_test)

    # Create halving-grid-search ensemble model using VotingClassifier
    halving_ensemble_classifier = VotingClassifier(
        estimators=[("halving_svm", halving_svm_clf), ("halving_random_forest", halving_rf_clf)],
        voting="soft",
    )

    # Train and assess prediction from the halving grid search ensemble model
    halving_ensemble_classifier.fit(X_train, y_train)
    halving_ensemble_pred = halving_ensemble_classifier.predict(X_test)
    halving_ensemble_acc = accuracy_score(y_test, halving_ensemble_pred)
    halving_ensemble_f1 = f1_score(y_test, halving_ensemble_pred, average="weighted")
    halving_ensemble_mcc = matthews_corrcoef(y_test, halving_ensemble_pred)

    ensemble_scores_dict["halving"]["matthews_corrcoef"] = halving_ensemble_mcc
    ensemble_scores_dict["halving"]["f1_score"] = halving_ensemble_f1
    ensemble_scores_dict["halving"]["acc_score"] = halving_ensemble_acc
    if messages:
        print(classification_report(y_test, halving_ensemble_pred, digits=num_dp))
        titles_options02 = [
            ("Halving Confusion Matrix Without Normalisation", None),
            ("Normalised Halving Confusion Matrix", "true"),
        ]
        for title, normalise in titles_options02:
            disp02 = ConfusionMatrixDisplay.from_estimator(
                halving_ensemble_classifier,
                X_test,
                y_test,
                cmap=plt.cm.Blues,
                normalize=normalise,
            )
            print(title)
            print(disp02.confusion_matrix)
        plt.show()

    # Using Optimised Ensemble Models
    if messages:
        print("Creating Optimised Ensemble Model...")
    ensemble_param_grid = {
        "svm__C": [0.1, 0.3, 0.5, 0.7, 1, 2, 3, 5, 7, 10, 11, 13, 17, 19, 23],
        "svm__gamma": [0.1, 0.05, 0.01, 0.001],
        "svm__kernel": ["linear", "poly", "rbf", "sigmoid",],
        "random_forest__n_estimators": [2, 3, 5, 7, 10, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 100, 151, 200, 233, 347, 500, 757, 1000, 1231, 1597, 1777, 2003, 2531, 3001, 3583,],
        "random_forest__max_depth": [None, 1, 2, 3, 5, 7, 10, 11, 13, 17, 19, 23],
        "random_forest__min_samples_leaf": [1, 2, 3, 4, 5, 7, 10, 11, 13, 17, 19, 23],
    }

    ensemble_clf_halving_grid_search = HalvingGridSearchCV(
        halving_ensemble_classifier, ensemble_param_grid, factor=3, aggressive_elimination=True, cv=3, scoring="f1_macro", random_state=RANDOM_SEED or 42, verbose=3
    )

    # Train and assess prediction from the fine-tuned halving grid search ensemble model
    ensemble_clf_halving_grid_search.fit(X_train, y_train)
    ensemble_clf_halving_grid_search_pred = ensemble_clf_halving_grid_search.predict(
        X_test
    )
    ensemble_clf_halving_grid_search_acc = accuracy_score(
        y_test, ensemble_clf_halving_grid_search_pred
    )
    ensemble_clf_halving_grid_search_f1 = f1_score(
        y_test, ensemble_clf_halving_grid_search_pred, average="weighted"
    )
    ensemble_clf_halving_grid_search_mcc = matthews_corrcoef(
        y_test, ensemble_clf_halving_grid_search_pred
    )
    ensemble_scores_dict["tuned"] = {}
    ensemble_scores_dict["tuned"]["matthews_corrcoef"] = ensemble_clf_halving_grid_search_mcc
    ensemble_scores_dict["tuned"]["f1_score"] = ensemble_clf_halving_grid_search_f1
    ensemble_scores_dict["tuned"]["acc_score"] = ensemble_clf_halving_grid_search_acc
    if messages:
        print(classification_report(y_test, ensemble_clf_halving_grid_search_pred, digits=num_dp))
        print("All Ensemble Scores:")
        for key01, val01 in ensemble_scores_dict.items():
            if not isinstance(val01, dict):
                if isinstance(val01, (int, float)):
                    print(f"{key01}: {np.round(val01, num_dp)}")
                else:
                    print(f"{key01}: {val01}")
                continue
            else:
                print(f"{key01}")
                for key02, val02 in val01.items():
                    print(f"{key02}: {np.round(val02, num_dp)}")
                print("*********************")
            print()
        titles_options03 = [
            ("Tuned Confusion Matrix Without Normalisation", None),
            ("Normalised Tuned Confusion Matrix", "true"),
        ]
        for title, normalise in titles_options03:
            disp03 = ConfusionMatrixDisplay.from_estimator(
                ensemble_clf_halving_grid_search,
                X_test,
                y_test,
                cmap=plt.cm.Blues,
                normalize=normalise,
            )
            print(title)
            print(disp03.confusion_matrix)
        plt.show()

    # Return scores
    return ensemble_scores_dict, ensemble_clf_halving_grid_search


def main(path_to_data="../../Data_Science_Analytics/000_common_dataset/arrhythmia.csv"):
    df = get_data(path_to_data)
    df = preprocess_dataframe(df, target="y")
    X_train, X_test, y_train, y_test = create_train_test(df)

    ensemble_scores_dict, _ = ensemble_classifier(X_train, y_train, X_test, y_test)

    # Print results
    print("\nEnsemble Scores")
    for k3, v3 in generator_nested_dict(ensemble_scores_dict):
        print(f"{k3}: {v3}")


if __name__ == "__main__":
    main()
