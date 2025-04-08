#!/usr/bin/env python3
try:
    import fireducks.pandas as pd
except ImportError:
    import pandas as pd
from os.path import realpath as realpath
from sklearn.ensemble import RandomForestClassifier as RandomForest
from sklearn.ensemble import VotingClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, matthews_corrcoef, precision_recall_fscore_support
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC


def get_data(file_path):
    real_path_to_file_path = realpath(file_path)
    df = pd.read_csv(real_path_to_file_path)
    return df


def preprocess_dataframe(df, file_path=None, target="target", use_robust_scaler=True):
    from sklearn.preprocessing import PowerTransformer
    from sklearn.preprocessing import RobustScaler
    if df.empty():
        df = get_data(file_path)

    df = df.drop_duplicates()
    df_target = df[target]
    df = df.drop(target, axis=1)
    for col in df.columns:
        if use_robust_scaler:
            df[col] = RobustScaler().fit_transform(df[col].values.reshape(-1, 1))
        else:
            df[col] = PowerTransformer().fit_transform(df[col].values.reshape(-1, 1))

    df["target"] = df_target
    return df


def create_train_test(df, file_path=None, target="target", test_prop=0.23, use_robust_scaler=True):
    df = preprocess_dataframe(df, file_path, target, use_robust_scaler)
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_prop, random_state=42)
    return X_train, X_test, y_train, y_test


## Random Forest Classification Function
def random_forest_classification(
    X_train,
    y_train,
    X_test,
    y_test,
    estimators=[2, 3, 5, 7, 10, 13, 89, 100, 200, 233, 500, 1000, 1597],
    num_dp=4,
    messages=False,
):
    import matplotlib.pyplot as plt
    import numpy as np
    try:
        import fireducks.pandas as pd
    except ImportError:
        import pandas as pd
    from matplotlib.cm import rainbow
    from sklearn.ensemble import RandomForestClassifier as RandomForest
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, matthews_corrcoef, precision_recall_fscore_support

    rf_scores_dict = {}
    max_k = len(estimators)

    for k in estimators:
        rf_clf = RandomForest(
            n_estimators=k, random_state=RANDOM_SEED if RANDOM_SEED else 42
        )
        rf_clf.fit(X_train, y_train)
        rf_pred = rf_clf.predict(X_test)
        rf_acc = accuracy_score(y_test, rf_pred)
        rf_f1 = f1_score(y_test, rf_pred)
        rf_mcc = matthews_corrcoef(y_test, rf_pred)

        rf_scores_dict[f"{k}_estimators"] = {}
        rf_scores_dict[f"{k}_estimators"]["acc_score"] = rf_acc
        rf_scores_dict[f"{k}_estimators"]["f1_score"] = rf_f1
        rf_scores_dict[f"{k}_estimators"]["matthews_corrcoef"] = rf_mcc

    if messages:
        print("All RFC Scores:")
        for key01, val01 in rf_scores_dict.items():
            if not isinstance(val01, dict):
                if isinstance(val01, (int, float)):
                    print(f"{key01}: {round(val01, num_dp)}")
                else:
                    print(f"{key01}: {val01}")
                continue
            else:
                print(f"{key01}")
                for key02, val02 in val01.items():
                    print(f"{key02}: {round(val02, num_dp)}")
                print("*********************")
            print()
        # Plot the scores on a barplot
        # colours = rainbow(np.linspace(0, 1, max_k))
        # plt.bar([i for i in range(max_k)], rf_acc_scores_list, color=colours, width=0.37)
        # for i in range(max_k):
        #     plt.text(
        #         i,
        #         rf_acc_scores_list[i],
        #         f"{round(rf_acc_scores_list[i], num_dp)}",
        #         rotation=60,
        #         va="bottom",
        #         fontsize=10,
        #     )
        # plt.xticks(
        #     ticks=[i for i in range(max_k)],
        #     labels=[str(estimator) for estimator in estimators],
        #     rotation=45,
        # )
        # y_bottom, y_top = plt.ylim()
        # plt.ylim(top=y_top * 1.13)
        # plt.xlabel("Number of Estimators")
        # plt.ylabel("Accuracy Scores")
        # plt.title(
        #     "Random Forest Classifier Accuracy Scores for Different Number of Estimators"
        # )
    return rf_scores_dict



## Support Vector Classification Function
def support_vector_classification(
    X_train,
    y_train,
    X_test,
    y_test,
    kernels=["linear", "poly", "rbf", "sigmoid", "precomputed"],
    num_dp=4,
    messages=False,
):
    import matplotlib.pyplot as plt
    import numpy as np
    try:
        import fireducks.pandas as pd
    except ImportError:
        import pandas as pd
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, matthews_corrcoef, precision_recall_fscore_support
    from sklearn.svm import SVC

    svc_scores_dict = {}
    max_k = len(kernels)

    for k in range(max_k):
        svc_clf = SVC(kernel=kernels[k])
        svc_clf.fit(X_train, y_train)
        svc_pred = svc_clf.predict(X_test)
        svc_acc = accuracy_score(y_test, svc_pred)
        svc_f1 = f1_score(y_test, svc_pred)
        svc_mcc = matthews_corrcoef(y_test, svc_pred)

        svc_scores_dict[f"{k}_kernel"] = {}
        svc_scores_dict[f"{k}_kernel"]["acc_score"] = svc_acc
        svc_scores_dict[f"{k}_kernel"]["f1_score"] = svc_f1
        svc_scores_dict[f"{k}_kernel"]["matthews_corrcoef"] = svc_mcc

    if messages:
        print("All SVC Scores:")
        for key01, val01 in svc_scores_dict.items():
            if not isinstance(val01, dict):
                if isinstance(val01, (int, float)):
                    print(f"{key01}: {round(val01, num_dp)}")
                else:
                    print(f"{key01}: {val01}")
                continue
            else:
                print(f"{key01}")
                for key02, val02 in val01.items():
                    print(f"{key02}: {round(val02, num_dp)}")
                print("*********************")
            print()
        # Plot the scores on a barplot
        # colours = rainbow(np.linspace(0, 1, max_k))
        # plt.bar(kernels, svc_acc_scores_list, color=colours)
        # for i in range(max_k):
        #     plt.text(
        #         i,
        #         svc_acc_scores_list[i],
        #         f"{round(svc_acc_scores_list[i], num_dp)}",
        #         rotation=45,
        #         va="bottom",
        #         fontsize=10,
        #     )
        # y_bottom, y_top = plt.ylim()
        # plt.ylim(top=y_top * 1.13)
        # plt.xlabel("Kernels")
        # plt.ylabel("Accuracy Scores")
        # plt.title("Support Vector Classifier Accuracy Scores for Different Kernels")
    return svc_scores_dict


def ensemble_classifier(X_train, y_train, X_test, y_test, num_dp=4, messages=False,):
    import matplotlib.pyplot as plt
    import numpy as np
    try:
        import fireducks.pandas as pd
    except ImportError:
        import pandas as pd
    from sklearn.ensemble import RandomForestClassifier as RandomForest
    from sklearn.ensemble import VotingClassifier
    from sklearn.experimental import enable_halving_search_cv
    from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, confusion_matrix, f1_score, matthews_corrcoef, precision_recall_fscore_support
    from sklearn.model_selection import HalvingGridSearchCV
    from sklearn.svm import SVC

    ensemble_scores_dict = {}

    # Using Base Models

    # Instantiate base models
    base_svm_clf = SVC()
    base_rf_clf = RandomForest()

    # Create base ensemble model using VotingClassifier
    base_ensemble_classifier = VotingClassifier(estimators=[("svm", base_svm_clf), ("random_forest", base_rf_clf)], voting="hard")
    # Train the base ensemble model
    base_ensemble_classifier.fit(X_train, y_train)
    # Predictions from the base ensemble model
    base_ensemble_pred = base_ensemble_classifier.predict(X_test)
    base_ensemble_acc = accuracy_score(y_test, base_ensemble_pred)
    base_ensemble_f1 = f1_score(y_test, base_ensemble_pred)
    base_ensemble_mcc = matthews_corrcoef(y_test, base_ensemble_pred)
    ensemble_scores_dict["base"] = {}
    ensemble_scores_dict["base"]["acc_score"] = base_ensemble_acc
    ensemble_scores_dict["base"]["f1_score"] = base_ensemble_f1
    ensemble_scores_dict["base"]["matthews_corrcoef"] = base_ensemble_mcc
    if messages:
        titles_options01 = [("Base Confusion Matrix Without Normalisation", None), ("Normalised Base Confusion Matrix", "true"),]
        for title, normalise in titles_options01:
            disp01 = ConfusionMatrixDisplay.from_estimator(base_ensemble_classifier, X_test, y_test, cmap=plt.cm.Blues, normalize=normalise)
            print(title)
            print(disp01.confusion_matrix)
        plt.show()

    # Using Halving-Grid Tuned Models
    svm_param_grid = {
        "C": [0.1, 0.3, 0.5, 0.7, 1, 2, 3, 5, 7, 10, 11, 13, 17, 19, 23],
        "gamma": [0.1, 0.05, 0.01, 0.001],
        "kernel": ["linear", "poly", "rbf", "sigmoid", "precomputed"],
    }

    rf_param_grid = {
        "n_estimators": [2, 3, 5, 7, 10, 13, 89, 100, 200, 233, 300, 500, 1000, 1597],
        "max_depth": [None, 1, 2, 3, 5, 7, 10, 11, 13, 17, 19, 23],
        "min_samples_leaf": [1, 2, 3, 4, 5, 7, 10, 11, 13, 17, 19, 23],
    }

    halving_svm_clf = HalvingGridSearchCV(base_svm_clf, svm_param_grid, factor=2, cv=11)
    halving_svm_clf.fit(X_train, y_train)

    halving_rf_clf = HalvingGridSearchCV(base_rf_clf, rf_param_grid, factor=2, cv=11)
    halving_rf_clf.fit(X_train, y_train)

    # Create halving-grid-search ensemble model using VotingClassifier
    halving_ensemble_classifier = VotingClassifier(estimators=[("svm", halving_svm_clf), ("random_forest", halving_rf_clf)], voting="hard")

    # Train and assess prediction from the halving grid search ensemble model
    halving_ensemble_classifier.fit(X_train, y_train)
    halving_ensemble_pred = base_ensemble_classifier.predict(X_test)
    halving_ensemble_acc = accuracy_score(y_test, halving_ensemble_pred)
    halving_ensemble_f1 = f1_score(y_test, halving_ensemble_pred)
    halving_ensemble_mcc = matthews_corrcoef(y_test, halving_ensemble_pred)
    ensemble_scores_dict["halving"] = {}
    ensemble_scores_dict["halving"]["acc_score"] = halving_ensemble_acc
    ensemble_scores_dict["halving"]["f1_score"] = halving_ensemble_f1
    ensemble_scores_dict["halving"]["matthews_corrcoef"] = halving_ensemble_mcc
    if messages:
        titles_options02 = [("Halving Confusion Matrix Without Normalisation", None), ("Normalised Halving Confusion Matrix", "true"),]
        for title, normalise in titles_options02:
            disp02 = ConfusionMatrixDisplay.from_estimator(halving_ensemble_classifier, X_test, y_test, cmap=plt.cm.Blues, normalize=normalise)
            print(title)
            print(disp02.confusion_matrix)
        plt.show()

    # Using Optimised Ensemble Models
    ensemble_param_grid = {
        "svm__C": [0.1, 0.3, 0.5, 0.7, 1, 2, 3, 5, 7, 10, 11, 13, 17, 19, 23],
        "svm__gamma": [0.1, 0.05, 0.01, 0.001],
        "svm__kernel": ["linear", "poly", "rbf", "sigmoid", "precomputed"],
        "random_forest__n_estimators": [2, 3, 5, 7, 10, 13, 89, 100, 200, 233, 300, 500, 1000, 1597],
        "random_forest__max_depth": [None, 1, 2, 3, 5, 7, 10, 11, 13, 17, 19, 23],
        "random_forest__min_samples_leaf": [1, 2, 3, 4, 5, 7, 10, 11, 13, 17, 19, 23],
    }

    ensemble_clf_halving_grid_search = HalvingGridSearchCV(halving_ensemble_classifier, ensemble_param_grid, factor=2, cv=11)

    # Train and assess prediction from the fine-tuned halving grid search ensemble model
    ensemble_clf_halving_grid_search.fit(X_train, y_train)
    ensemble_clf_halving_grid_search_pred = ensemble_clf_halving_grid_search.predict(X_test)
    ensemble_clf_halving_grid_search_acc = accuracy_score(y_test, ensemble_clf_halving_grid_search_pred)
    ensemble_clf_halving_grid_search_f1 = f1_score(y_test, ensemble_clf_halving_grid_search_pred)
    ensemble_clf_halving_grid_search_mcc = matthews_corrcoef(y_test, ensemble_clf_halving_grid_search_pred)
    ensemble_scores_dict["tuned"] = {}
    ensemble_scores_dict["tuned"]["acc_score"] = ensemble_clf_halving_grid_search_acc
    ensemble_scores_dict["tuned"]["f1_score"] = ensemble_clf_halving_grid_search_f1
    ensemble_scores_dict["tuned"]["matthews_corrcoef"] = ensemble_clf_halving_grid_search_mcc
    if messages:
        titles_options03 = [("Tuned Confusion Matrix Without Normalisation", None), ("Normalised Tuned Confusion Matrix", "true"),]
        for title, normalise in titles_options03:
            disp03 = ConfusionMatrixDisplay.from_estimator(halving_ensemble_classifier, X_test, y_test, cmap=plt.cm.Blues, normalize=normalise)
            print(title)
            print(disp03.confusion_matrix)
        plt.show()

    # Return scores
    return ensemble_scores_dict





