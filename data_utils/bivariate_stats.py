#!/usr/bin/env python3

# ----------------------------------------------------------------------#
# Functions to get bivariate statistics and plots from Pandas DataFrame #
# ----------------------------------------------------------------------#
def scatterplot(df, feature, label, num_dp=4, linecolour="darkorange"):
    """
    Creates a scatterplot between two features in a DataFrame, with a regression line included.

    Parameters
    ----------
    - df : pandas.DataFrame
        The DataFrame containing the data
    - feature : str
        The feature to plot on the x-axis
    - label : str
        The feature to plot on the y-axis
    - num_dp : int
        The number of decimal places to round the regression equation to
    - linecolour : str
        The colour of the regression line

    Returns
    -------
    None

    Example
    -------
    >>> # Sample data
    >>> rng = np.random.default_rng()
    >>> age = rng.integers(low=18, high=61, size=103)
    >>> sex = rng.choice(["female", "male"], size=103)
    >>> bmi = rng.normal(loc=23, scale=7, size=103)
    >>> children = rng.integers(low=0, high=9, size=103)
    >>> smoker = rng.choice(["smoker", "non-smoker"], size=103)
    >>> region = rng.choice(["southwest", "northwest", "southeast", "northeast"], size=103)
    >>> charges = rng.normal(loc=5300, scale=2113, size=103)
    >>>
    >>> data_dict = {"age": age, "sex": sex, "bmi": bmi, "children": children, "smoker": smoker, "region": region, "charges": charges}
    >>> df = pd.DataFrame(data_dict)
    >>>
    >>> scatterplot(df, "age", "charges")
    
    Author
    ------
        tunprimus
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    try:
        import fireducks.pandas as pd
    except ImportError:
        import pandas as pd
    from scipy import stats

    # Create the plot
    sns.regplot(x=df[feature], y=df[label], line_kws={"color": linecolour})
    # Calculate the regression line
    ## Normality satisfied
    results = stats.linregress(df[feature], df[label])
    slope = results.slope
    slope = round(slope, num_dp)
    intercept = results.intercept
    intercept = round(intercept, num_dp)
    r = results.rvalue
    r = round(r, num_dp)
    p = results.pvalue
    p = round(p, num_dp)
    stderr = results.stderr
    intercept_stderr = results.intercept_stderr
    ## Other linear regressions
    results_k = stats.kendalltau(df[feature], df[label])
    tau = results_k.statistic
    tau = round(tau, num_dp)
    tp = results_k.pvalue
    tp = round(tp, num_dp)
    results_r = stats.spearmanr(df[feature], df[label])
    rho = results_r.statistic
    rho = round(rho, num_dp)
    rp = results_r.pvalue
    rp = round(rp, num_dp)
    ## Skew
    feature_skew = round((df[feature].skew()), num_dp)
    label_skew = round((df[label].skew()), num_dp)
    # Create text string
    text_str = f"y = {slope}x + {intercept}\n"
    text_str += f"r = {r}, p = {p}\n"
    text_str += f"τ = {tau}, p = {tp}\n"
    text_str += f"ρ = {rho}, p = {rp}\n"
    text_str += f"{feature} skew = {feature_skew}\n"
    text_str += f"{label} skew = {label_skew}"
    # Add annotations
    plt.text(0.95, 0.2, text_str, fontsize=12, transform=plt.gcf().transFigure)
    # Show plot
    plt.show()


def bar_chart(df, feature, label, num_dp=4, alpha=0.05, sig_ttest_only=True):
    """
    Creates a bar chart with a one-way ANOVA and pairwise t-tests with Bonferroni correction.

    Parameters
    ----------
    - df : pandas.DataFrame
        The DataFrame containing the data
    - feature : str
        The feature to plot on the x-axis
    - label : str
        The feature to plot on the y-axis
    - num_dp : int
        The number of decimal places to round the results to
    - alpha : float
        The significance level for the t-tests
    - sig_ttest_only : bool
        If True, only print the t-tests with p <= alpha / number of ttest comparisons

    Returns
    -------
    None

    Example
    -------
    >>> # Sample data
    >>> rng = np.random.default_rng()
    >>> age = rng.integers(low=18, high=61, size=103)
    >>> sex = rng.choice(["female", "male"], size=103)
    >>> bmi = rng.normal(loc=23, scale=7, size=103)
    >>> children = rng.integers(low=0, high=9, size=103)
    >>> smoker = rng.choice(["smoker", "non-smoker"], size=103)
    >>> region = rng.choice(["southwest", "northwest", "southeast", "northeast"], size=103)
    >>> charges = rng.normal(loc=5300, scale=2113, size=103)
    >>>
    >>> data_dict = {"age": age, "sex": sex, "bmi": bmi, "children": children, "smoker": smoker, "region": region, "charges": charges}
    >>> df = pd.DataFrame(data_dict)
    >>>
    >>> bar_chart(df, "smoker", "charges")
    
    Author
    ------
        tunprimus
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    try:
        import fireducks.pandas as pd
    except ImportError:
        import pandas as pd
    from scipy import stats

    # Make sure that the feature is categorical and the label is numerical
    if pd.api.types.is_numeric_dtype(df[feature]):
        num = feature
        cat = label
    else:
        num = label
        cat = feature
    # Create the plot
    sns.barplot(x=df[cat], y=df[num])
    # Create the numerical lists to calculate the ANOVA
    groups = df[cat].unique()
    # print(groups)
    group_lists = []
    for g in groups:
        n_list = df[df[cat] == g][num]
        group_lists.append(n_list)
    F, p = stats.f_oneway(*group_lists)
    F, p = round(F, num_dp), round(p, num_dp)
    # Calculate pairwise t-test for groups
    ttests = []
    for i1, g1 in enumerate(groups):
        for i2, g2 in enumerate(groups):
            if i2 > i1:
                list01 = df[df[cat] == g1][num]
                list02 = df[df[cat] == g2][num]
                ttest_result = stats.ttest_ind(list01, list02)
                ttest = ttest_result.statistic
                ttest = round(ttest, num_dp)
                ttest_p = ttest_result.pvalue
                ttest_p = round(ttest_p, num_dp)
                # if ttest_result.df or ttest_result.confidence_interval():
                #     dof = ttest_result.df
                #     dof = round(dof, num_dp)
                #     low_ci = ttest_result.confidence_interval()[0]
                #     low_ci = round(low_ci, num_dp)
                #     high_ci = ttest_result.confidence_interval()[1]
                #     high_ci = round(high_ci, num_dp)
                # ttests.append([f"{g1} vs {g2}", ttest, ttest_p, dof, low_ci, high_ci])
                ttests.append([f"{g1} vs {g2}", ttest, ttest_p])
    # Bonferroni correction -> adjust p-value threshold to be 0.05/number of ttest comparisons
    bonferroni = alpha / len(ttests) if len(ttests) > 0 else 0
    bonferroni = round(bonferroni, num_dp)
    # Create text string
    text_str = f"F: {F}\n"
    text_str += f"p: {p}\n"
    text_str += f"Bonferroni p: {bonferroni}"
    for ttest in ttests:
        if sig_ttest_only:
            if ttest[2] <= bonferroni:
                # text_str += f"\n{ttest[0]}: t = {ttest[1]}, p = {ttest[2]}, dof = {ttest[3]}, CI = [{ttest[4]}, {ttest[5]}]"
                text_str += f"\n{ttest[0]}:\n     t = {ttest[1]}, p = {ttest[2]}"
        else:
            text_str += f"\n{ttest[0]}: t = {ttest[1]}, p = {ttest[2]}"
    # If there are too many feature groups, print x labels vertically
    if df[feature].nunique() > 7:
        plt.xticks(rotation=90)
    # Annotations
    plt.text(0.95, 0.1, text_str, fontsize=12, transform=plt.gcf().transFigure)
    # Show plot
    plt.show()


def crosstab(df, feature, label, num_dp=4):
    """
    Creates a heatmap of a contingency table between two categorical features in a DataFrame and calculates the Chi-Squared statistic.

    Parameters
    ----------
    - df : pandas.DataFrame
        The DataFrame containing the data
    - feature : str
        The feature to plot on the x-axis
    - label : str
        The feature to plot on the y-axis
    - num_dp : int
        The number of decimal places to round the Chi-Squared results to

    Returns
    -------
    None

    Example
    -------
    >>> # Sample data
    >>> rng = np.random.default_rng()
    >>> age = rng.integers(low=18, high=61, size=103)
    >>> sex = rng.choice(["female", "male"], size=103)
    >>> bmi = rng.normal(loc=23, scale=7, size=103)
    >>> children = rng.integers(low=0, high=9, size=103)
    >>> smoker = rng.choice(["smoker", "non-smoker"], size=103)
    >>> region = rng.choice(["southwest", "northwest", "southeast", "northeast"], size=103)
    >>> low_earn = rng.normal(loc=997, scale=13, size=103)
    >>> normal_earn = rng.normal(loc=7933, scale=23, size=103)
    >>> high_earn = rng.normal(loc=20011, scale=43, size=103)
    >>> monthly_salary = rng.choice(np.concatenate([low_earn, normal_earn, high_earn]), size=103)
    >>> cust_class = rng.choice(["economy", "business", "first-class"], size=103)
    >>> satisfaction = rng.choice(["extremely dissatisfied", "dissatisfied", "fair", "okay", "satisfied", "extremely satisfied", "excellent"], size=103)
    >>> charges = rng.normal(loc=5300, scale=2113, size=103)
    >>>
    >>> data_dict = {"age": age, "sex": sex, "bmi": bmi, "children": children, "smoker": smoker, "region": region, "monthly_salary": monthly_salary, "cust_class": cust_class, "satisfaction": satisfaction, "charges": charges}
    >>> df = pd.DataFrame(data_dict)
    >>>
    >>> crosstab(df, "class", "satisfaction")
    
    Author
    ------
        tunprimus
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    try:
        import fireducks.pandas as pd
    except ImportError:
        import pandas as pd
    from scipy import stats

    contingency_table = pd.crosstab(df[feature], df[label])
    results = stats.chi2_contingency(contingency_table)
    X2 = results.statistic
    X2 = round(X2, num_dp)
    p = results.pvalue
    p = round(p, num_dp)
    dof = results.dof
    dof = round(dof, num_dp)
    expected_freq = results.expected_freq
    # Create text string
    text_str = f"X2: {X2}\n"
    text_str += f"p: {p}\n"
    text_str += f"dof: {dof}"
    # Annotations
    plt.text(0.95, 0.2, text_str, fontsize=12, transform=plt.gcf().transFigure)
    # Generate heatmap
    ct_df = pd.DataFrame(
        np.rint(expected_freq).astype("int64"),
        columns=contingency_table.columns,
        index=contingency_table.index,
    )
    sns.heatmap(ct_df, annot=True, fmt="d", cmap="coolwarm")
    # Show plot
    plt.show()


def bivariate_stats(df, label, num_dp=4):
    """
    Generates a DataFrame containing a variety of bivariate statistics for each feature in a DataFrame vs a given label.

    Parameters
    ----------
    - df : pandas.DataFrame
        The DataFrame containing the data
    - label : str
        The label to be used for the target variable
    - num_dp : int
        The number of decimal places to round the results to

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the results of the bivariate statistics

	Example
	-------
	>>> # Sample data
	>>> rng = np.random.default_rng()
	>>> age = rng.integers(low=18, high=61, size=103)
	>>> sex = rng.choice(["female", "male"], size=103)
	>>> bmi = rng.normal(loc=23, scale=7, size=103)
	>>> children = rng.integers(low=0, high=9, size=103)
	>>> smoker = rng.choice(["smoker", "non-smoker"], size=103)
	>>> region = rng.choice(["southwest", "northwest", "southeast", "northeast"], size=103)
	>>> low_earn = rng.normal(loc=997, scale=13, size=103)
	>>> normal_earn = rng.normal(loc=7933, scale=23, size=103)
	>>> high_earn = rng.normal(loc=20011, scale=43, size=103)
	>>> monthly_salary = rng.choice(np.concatenate([low_earn, normal_earn, high_earn]), size=103)
    >>> cust_class = rng.choice(["economy", "business", "first-class"], size=103)
    >>> satisfaction = rng.choice(["extremely dissatisfied", "dissatisfied", "fair", "okay", "satisfied", "extremely satisfied", "excellent"], size=103)
    >>> charges = rng.normal(loc=5300, scale=2113, size=103)
    >>>
    >>> data_dict = {"age": age, "sex": sex, "bmi": bmi, "children": children, "smoker": smoker, "region": region, "monthly_salary": monthly_salary, "cust_class": cust_class, "satisfaction": satisfaction, "charges": charges}
	>>> df = pd.DataFrame(data_dict)
	>>>
	>>> bivariate_stats(df, "charges")
    >>> bivariate_stats(df, "satisfaction")
    >>> bivariate_stats(df, "monthly_salary")
    
    Author
    ------
        tunprimus
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    # Avoid below due to bug in Fireducks
    # https://github.com/fireducks-dev/fireducks/issues/45
    # try:
    #     import fireducks.pandas as pd
    # except ImportError:
    #     import pandas as pd
    from scipy import stats

    output_df = pd.DataFrame(
        columns=[
            "missing",
            "missing_%",
            "skew",
            "type",
            "num_unique",
            "p",
            "r",
            "tau",
            "rho",
            "y = m(x) + b",
            "F",
            "X2",
        ]
    )

    for feature in df.columns:
        if feature != label:
            # Calculate statistics that apply to all datatypes
            df_temp = df[[feature, label]].copy()
            df_temp = df_temp.dropna().copy()
            missing = df.shape[0] - df_temp.shape[0]
            buffer = (df.shape[0] - df_temp.shape[0]) / df.shape[0]
            missing_pct = round(buffer * 100, num_dp)
            dtype = df_temp[feature].dtype
            num_unique = df_temp[feature].nunique()
            if (pd.api.types.is_numeric_dtype(df_temp[feature])) and (
                pd.api.types.is_numeric_dtype(df_temp[label])
            ):
                # Process N2N relationships
                ## Pearson linear regression
                results_p = stats.linregress(df_temp[feature], df_temp[label])
                slope = results_p.slope
                slope = round(slope, num_dp)
                intercept = results_p.intercept
                intercept = round(intercept, num_dp)
                r = results_p.rvalue
                r = round(r, num_dp)
                p = results_p.pvalue
                p = round(p, num_dp)
                stderr = results_p.stderr
                intercept_stderr = results_p.intercept_stderr
                ## Other linear regressions
                results_k = stats.kendalltau(df_temp[feature], df_temp[label])
                tau = results_k.statistic
                tau = round(tau, num_dp)
                tp = results_k.pvalue
                results_r = stats.spearmanr(df_temp[feature], df_temp[label])
                rho = results_r.statistic
                rho = round(rho, num_dp)
                rp = results_r.pvalue
                ## Skew
                skew = round((df_temp[feature].skew()), num_dp)
                output_df.loc[feature] = [
                    missing,
                    f"{missing_pct}%",
                    skew,
                    dtype,
                    num_unique,
                    p,
                    r,
                    tau,
                    rho,
                    f"y = {slope}x + {intercept}",
                    "--",
                    "--",
                ]
                scatterplot(df_temp, feature, label)
            elif not (pd.api.types.is_numeric_dtype(df_temp[feature])) and not (
                pd.api.types.is_numeric_dtype(df_temp[label])
            ):
                # Process C2C relationships
                contingency_table = pd.crosstab(df_temp[feature], df_temp[label])
                results = stats.chi2_contingency(contingency_table)
                X2 = results.statistic
                X2 = round(X2, num_dp)
                p = results.pvalue
                p = round(p, num_dp)
                dof = results.dof
                expected_freq = results.expected_freq
                output_df.loc[feature] = [
                    missing,
                    f"{missing_pct}%",
                    "--",
                    dtype,
                    num_unique,
                    p,
                    "--",
                    "--",
                    "--",
                    "--",
                    "--",
                    X2,
                ]
                crosstab(df_temp, feature, label)
            else:
                # Process C2N and N2C relationships
                if pd.api.types.is_numeric_dtype(df_temp[feature]):
                    num = feature
                    cat = label
                    skew = round((df_temp[feature].skew()), num_dp)
                else:
                    num = label
                    cat = feature
                    skew = "--"
                groups = df_temp[cat].unique()
                group_lists = []
                for g in groups:
                    n_list = df_temp[df_temp[cat] == g][num]
                    group_lists.append(n_list)
                F, p = stats.f_oneway(*group_lists)
                F, p = round(F, num_dp), round(p, num_dp)
                output_df.loc[feature] = [
                    missing,
                    f"{missing_pct}%",
                    skew,
                    dtype,
                    num_unique,
                    p,
                    "--",
                    "--",
                    "--",
                    "--",
                    F,
                    "--",
                ]
                bar_chart(df_temp, cat, num)
    try:
        return output_df.sort_values(by="p", ascending=True)
    except Exception:
        return output_df

