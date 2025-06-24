#!/usr/bin/env python3


def info_value_n_weight_of_evidence_calc(
    df, target_column, num_bins=10, num_dp=8, messages=False
):
    """
    Calculate information value (IV) and weight of evidence (WoE) for a given dataframe and target column.

    Parameters
    ----------
    df:  pd.DataFrame
        Input Pandas DataFrame
    target_column: str
        Name of the target column
    num_bins: int, optional
        Number of bins for numerical features. Defaults to 10.
    num_dp: int, optional
        Number of decimal places to round results to. Defaults to 8.
    messages: bool, optional
        If True, prints IV and WoE dataframes. Defaults to False.

    Returns
    -------
    iv_data: pd.DataFrame
        DataFrame with IV, IV_pred_power, and feature names
    woe_data: pd.DataFrame
        DataFrame with WoE values for each feature

    Notes
    -----
    This function processes each feature while skipping the target and high cardinality features.
    For numerical features, it bins using quantiles and ensures at least 5% of elements in each bin.
    For categorical features, it computes events and non-events.
    It then calculates WoE and IV for each feature and prints the results if messages is True.

        Example
        -------
        >>> # Sample data
        >>> rng = np.random.default_rng(42)
        >>>
        >>> surnames = ["Olafusi", "Abiola", "Eze", "Owolabi", "Owoeye", "Nnamdi", "Aluko", "Bello", "Tsangi", "Hoffman", "Williams", "Stone", "Carter", "Gilbert", "Oconnor", "Hill", "Hernandez", "Leonard", "Berg", "Hans-Otto", "Riehl", "Biggen", "Pareja", "Vicenta", "Coronado", "Bonet", "Blanch", "Fontana", "Ferri", "Roosenboom", "Huijzing", "Nijman",]
        >>>
        >>> firstnames = ["Michael", "John", "Mary", "Segun", "Tolu", "Uche", "David", "Lekan", "Luke", "Jason", "Scott", "Jordan", "Hannah", "Robert", "Patricia", "Carla", "Nancy", "James", "Anne", "Mariusz", "Dagobert", "Sahin", "Francesca", "Niels", "Remedios", "Sonia", "Angela", "Catalina", "Sarah", "Joshua", "Fabio", "Sienna",]
        >>>
        >>> prefixes = ["Ade", "Olu", "Al", "Oluwa", "Chi", "Wa", "Fa", "Bet", "Sa", "Ah", "Abu", "Bint", "Ter", "Bath", "Ben", "Mac", "Nic", "Del", "Ab", "Bar", "Kil", "Mal", "Öz", "Mala", "Dos", "Fitz", "Ibn", "Du", "Gil", "Tre", "Mul", "Av",]
        >>>
        >>> suffixes = ["bert", "stan", "addin", "tou", "zi", "jiā", "biɑn", "chan", "kun", "san", "sama", "shi", "ujin", "joshi", "ach", "ant", "appa", "anu", "chian", "eanu", "enko", "ius", "kar", "onak", "on", "oui", "quin", "sen", "ulis", "ema", "awan", "ak",]
        >>>
        >>>
        >>> def random_date_gen(start_date, range_in_days, count):
        >>>     try:
        >>>         start_date = np.datetime64(start_date)
        >>>         base = np.full(count, start_date)
        >>>         offset = rng.integers(low=0, high=range_in_days, size=count)
        >>>         offset = offset.astype("timedelta64[D]")
        >>>         random_date = base + offset
        >>>     except Exception:
        >>>         days_to_add = np.arange(0, range_in_days)
        >>>         random_date = np.datetime64(start_date) + rng.choice(days_to_add)
        >>>
        >>>     return random_date
        >>>
        >>>
        >>> def generate_name_advanced():
        >>>     surname = rng.choice(surnames)
        >>>     firstname = rng.choice(firstnames)
        >>>     prefix = rng.choice(prefixes)
        >>>     suffix = rng.choice(suffixes)
        >>>     use_prefix_surname = rng.choice([True, False])
        >>>     use_prefix_firstname = rng.choice([True, False])
        >>>     use_suffix_surname = rng.choice([True, False])
        >>>     use_suffix_firstname = rng.choice([True, False])
        >>>     if use_prefix_surname:
        >>>         fullname = f"{prefix}{surname.lower()} {firstname}"
        >>>     elif use_prefix_firstname:
        >>>         fullname = f"{surname} {prefix}{firstname.lower()}"
        >>>     elif use_prefix_surname and use_prefix_firstname:
        >>>         fullname = f"{prefix}{surname.lower()} {prefix}{firstname.lower()}"
        >>>     elif use_suffix_surname:
        >>>         fullname = f"{surname}{suffix} {firstname}"
        >>>     elif use_suffix_firstname:
        >>>         fullname = f"{surname} {firstname}{suffix}"
        >>>     elif use_suffix_surname and use_suffix_firstname:
        >>>         fullname = f"{surname}{suffix} {firstname}{suffix}"
        >>>     else:
        >>>         fullname = f"{surname} {firstname}"
        >>>
        >>>     return fullname
        >>>
        >>>
        >>> name = [generate_name_advanced() for _ in range(103)]
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
        satisfaction = rng.choice(["extremely dissatisfied", "dissatisfied", "fair", "okay", "satisfied", "extremely satisfied", "excellent"], size=103)
        >>> charges = rng.normal(loc=5300, scale=2113, size=103)
        >>> onboard_date = random_date_gen("2023-01-01", 365, 103)
        >>>
        >>> data_dict = {"name": name, "age": age, "sex": sex, "bmi": bmi, "children": children, "smoker": smoker, "region": region, "monthly_salary": monthly_salary, "cust_class": cust_class, "satisfaction": satisfaction, "charges": charges, "onboard_date": onboard_date}
        >>> df = pd.DataFrame(data_dict)
        >>>
        >>> info_value_n_weight_of_evidence_calc(df, "age")
    >>> info_value_n_weight_of_evidence_calc(df, "charges")

    Author
    ------
        tunprimus
    """
    import numpy as np

    try:
        import fireducks.pandas as pd
    except ImportError:
        import pandas as pd

    # Extract column names
    cols = df.columns

    # Check that target_column is in the list of columns
    try:
        assert target_column in cols
    except Exception as exc:
        raise ValueError(f"{target_column} not in DataFrame columns")

    # Check that target is numeric
    try:
        assert df.groupby(cols[0])[target_column].agg(["sum"]).values.dtype.kind in [
            "b",
            "f",
            "i",
            "u",
            "c",
        ]
    except Exception as exc:
        raise ValueError(f"Values of target column - {target_column} - must be numeric")

    # Copy data to avoid modifying the original
    data = df.copy()

    # Initialise lists
    iv_values = []
    woe_data = data[[target_column]].copy()

    def adjust_bins(series, num_bin, min_bin_percent=0.05):
        """
        Adjust the number of bins for a Pandas series such that each bin is at least a certain percentage of the original array.

        Parameters
        ----------
        series: pd.Series
            Input Pandas series
        num_bin: int
            Initial number of bins.
        min_bin_percent: float, optional
            Minimum percentage of the original array for each bin. Defaults to 0.05 (5%).

        Returns
        -------
        int: Adjusted number of bins
        """
        # Calculate the minimum number of samples per bin
        min_samples_per_bin = int(len(series) * min_bin_percent)

        # Calculate the adjusted number of bins
        adjusted_bins = min(num_bin, int(np.ceil(len(series) / min_samples_per_bin)))

        return adjusted_bins if (adjusted_bins < num_bin) else num_bin

    # Process each feature while skipping the target and high cardinality features
    for col in data.columns:
        if (col == target_column) or (data[col].nunique() > 20):
            continue

        # Handle numerical features: bin using quantiles
        if data[col].dtype.kind in "bifc":
            # Ensure at least 5% of elements in each bin
            optimal_num_bins = adjust_bins(data[col], num_bins)
            data[col + "_bin"] = pd.qcut(
                data[col], q=optimal_num_bins, duplicates="drop", labels=False
            )
            feature = col + "_bin"
        else:
            feature = col

        # Compute events and non-events only for numeric features
        counts = data.groupby(feature)[target_column].agg(["count", "sum"])
        if counts["sum"].dtype.kind in "bifc":
            counts["non_event"] = counts["count"] - counts["sum"]
        else:
            continue
        counts["event"] = counts["sum"]

        # Add small constant to avoid division by zero
        total_events = data[target_column].sum()
        total_non_events = data[target_column].count() - total_events
        counts["event_dist"] = (counts["event"] + 0.5) / (
            (total_events + 0.5) * len(counts)
        )
        counts["non_event_dist"] = (counts["non_event"] + 0.5) / (
            (total_non_events + 0.5) * len(counts)
        )

        # Calculate WOE
        counts["WoE"] = np.log(counts["non_event_dist"] / counts["event_dist"])

        # Calculate IV
        counts["IV_contrib"] = (
            counts["non_event_dist"] - counts["event_dist"]
        ) * counts["WoE"]
        info_val = counts["IV_contrib"].sum()

        # Compare IV to thresholds to get predict value
        condition_list = [
            (info_val < 0.02),
            (info_val >= 0.02) & (info_val < 0.1),
            (info_val >= 0.1) & (info_val < 0.3),
            (info_val >= 0.3) & (info_val < 0.5),
            (info_val > 0.5),
        ]
        choice_list = [
            "useless for prediction",
            "weak predictor",
            "medium predictor",
            "strong predictor",
            "suspicious or too good to be true",
        ]
        iv_pred_power = np.select(condition_list, choice_list, default="")
        counts["IV_pred_power"] = iv_pred_power
        iv_values.append(
            {
                "Feature": col,
                "IV": round(info_val, num_dp),
                "IV_pred_power": iv_pred_power,
            }
        )

        # Transform feature to WOE values
        woe_map = counts["WoE"].to_dict()
        woe_data[col + "_WoE"] = data[feature].map(woe_map)

    # Create IV DataFrame
    iv_data = pd.DataFrame(iv_values)

    # Sort IV DataFrame
    try:
        iv_data = iv_data.sort_values(by="IV", ascending=False)
    except Exception as exc:
        iv_data

    # Print IV DataFrame if messages is True
    if messages:
        print(iv_data)
    return iv_data, woe_data

