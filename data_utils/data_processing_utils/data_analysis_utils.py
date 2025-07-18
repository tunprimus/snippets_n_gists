#!/usr/bin/env python3
from matplotlib import rcParams
from matplotlib.cm import rainbow
from auto_pip_finder import PipFinder
from dbscan_pp import DBSCANPP


## Some Constants
##*********************##
GOLDEN_RATIO = 1.618033989
FIG_WIDTH = 12
FIG_HEIGHT = FIG_WIDTH / GOLDEN_RATIO
FIG_SIZE = (FIG_WIDTH, FIG_HEIGHT)
FIG_DPI = 72
RANDOM_SEED = 42
RANDOM_SAMPLE_SIZE = 13
NUM_DEC_PLACES = 4
ALPHA_VALUE = 0.05


## Some Useful Functions
##*********************##


### Get Number of CPU Logical Cores
def cpu_logical_cores_count():
    """
    Return the number of logical cores on the machine.

    The number of logical cores is the number of physical cores times the
    number of threads that can run on each core (Simultaneous Multithreading,
    SMT). If the number of logical cores cannot be determined, an exception is
    raised.
age
    Author
    ------
        tunprimus
    """
    import joblib
    import multiprocessing
    import os
    import psutil
    import re
    import subprocess

    # For Python 2.6+
    # Using multiprocessing module
    try:
        n_log_cores = multiprocessing.cpu_count()
        if n_log_cores > 0:
            return n_log_cores
    except (ImportError, NotImplementedError):
        pass
    # Using joblib module
    try:
        n_log_cores = joblib.cpu_count()
        if n_log_cores > 0:
            return n_log_cores
    except (ImportError, NotImplementedError):
        pass
    # Using psutil module
    try:
        n_log_cores = psutil.cpu_count()
        if n_log_cores > 0:
            return n_log_cores
    except (ImportError, AttributeError):
        pass
    # Using os module
    try:
        n_log_cores = os.cpu_count()
        if n_log_cores is None:
            raise NotImplementedError
        if n_log_cores > 0:
            return n_log_cores
    except:
        pass
    # Check proc process
    try:
        m = re.search(r"(?m)^Cpus_allowed:\s*(.*)$", open("/proc/self/status").read())
        if m:
            res = bin(int(m.group(1).replace(",", "")))
            if res > 0:
                n_log_cores = res
                return n_log_cores
    except IOError:
        pass
    # POSIX
    try:
        res = int(os.sysconf("SC_NPROCESSORS_ONLN"))
        if res > 0:
            n_log_cores = res
            return n_log_cores
    except (AttributeError, ValueError):
        pass
    # Windows
    try:
        res = int(os.environ["NUMBER_OF_PROCESSORS"])
        if res > 0:
            n_log_cores = res
            return n_log_cores
    except (AttributeError, ValueError):
        pass
    # Jython
    try:
        from java.lang import Runtime

        runtime = Runtime.getRuntime()
        res = runtime.availableProcessors()
        if res > 0:
            n_log_cores = res
            return n_log_cores
    except ImportError:
        pass
    # BSD
    try:
        sysctl = subprocess.Popen(["sysctl", "-n", "hw.ncpu"], stdout=subprocess.PIPE)
        sc_stdout = sysctl.communicate()[0]
        res = int(sc_stdout)
        if res > 0:
            n_log_cores = res
            return n_log_cores
    except (OSError, ValueError):
        pass
    # Linux
    try:
        with open("/proc/cpuinfo") as fin:
            res = fin.read().count("processor\t:")
            if res > 0:
                n_log_cores = res
                return n_log_cores
    except IOError:
        pass
    # Solaris
    try:
        pseudo_devices = os.listdir("/dev/pseudo")
        res = 0
        for pd in pseudo_devices:
            if re.match(r"^cpuid@[0-9]+$", pd):
                res += 1
        if res > 0:
            n_log_cores = res
            return n_log_cores
    except OSError:
        pass
    # Other UNIXes (heuristic)
    try:
        try:
            dmesg = open("/var/run/dmesg.boot").read()
        except IOError:
            dmesg_process = subprocess.Popen(["dmesg"], stdout=subprocess.PIPE)
            dmesg = dmesg_process.communicate()[0]
        res = 0
        while "\ncpu" + str(res) + ":" in dmesg:
            res += 1
        if res > 0:
            n_log_cores = res
            return n_log_cores
    except OSError:
        pass
    raise Exception("Cannot determine number of cores on this system.")


LOGICAL_CORES = cpu_logical_cores_count()


### Extended Print Report
def extended_print_report_object(obj, num_dp=4):
    """
    Prints out a nested dictionary (obj) in a more user-friendly manner.

    Parameters
    ----------
    obj : dict
        The dictionary to print
    num_dp : int, optional
        The number of decimal places to round any numerical values to. Defaults to 4.

    Returns
    -------
    None

    Author
    ------
        tunprimus
    """
    print("Full report history...\n")
    for key01, val01 in obj.items():
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


# Adapted from @georgerichardson -> https://gist.github.com/georgerichardson/db66b686b4369de9e7196a65df45fc37
def standardise_column_names(df, remove_punct=True):
    """
    Standardises column names in a pandas DataFrame. By default, removes punctuation, replaces spaces with underscores and removes trailing underscores.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to standardise
    remove_punct : bool, optional
        Whether to remove punctuation from column names. Defaults to True.

    Returns
    -------
    pandas.DataFrame
        DataFrame with standardised column names

    Example
    -------
    >>> df = pd.DataFrame({'Column With Spaces': [1,2,3,4,5],
                            'Column-With-Hyphens&Others/': [6,7,8,9,10],
                            'Too    Many Spaces': [11,12,13,14,15],
                            })
    >>> df = standardise_column_names(df)
    >>> print(df.columns)
    Index(['column_with_spaces',
            'column_with_hyphens_others',
            'too_many_spaces'], dtype='object')

    Author
    ------
        tunprimus
        Adapted from: @georgerichardson -> https://gist.github.com/georgerichardson/db66b686b4369de9e7196a65df45fc37
    """
    import re
    import string

    try:
        import fireducks.pandas as pd
    except ImportError:
        import pandas as pd

    # Handle CamelCase
    for i, col in enumerate(df.columns):
        matches = re.findall(re.compile("[a-z][A-Z]"), col)
        column = col
        for match in matches:
            column = column.replace(match, f"{match[0]}_{match[1]}")
            df = df.rename(columns={df.columns[i]: column})
    # Main cleaning
    translator = str.maketrans(string.punctuation, " " * len(string.punctuation))
    for c in df.columns:
        c_mod = c.lower()
        if remove_punct:
            c_mod = c_mod.translate(translator)
        c_mod = "_".join(c_mod.split(" "))
        if c_mod.endswith("_") or c_mod[-1] == "_":
            c_mod = c_mod[:-1]
        c_mod = re.sub(r"\_+", "_", c_mod)
        df.rename({c: c_mod}, inplace=True, axis=1)
    # Further cleaning of unusual languages
    df.columns = (
        df.columns.str.replace("\n", "_", regex=False)
        .str.replace("(", "_", regex=False)
        .str.replace(")", "_", regex=False)
        .str.replace("'", "_", regex=False)
        .str.replace('"', "_", regex=False)
        .str.replace(".", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace(r"[!?:;/]", "_", regex=True)
        .str.replace("+", "_plus_", regex=False)
        .str.replace("*", "_times_", regex=False)
        .str.replace("<", "_smaller", regex=False)
        .str.replace(">", "_larger_", regex=False)
        .str.replace("=", "_equal_", regex=False)
        .str.replace("ä", "ae", regex=False)
        .str.replace("ö", "oe", regex=False)
        .str.replace("ü", "ue", regex=False)
        .str.replace("ß", "ss", regex=False)
        .str.replace("%", "_pct_", regex=False)
        .str.replace("$", "_dollar_", regex=False)
        .str.replace("€", "_euro_", regex=False)
        .str.replace("@", "_at_", regex=False)
        .str.replace("#", "_hash_", regex=False)
        .str.replace("&", "_and_", regex=False)
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
        .str.lower()
    )
    return df


# ------------------------------------------------------------------#
# Function to generate a Pandas DataFrame from various data sources #
# ------------------------------------------------------------------#
def generate_df_from_data_source(
    data_source,
    from_aws=False,
    aws_access_key=None,
    aws_secret_key=None,
    aws_bucket_name=None,
    aws_file_key=None,
    from_googlesheet=False,
    path_to_googlesheet_cred=None,
    googlesheet_name=None,
    from_bigquery=False,
    path_to_bigquery_cred=None,
    bigquery_id=None,
    bigquery_dataset_id=None,
    bigquery_table_id=None,
):
    """
    Generate a Pandas DataFrame from various data sources.

    This function supports reading data from local files with different extensions,
    AWS S3, Google Sheets, and Google BigQuery. It automatically detects the
    file type based on the file extension and uses the appropriate pandas
    reader function to load the data into a DataFrame.

    Parameters
    ----------
    - data_source: str
        Path to the data source file.
    - from_aws: bool, optional
        If True, load data from AWS S3.
    - aws_access_key: str, optional
        AWS access key for S3.
    - aws_secret_key: str, optional
        AWS secret key for S3.
    - aws_bucket_name: str, optional
        S3 bucket name.
    - aws_file_key: str, optional
        S3 file key.
    - from_googlesheet: bool, optional
        If True, load data from Google Sheets.
    - path_to_googlesheet_cred: str, optional
        Path to Google Sheets credentials.
    - googlesheet_name: str, optional
        Name of the Google Sheet.
    - from_bigquery: bool, optional
        If True, load data from Google BigQuery.
    - path_to_bigquery_cred: str, optional
        Path to BigQuery credentials.
    - bigquery_id: str, optional
        BigQuery project ID.
    - bigquery_dataset_id: str, optional
        BigQuery dataset ID.
    - bigquery_table_id: str, optional
        BigQuery table ID.

    Returns
    --------
    - pd.DataFrame
        A DataFrame containing the loaded data.

    Raises
    -------
    - ValueError
        If the file extension is unsupported.

    Notes
    -----
    - The function uses different pandas functions to read different file types.
    - AWS, Google Sheets, and BigQuery connections require appropriate credentials.

    Author
    ------
        tunprimus
    """
    import boto3
    import gspread
    import numpy as np
    import pyarrow.parquet as pq
    import rpy2.robjects as ro
    import yaml
    import zipfile

    try:
        import fireducks.pandas as pd
    except ImportError:
        import pandas as pd
    from fastavro import reader
    from google.cloud import bigquery
    from io import StringIO, BytesIO
    from os.path import expanduser, realpath
    from os.path import splitext as splitext
    from rpy2.robjects import pandas2ri
    from scipy.io import arff
    from scipy.io import loadmat
    from sqlalchemy import create_engine, text

    # Monkey patching NumPy >= 1.24 in order to successfully import model from sklearn and other libraries
    np.float = np.float64
    np.int = np.int_
    np.object = np.object_
    np.bool = np.bool_

    pd.set_option("mode.copy_on_write", True)

    # Get file extension type
    real_path_to_data_source = realpath(expanduser(data_source))
    file_path_name, ext_buffer = splitext(real_path_to_data_source)
    ext = ext_buffer.lstrip(".")

    # Read data based on file extension
    if ext == "csv":
        return pd.read_csv(real_path_to_data_source)
    elif ext == "tsv":
        return pd.read_csv(real_path_to_data_source, sep="\t")
    elif ext == "json":
        return pd.read_json(real_path_to_data_source)
    elif ext == "xls" or ext == "xlsx":
        return pd.read_excel(real_path_to_data_source)
    elif ext == "txt":
        return pd.read_csv(real_path_to_data_source, sep="\t")
    # elif ext == "db" or ext == "sqlite" or ext == "sqlite3" or ext == "sql":
    #     query = "SELECT name FROM sqlite_master WHERE type = 'table';"
    #     cnx = create_engine(f"sqlite:///{real_path_to_data_source}").connect()
    #     return pd.read_sql(sql=text(query), con=cnx)
    elif ext == "html":
        return pd.read_html(real_path_to_data_source)
    elif ext == "h5":
        return pd.read_hdf(real_path_to_data_source)
    elif ext == "feather":
        return pd.read_feather(real_path_to_data_source)
    elif ext == "parquet":
        return pd.read_parquet(real_path_to_data_source)
    elif ext == "msgpack":
        return pd.read_msgpack(real_path_to_data_source)
    elif ext == "stata" or ext == "dta":
        return pd.read_stata(real_path_to_data_source)
    elif ext == "sas":
        return pd.read_sas(real_path_to_data_source)
    elif ext == "spss":
        return pd.read_spss(real_path_to_data_source)
    elif ext == "ods":
        return pd.read_ods(real_path_to_data_source)
    elif ext == "RData" or ext == "rda" or ext == "sav":
        pandas2ri.activate()
        r_file = real_path_to_data_source
        ro.r["load"](r_file)
        r_df = ro.r["get"](r_file)
        return pandas2ri.ri2py(r_df)
    elif ext == "jsonl":
        return pd.read_json(real_path_to_data_source, lines=True)
    elif ext == "orc":
        return pd.read_orc(real_path_to_data_source)
    elif ext == "xml":
        return pd.read_xml(StringIO(real_path_to_data_source))
    elif ext == "avro":
        with open(real_path_to_data_source, "rb") as f:
            avro_reader = reader(f)
            return pd.DataFrame(avro_reader)
    elif ext == "mat":
        return pd.DataFrame(loadmat(real_path_to_data_source))
    elif ext == "gz":
        with open(real_path_to_data_source, "rb") as f:
            return pd.read_csv(f, compression="gzip")
    elif ext == "pkl":
        with open(real_path_to_data_source, "rb") as f:
            return pd.read_pickle(f)
    elif ext == "zip":
        with zipfile.ZipFile(real_path_to_data_source, "r") as zip_ref:
            return pd.read_csv(zip_ref.open(zip_ref.namelist()[0]))
    elif ext == "parquet":
        return pq.read_table(real_path_to_data_source).to_pandas()
    elif ext == "arff":
        arff_file = arff.loadarff(real_path_to_data_source)
        return pd.DataFrame(arff_file)[0]
    elif ext == "yaml" or ext == "yml":
        with open(real_path_to_data_source, "r") as f:
            return pd.json_normalize(yaml.safe_load(f))
    elif from_aws:
        s3 = boto3.client(
            "s3", aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key
        )
        obj = s3.get_object(Bucket=aws_bucket_name, Key=aws_file_key)
        return pd.read_csv(BytesIO(obj["Body"].read()))
    elif from_googlesheet:
        gc = gspread.service_account(filename=realpath(path_to_googlesheet_cred))
        worksheet = gc.open(googlesheet_name)
        return pd.DataFrame(worksheet.get_worksheet(0).get_all_records())
    elif from_bigquery:
        client = bigquery.Client.from_service_account_json(
            realpath(path_to_bigquery_cred)
        )
        query = f"SELECT * FROM {bigquery_id}.{bigquery_dataset_id}.{bigquery_table_id}"
        try:
            return client.query(query).to_dataframe()
        except Exception as exc:
            print(exc)
            return pd.read_gbq(query, dialect="standard", project_id=bigquery_id)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def missing_value_overview(df, numdp=4, messages=True):
    """
    Provides an overview of missing values in a DataFrame.

    This function calculates and returns a table that shows the number of missing
    values, the percentage of missing values and the data type for each column in
    the provided DataFrame. It also prints a summary message if requested.

    Parameters
    ----------
    - df: pd.DataFrame
        The DataFrame for which to calculate missing values overview.
    - numdp: int, optional (default=4)
        Number of decimal places to round the percentage of missing values.
    - messages: bool, optional (default=True)
        If True, prints a message about the number of columns and columns with missing
        values.

    Returns
    -------
    - pd.DataFrame
        A DataFrame containing columns 'missing_values', 'pct_of_total_values', and
        'data_type', sorted by percentage of missing values in descending order.

    Author
    ------
        tunprimus
    """
    import pandas as pd

    miss_val = df.isnull().sum()
    df_length = len(df)
    miss_val_pct = round(((miss_val / df_length) * 100), numdp)
    data_types = df.dtypes

    miss_val_table_buffer = pd.concat([miss_val, miss_val_pct, data_types], axis=1)
    # Rename the columns
    miss_val_table = miss_val_table_buffer.rename(
        columns={0: "missing_values", 1: "pct_of_total_values", 2: "data_type"}
    )
    # Sort the table by percentage of missing descending
    try:
        miss_val_table = (
            miss_val_table[miss_val_table.iloc[:, 1] != 0]
            .sort_values("pct_of_total_values", ascending=False)
            .round(1)
        )
    except Exception:
        pass
    if messages:
        print(
            "Your selected dataframe has "
            + str(df.shape[1])
            + " columns.\n"
            + str(miss_val_table.shape[0])
            + " column(s) with missing values."
        )

    return miss_val_table


def missing_value_heatmap(df, figsize=(20, 12.4)):
    """
    Create a heatmap of missing values in a Pandas DataFrame.

    This function generates a heatmap representation of missing values in the input DataFrame `df`.
    The heatmap is a 2D representation of the matrix of missing values in the DataFrame, with NaN values
    represented as black cells and non-missing values represented as white cells. It is a useful tool
    for quickly visualising which columns in the DataFrame have missing values and which do not.

    Parameters
    ----------
    - df (pd.DataFrame): The DataFrame for which the heatmap of missing values is to be generated.
    - figsize (tuple): The size of the figure to be generated. Defaults to (20, 12.4).

    Author
    ------
        tunprimus
        Adapted from https://medium.com/@itk48/missing-data-imputation-with-chi-square-tests-mcar-mar-3278956387c8
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    plt.figure(figsize=figsize)
    plt.imshow(df.isnull(), aspect="auto", cmap="viridis", interpolation="nearest")
    plt.xticks(ticks=range(len(df.columns)), labels=df.columns, rotation=45)
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.title("Heatmap of Missing Values in Dataset")
    plt.show()


# ----------------------------------------------------------------------#
# Function to get quick summary of Pandas DataFrame #
# ----------------------------------------------------------------------#


def quick_data_summary(df, numdp=4, messages=False):
    """
    Generate a quick summary of a Pandas DataFrame.

    Author: tunprimus

    Parameters
    ----------
    - df : Pandas DataFrame
        The DataFrame to be summarised
    - numdp : int, default 4
        Number of decimal places to round to
    - messages : bool, default False
        If `True`, print messages indicating the summary of the DataFrame

    Returns
    -------
    summary : dict
        A dictionary containing the summary of the DataFrame.

    Author
    ------
        tunprimus
    """
    import numpy as np
    import pandas as pd

    summary = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "missing_values_pct": (
            round((df.isnull().sum() / df.shape[0] * 100), numdp)
        ).to_dict(),
        "description": df.describe(include="all").T.round(numdp),
    }
    if messages:
        for key01, val01 in summary.items():
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
    return summary


def column_summary(df, top_n=10, message=False):
    """
    Generates a summary DataFrame for each column in the input DataFrame.

    This function iterates over each column in the provided DataFrame, calculating
    and compiling statistics including the column's data type, the number of null
    and non-null values, the number of distinct values, and the counts of distinct
    values. For columns with more distinct values than a specified threshold, only
    the top N most frequent values are included in the summary.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to be summarised.
    top_n : int, optional (default=10)
        The number of top distinct values to include in the summary if a column
        has more distinct values than this threshold.
    message : bool, optional (default=False)
        If True, prints the resulting summary DataFrame.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the summary statistics for each column in the input
        DataFrame.

    Author
    ------
        tunprimus
    """
    import numpy as np
    import pandas as pd

    summary_list = []

    for col in df.columns:
        col_dtype = df[col].dtype
        num_of_nulls = df[col].isnull().sum()
        num_of_non_nulls = df[col].notnull().sum()
        num_of_distinct_val = df[col].nunique()

        if num_of_distinct_val <= top_n:
            distinct_val_counts = df[col].value_counts().to_dict()
        else:
            top_n_val_counts = df[col].value_counts().head(top_n).to_dict()
            distinct_val_counts = {
                key: val
                for key, val in sorted(
                    top_n_val_counts.items(), key=lambda item: item[1], reverse=True
                )
            }

        summary_list.append(
            {
                "col_name": col,
                "col_dtype": col_dtype,
                "num_of_nulls": num_of_nulls,
                "num_of_non_nulls": num_of_non_nulls,
                "num_of_distinct_val": num_of_distinct_val,
                "distinct_val_counts": distinct_val_counts,
            }
        )

    summary_df = pd.DataFrame(summary_list)
    if messages:
        print(summary_df)

    return summary_df


def column_summary_plus(df, top_n=10, messages=False):
    """
    Creates a summary of a given DataFrame, including the following information:
        - The column's data type
        - The number of distinct values
        - The minimum and maximum values
        - The median value of non-null values
        - The average value of non-null values
        - The average value of non-zero values
        - Whether null values are present
        - The number of null values
        - The number of non-null values
        - A dictionary where the keys are the top N distinct values and the values are the counts of each distinct value.

    Parameters
    ----------
    df : DataFrame
        The DataFrame to be summarised.
    top_n : int, default 10
        The number of distinct values to include in the summary.
    messages : bool, default False
        If True, prints a message about the start of the processing of each column and the final summary DataFrame.

    Returns
    -------
    DataFrame
        A DataFrame containing the summary of the given DataFrame.

    Author
    ------
        tunprimus
    """
    import numpy as np
    import pandas as pd

    result_df = pd.DataFrame(
        columns=[
            "col_name",
            "col_dtype",
            "num_of_distinct_val",
            "min_val",
            "max_val",
            "median_no_na",
            "average_no_na",
            "average_non_zero",
            "null_present",
            "num_of_nulls",
            "num_of_non_nulls",
            f"top_{top_n}_distinct_values",
        ]
    )

    for col in df.columns:
        if messages:
            print(f"Start to process column {col} with dtype {df[col].dtype}")
        col_dtype = df[col].dtype

        # Get distinct values and their counts
        val_counts = df[col].value_counts()
        distinct_values = val_counts.index.tolist()
        num_of_distinct_val = len(distinct_values)
        sorted_values = sorted(distinct_values)
        min_val = sorted_values[0] if sorted_values else None
        max_val = sorted_values[-1] if sorted_values else None

        # Get median value
        non_distinct_val_list = sorted(df[col].dropna().tolist())
        length_non_distinct_list = len(non_distinct_val_list)
        if len(non_distinct_val_list) == 0:
            median_no_na = None
        else:
            median_no_na = non_distinct_val_list[length_non_distinct_list // 2]

        # Get average value if it is a number
        if np.issubdtype(df[col].dtype, np.number):
            if len(non_distinct_val_list) > 0:
                avg = sum(non_distinct_val_list) / len(non_distinct_val_list)
                non_zero_val_list = [val for val in non_distinct_val_list if val != 0]
                avg_non_zero = sum(non_zero_val_list) / length_non_distinct_list
            else:
                avg = None
                avg_non_zero = None
        else:
            avg = None
            avg_non_zero = None

        null_present = 1 if df[col].isnull().any() else 0

        num_of_nulls = df[col].isnull().sum()
        num_of_non_nulls = df[col].notnull().sum()

        # Distinct_values only take top N distinct values count
        top_n_d_val = df[col].value_counts().head(top_n).index.tolist()
        top_n_cnt = df[col].value_counts().head(top_n).tolist()
        top_n_d_v_dict = dict(zip(top_n_d_val, top_n_cnt))

        result_df = result_df.append(
            {
                "col_name": col,
                "col_dtype": col_dtype,
                "num_of_distinct_val": num_of_distinct_val,
                "min_val": min_val,
                "max_val": max_val,
                "median_no_na": median_no_na,
                "average_no_na": avg,
                "average_non_zero": avg_non_zero,
                "null_present": null_present,
                "num_of_nulls": num_of_nulls,
                "num_of_non_nulls": num_of_non_nulls,
                f"top_{top_n}_distinct_values": top_n_d_v_dict,
            },
            ignore_index=True,
        )

    if messages:
        print(result_df)

    return result_df


# ----------------------------------------------------------------------#
# Function to get univariate statistics and plots from Pandas DataFrame #
# ----------------------------------------------------------------------#

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


def univariate_stats(df):
    """
    Generate descriptive statistics and visualisations for each feature in a DataFrame.

    Author: tunprimus

    This function computes and returns a DataFrame containing a variety of univariate
    statistics for each feature (column) in the input DataFrame `df`. It calculates
    metrics such as the data type, count of non-missing values, number of missing values,
    number of unique values, and mode for all features. For numerical features, it
    additionally calculates minimum, lower boundary of normal (2.5 percentile), first quartile,
    median, third quartile, upper boundary of normal (97.5 percentile), maximum, mean,
    standard deviation, skewness, and kurtosis. It also creates a histogram for
    numerical features and a count plot for categorical features.

    Parameters
    ----------
    - df (pd.DataFrame): The DataFrame for which univariate statistics are to be computed.

    Returns
    -------
    - pd.DataFrame: A DataFrame where each row corresponds to a feature from the input
      DataFrame and columns contain the calculated statistics.

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

    output_df = pd.DataFrame(
        columns=[
            "feature",
            "type",
            "count",
            "missing",
            "unique",
            "mode",
            "min",
            "lbn_2_5pct",
            "q1",
            "median",
            "q3",
            "ubn_97_5pct",
            "max",
            "mean",
            "std",
            "skew",
            "kurt",
        ]
    )
    output_df.set_index("feature", inplace=True)
    for col in df.columns:
        # Calculate metrics that apply to all columns dtypes
        dtype = df[col].dtype
        count = df[col].count()
        missing = df[col].isna().sum()
        unique = df[col].nunique()
        mode = df[col].mode()[0]
        if pd.api.types.is_numeric_dtype(df[col]):
            # Calculate metrics that apply only to numerical features
            min_ = df[col].min()
            lbn_2_5pct = df[col].quantile(0.025)
            q1 = df[col].quantile(0.25)
            median = df[col].median()
            q3 = df[col].quantile(0.75)
            ubn_97_5pct = df[col].quantile(0.975)
            max_ = df[col].max()
            mean = df[col].mean()
            std = df[col].std()
            skew = df[col].skew()
            kurt = df[col].kurt()
            output_df.loc[col] = [
                dtype,
                count,
                missing,
                unique,
                mode,
                min_,
                lbn_2_5pct,
                q1,
                median,
                q3,
                ubn_97_5pct,
                max_,
                mean,
                std,
                skew,
                kurt,
            ]
            sns.histplot(data=df, x=col)
            sns.swarmplot(data=df, x=col)
        else:
            output_df.loc[col] = [
                dtype,
                count,
                missing,
                unique,
                mode,
                "--",
                "--",
                "--",
                "--",
                "--",
                "--",
                "--",
                "--",
                "--",
                "--",
                "--",
            ]
            sns.countplot(data=df, x=col)
        plt.show()
    try:
        return output_df.sort_values(by=["missing", "unique", "skew"], ascending=False)
    except Exception:
        return output_df


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
    >>> rng = np.random.default_rng(42)
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
    >>> rng = np.random.default_rng(42)
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
    >>> rng = np.random.default_rng(42)
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
        >>> rng = np.random.default_rng(42)
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


# ----------------------------------------------------------------------#
# Functions to get multivariate statistics and plots from Pandas DataFrame #
# ----------------------------------------------------------------------#

def plot_correlation_heatmap_with_matshow(df):
    """
    Plot the correlation heatmap of the given DataFrame using plt.matshow.

    Parameters
    ----------
    - df : pandas.core.frame.DataFrame
        The DataFrame to be plotted.

    Returns
    -------
    None

        Example
        -------
        >>> # Sample data
        >>> rng = np.random.default_rng(42)
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
        >>> plot_correlation_heatmap_with_matshow(df)

    Notes
    -----
    This function will show a heatmap of the correlation between columns in the given DataFrame.

    Author
    ------
        tunprimus
    """
    import matplotlib.pyplot as plt
    import numpy as np

    try:
        import fireducks.pandas as pd
    except ImportError:
        import pandas as pd

    plt.matshow(df.corr())
    plt.yticks(np.arange(df.shape[1]), df.columns)
    plt.xticks(np.arange(df.shape[1]), df.columns, rotation=90)
    plt.colorbar()
    plt.show()


def plot_correlation_heatmap(df, messages=True):
    """
    Plot the correlation heatmap of the given DataFrame using style.background_gradient.

    Parameters
    ----------
    - df : pandas.core.frame.DataFrame
        The DataFrame to be plotted.
    - messages: bool, default True
        Whether to print out the correlation matrix

    Returns
    -------
    pandas.io.formats.style.Styler
        The styled DataFrame containing the correlation matrix.

        Example
        -------
        >>> # Sample data
        >>> rng = np.random.default_rng(42)
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
        >>> plot_correlation_heatmap(df)

    Notes
    -----
    This function will show a heatmap of the correlation between columns in the given DataFrame.

    Author
    ------
        tunprimus
    """
    import numpy as np

    try:
        import fireducks.pandas as pd
    except ImportError:
        import pandas as pd

    corr = df.corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    corr[mask] = np.nan

    upper_triangle = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
    max_corr = upper_triangle.max().max()

    if messages:
        print(f"Maximum pairwise correlation: {max_corr:.2f}")
        print(corr.round(2))

    return (
        corr.style.background_gradient(cmap="coolwarm", axis=None, vmin=-1, vmax=1)
        .highlight_null(color="#F1F1F1")
        .format(precision=2)
    )


# ----------------------------------------------------------------------#
# Functions to automate data cleaning from Pandas DataFrame #
# ----------------------------------------------------------------------#

## Basic Data Wrangling
##*********************##

### Eliminate Empty Columns, Columns with All Unique Values and Columns with Single Values

def check_uniqueness(df, top_n=10):
    """
    Check uniqueness of each column in a DataFrame.

    Parameters
    ----------
    - df : pandas.core.frame.DataFrame
        The DataFrame to be checked.
    - top_n: : int
                The number of top items to keep. Defaults to 10

    Returns
    -------
    pandas.core.frame.DataFrame
        A DataFrame with columns "column", "num_unique", and "counts".
        "column" is the name of the column in the original DataFrame.
        "num_unique" is the number of unique values in that column.
        "counts" is a numpy array of the counts of each unique value in that column,
        sorted by the unique value itself.
        "top_{top_n}_unique" is the top N unique values.

        Example
        -------
        >>> # Sample data
        >>> rng = np.random.default_rng(42)
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
        >>> check_uniqueness(df)

        Author
    ------
        tunprimus
    """
    try:
        import fireducks.pandas as pd
    except ImportError:
        import pandas as pd

    df_buffer = pd.DataFrame(
        columns=["column", "num_unique", "counts", f"top_{top_n}_unique"]
    )
    for col in df.columns:
        df_buffer.loc[col] = [
            col,
            df[col].nunique(),
            df[col].value_counts().sort_index().values,
            ", ".join(
                (df[col].value_counts().sort_index()[:top_n].index.values).astype(str)
            ),
        ]
    return df_buffer


def basic_wrangling(
    df,
    features=[],
    drop_duplicates=False,
    drop_duplicates_subset_name=None,
    missing_threshold=0.95,
    unique_threshold=0.95,
    messages=True,
):
    """
    Perform basic data wrangling on a DataFrame, including renaming columns,
    dropping duplicates, and removing columns with excessive missing or unique values.

    Parameters
    ----------
    - df : pandas DataFrame
        The DataFrame to be wrangled.
    - features : list of str, optional
        List of column names to focus on. If empty, all columns are considered.
    - drop_duplicates : bool, default False
        Whether to drop duplicate rows.
    - drop_duplicates_subset_name : str, optional
        Column name to consider when dropping duplicates. If None, all columns are considered.
    - missing_threshold : float, default 0.95
        Threshold for dropping columns with a high proportion of missing values.
    - unique_threshold : float, default 0.95
        Threshold for dropping columns with a high proportion of unique values.
    - messages : bool, default True
        If True, print messages about the operations performed.

    Returns
    -------
    df : pandas DataFrame
        The wrangled DataFrame.

        Example
        -------
        >>> # Sample data
        >>> rng = np.random.default_rng(42)
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
        >>>
        >>> data_dict = {"age": age, "sex": sex, "bmi": bmi, "children": children, "smoker": smoker, "region": region, "monthly_salary": monthly_salary, "cust_class": cust_class, "satisfaction": satisfaction, "charges": charges}
        >>> df = pd.DataFrame(data_dict)
        >>>
        >>> basic_wrangling(df)

        Author
    ------
        tunprimus
    """
    try:
        import fireducks.pandas as pd
    except ImportError:
        import pandas as pd

    try:
        df = standardise_column_names(df)
    except Exception:
        df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

    if drop_duplicates:
        if drop_duplicates_subset_name is None:
            df.drop_duplicates(inplace=True)
            if messages:
                print("Dropped duplicate rows across all columns.")
        else:
            df = df.drop_duplicates(subset=[drop_duplicates_subset_name])
            if messages:
                print("Dropped duplicate rows from desired column(s).")

    all_cols = df.columns
    if len(features) == 0:
        features = all_cols
    for feat in features:
        if feat in all_cols:
            missing = df[feat].isna().sum()
            unique = df[feat].nunique()
            rows = df.shape[0]
            if (missing / rows) >= missing_threshold:
                if messages:
                    print(
                        f"Too much missing ({missing} out of {rows} rows, {round(((missing / rows) * 100), 1)}%) for {feat}"
                    )
                df.drop(columns=[feat], inplace=True)
            elif (unique / rows) >= unique_threshold:
                if df[feat].dtype in ["int64", "object"]:
                    if messages:
                        print(
                            f"Too many unique values ({unique} out of {rows} rows, {round(((unique / rows) * 100), 1)}%) for {feat}"
                        )
                    df.drop(columns=[feat], inplace=True)
            elif unique == 1:
                if messages:
                    print(f"Only one value ({df[feat].unique()[0]}) for {feat}")
                df.drop(columns=[feat], inplace=True)
    else:
        if messages:
            print(
                f"The feature '{feat}' does not exist as spelled in the DataFrame provided."
            )
    return df


### Date and Time Management


def can_convert_dataframe_to_datetime(
    df, col_list=[], return_result=True, messages=False
):
    """
    Check if columns in a DataFrame can be converted to datetime format.

    Parameters
    ----------
    - df : pandas DataFrame
        DataFrame to check
    - col_list : list, default []
        List of columns to check. If empty, all columns with dtype of object will be checked.
    - return_result : bool, default True
        If `True`, return a dictionary with column names as keys and boolean values indicating
        whether each column can be converted to datetime format. If `False`, return nothing and print
        messages.
    messages : bool, default False
        If `True`, print messages indicating whether each column can be converted to datetime format.

    Returns
    -------
    result : dict or list
        Dictionary with column names as keys and boolean values indicating
        whether each column can be converted to datetime format, or list of boolean values.

    Author
    ------
        tunprimus
    """
    import numpy as np
    import pandas as pd
    from pandas.api.types import is_datetime64_any_dtype as is_datetime

    # Change message flag based on whether or not to return result
    if not return_result:
        messages = True

    length_col_list = len(col_list)
    # Define type of return object
    if length_col_list == 1:
        result = []
    else:
        result = {}
    # Determine how many columns to use in DataFrame
    if length_col_list == 0:
        columns_to_check = df.columns
    else:
        columns_to_check = df.columns[df.columns.isin(col_list)]
    # Check only columns with dtype of object
    for col in columns_to_check:
        # result = []
        if df[col].dtype == "object":
            can_be_datetime = False
            try:
                df_flt_tmp = df[col].astype(np.float64)
                can_be_datetime = False
            except:
                try:
                    df_dt_tmp = pd.to_datetime(df[col])
                    can_be_datetime = is_datetime(df_dt_tmp)
                except:
                    pass
            if messages:
                print(f"Can convert {col} to datetime? {can_be_datetime}")
            # Choose return data structure
            if length_col_list == 1:
                result.append(can_be_datetime)
            else:
                result[col] = can_be_datetime
    if return_result:
        return result


def batch_convert_to_datetime(
    df,
    split_datetime=True,
    add_hr_min_sec=False,
    days_to_today=False,
    drop_date=True,
    messages=True,
):
    """
    Convert object columns in a DataFrame to datetime format if possible.

    Parameters
    ----------
    - df : pandas DataFrame
        DataFrame to convert
    - split_datetime : bool, default True
        If `True`, split datetime columns into separate columns for year, month, day and weekday.
    - add_hr_min_sec : bool, default False
        If `True`, add separate columns for hour, minute and second.
    - days_to_today : bool, default False
        If `True`, add a column with the number of days to today.
    - drop_date : bool, default True
        If `True`, drop the original column after conversion.
    - messages : bool, default True
        If `True`, print messages indicating which columns were converted.

    Returns
    -------
    df : pandas DataFrame
        DataFrame with converted datetime columns.

    Author
    ------
        tunprimus
    """
    import numpy as np
    import pandas as pd
    import sys
    from pandas.api.types import is_datetime64_any_dtype as is_datetime

    for col in df.columns[df.dtypes == "object"]:
        try:
            df_dt_tmp = pd.to_datetime(df[col])
            try:
                df_flt_tmp = df[col].astype(np.float64)
                if messages:
                    print(
                        f"Warning, NOT converting column '{col}', because it is ALSO convertible to float64.",
                        file=sys.stderr,
                    )
            except:
                df[col] = df_dt_tmp
                if split_datetime:
                    df[f"{col}_datetime"] = df[col]
                    df[f"{col}_year"] = df[col].dt.year
                    df[f"{col}_month"] = df[col].dt.month
                    df[f"{col}_day"] = df[col].dt.day
                    df[f"{col}_weekday"] = df[col].dt.day_name()
                if add_hr_min_sec:
                    df[f"{col}_hour"] = df[col].dt.hour
                    df[f"{col}_minute"] = df[col].dt.minute
                    df[f"{col}_second"] = df[col].dt.second
                if days_to_today:
                    df[f"{col}_days_to_today"] = (
                        pd.to_datetime("now") - df[col]
                    ).dt.days
                if drop_date:
                    df.drop(columns=[col], inplace=True)
                if messages:
                    print(
                        f"FYI, converted column '{col}' to datetime.", file=sys.stderr
                    )
                    print(f"Is '{df[col]}' now datetime? {is_datetime(df[col])}")
        # Cannot convert some elements of the column to datetime...
        except:
            pass
    return df


def parse_column_as_date(
    df, features=[], days_to_today=False, drop_date=True, messages=True
):
    """
    Parse specified date features in a DataFrame and extract related information.

    Parameters
    ----------
    - df : pandas DataFrame
        DataFrame containing the features to be parsed.
    - features : list, default []
        List of column names to parse as date features. If empty, no columns are parsed.
    - days_to_today : bool, default False
        If `True`, calculate the number of days from each date to today's date.
    - drop_date : bool, default True
        If `True`, drop the original date columns after parsing.
    - messages : bool, default True
        If `True`, print messages indicating parsing status for each feature.

    Returns
    -------
    df : pandas DataFrame
        DataFrame with parsed date features and additional extracted information.

    Author
    ------
        tunprimus
    """
    import pandas as pd
    from datetime import datetime as pydt

    all_cols = df.columns
    for feat in features:
        if feat in all_cols:
            try:
                df[feat] = pd.to_datetime(df[feat])
                df[f"{feat}_datetime"] = df[feat]
                df[f"{feat}_year"] = df[feat].dt.year
                df[f"{feat}_month"] = df[feat].dt.month
                df[f"{feat}_day"] = df[feat].dt.day
                df[f"{feat}_weekday"] = df[feat].dt.day_name()
                if days_to_today:
                    df[f"{feat}_days_until_today"] = (pydt.today() - df[feat]).dt.days
                if drop_date:
                    df.drop(columns=[feat], inplace=True)
            except:
                if messages:
                    print(f"Could not convert feature '{feat}' to datetime.")
        else:
            if messages:
                print(
                    f"Feature '{feat}' does not exist as spelled in the DataFrame provided."
                )
    return df


### Bin Low Count Groups Values


def bin_categories(df, features=[], cutoff=0.05, replace_with="Other", messages=True):
    """
    Bins low count groups values into one category

    Parameters
    ----------
    - df : pandas DataFrame
        DataFrame to clean
    - features : list of str, default []
        Columns to consider for binning
    - cutoff : float, default 0.05
        Proportion of total samples to define low count groups
    - replace_with : str, default 'Other'
        Value to replace low count groups with
    - messages : bool, default True
        If `True`, print messages indicating which columns were cleaned

    Returns
    -------
    df : pandas DataFrame
        DataFrame with low count groups binned

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
        >>> bin_categories(df, features=["satisfaction"], cutoff=0.025)

        Author
    ------
        tunprimus
    """
    try:
        import fireducks.pandas as pd
    except ImportError:
        import pandas as pd

    for feat in features:
        if feat in df.columns:
            if not pd.api.types.is_numeric_dtype(df[feat]):
                other_list = (
                    df[feat]
                    .value_counts()[(df[feat].value_counts() / df.shape[0]) < cutoff]
                    .index
                )
                df.loc[df[feat].isin(other_list), feat] = replace_with
        else:
            if messages:
                print(
                    f"The feature '{feat}' does not exist in the DataFrame provided. No binning performed."
                )
    return df


## Outliers
##*********************##

### Traditional One-at-a-time Methods


def clean_outlier_per_column(
    df,
    features=[],
    skew_threshold=1,
    tukey_lo_val=0.25,
    tukey_hi_val=0.75,
    iqr_k=1.5,
    emp_rule_val=3.0,
    handle_outliers="remove",
    num_dp=4,
    messages=True,
):
    """
    Clean outliers from a column in a DataFrame

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame to clean
    features : list of str, default []
        Columns to consider for outlier cleaning
    skew_threshold : float, default 1
        Threshold to determine if a column is skewed
    tukey_lo_val : float, default 0.25
        Lower bound for Tukey boxplot rule
    tukey_hi_val : float, default 0.75
        Upper bound for Tukey boxplot rule
    iqr_k : float, default 1.5
        Multiplier for IQR for Tukey boxplot rule
    emp_rule_val : float, default 3.0
        Multiplier for standard deviation for Empirical rule
    handle_outliers : str, default 'remove'
        How to handle outliers. Options are 'remove', 'replace', 'impute', 'null'
    num_dp : int, default 4
        Number of decimal places to round to
    messages : bool, default True
        If `True`, print messages indicating which columns were cleaned

    Returns
    -------
    df : pandas DataFrame
        DataFrame with outliers cleaned

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
        >>> clean_outlier_per_column(df, features=df.columns)

        Author
    ------
        tunprimus
    """
    import numpy as np

    try:
        import fireducks.pandas as pd
    except ImportError:
        import pandas as pd
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer

    # Monkey patching NumPy >= 1.24 in order to successfully import model from sklearn and other libraries
    np.float = np.float64
    np.int = np.int_
    np.object = np.object_
    np.bool = np.bool_

    pd.set_option("mode.copy_on_write", True)

    for feat in features:
        if feat in df.columns:
            if pd.api.types.is_numeric_dtype(df[feat]):
                if len(df[feat].unique()) != 1:
                    if not all(df[feat].value_counts().index.isin([0, 1])):
                        skew = df[feat].skew()
                        # Tukey boxplot rule
                        if abs(skew) > skew_threshold:
                            q1 = df[feat].quantile(tukey_lo_val)
                            q3 = df[feat].quantile(tukey_hi_val)
                            iqr = q3 - q1
                            lo_bound = q1 - iqr_k * iqr
                            hi_bound = q3 + iqr_k * iqr
                        # Empirical rule
                        else:
                            lo_bound = df[feat].mean() - (emp_rule_val * df[feat].std())
                            hi_bound = df[feat].mean() + (emp_rule_val * df[feat].std())
                        # Get the number of outlier data points
                        min_count = df.loc[df[feat] < lo_bound, feat].shape[0]
                        max_count = df.loc[df[feat] > hi_bound, feat].shape[0]
                        if (min_count > 0) or (max_count > 0):
                            # Remove rows with the outliers
                            if handle_outliers == "remove":
                                df = df[(df[feat] >= lo_bound) & (df[feat] <= hi_bound)]
                            # Replace outliers with min-max cutoff
                            elif handle_outliers == "replace":
                                df.loc[df[feat] < lo_bound, feat] = lo_bound
                                df.loc[df[feat] > hi_bound, feat] = hi_bound
                            # Impute with linear regression after deleting
                            elif handle_outliers == "impute":
                                df.loc[df[feat] < lo_bound, feat] = np.nan
                                df.loc[df[feat] > hi_bound, feat] = np.nan
                                imputer = IterativeImputer(max_iter=10)
                                df_temp = df.copy()
                                df_temp = bin_categories(
                                    df_temp, features=df_temp.columns, messages=False
                                )
                                df_temp = basic_wrangling(
                                    df_temp, features=df_temp.columns, messages=False
                                )
                                df_temp = pd.get_dummies(df_temp, drop_first=True)
                                df_temp = pd.DataFrame(
                                    imputer.fit_transform(df_temp),
                                    columns=df_temp.columns,
                                    index=df_temp.index,
                                    dtype="float",
                                )
                                df_temp.columns = df_temp.columns.get_level_values(0)
                                df_temp.index = df_temp.index.astype("int64")
                                # Save only the column from df_temp being iterated on
                                df[feat] = df_temp[feat]
                            # Replace with null
                            elif handle_outliers == "null":
                                df.loc[df[feat] < lo_bound, feat] = np.nan
                                df.loc[df[feat] > hi_bound, feat] = np.nan
                        if messages:
                            print(
                                f"Feature '{feat}' has {min_count} value(s) below the lower bound ({round(lo_bound, num_dp)}) and {max_count} value(s) above the upper bound ({round(hi_bound, num_dp)})."
                            )
                    else:
                        if messages:
                            print(
                                f"Feature '{feat}' is dummy coded (0, 1) and was ignored."
                            )
                else:
                    if messages:
                        print(
                            f"Feature '{feat}' has only one unique value ({df[feat].unique()[0]})."
                        )
            else:
                if messages:
                    print(f"Feature '{feat}' is categorical and was ignored.")
        else:
            if messages:
                print(
                    f"Feature '{feat}' does not exist in the DataFrame provided. No outlier removal performed."
                )
    return df


### Newer All-at-once Methods Based on Clustering


def clean_outlier_by_all_columns(
    df,
    drop_proportion=0.013,
    distance_method="manhattan",
    min_samples=5,
    num_dp=4,
    num_cores_for_dbscan=-1,
    messages=True,
):
    """
    Clean outliers from a DataFrame based on a range of eps values.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame to clean
    drop_proportion : float, default 0.013
        Proportion of total samples to define outliers
    distance_method : str, default "manhattan"
        Distance method to use in DBSCAN
    min_samples : int, default 5
        Minimum samples to form a dense region
    num_dp : int, default 4
        Number of decimal places to round to
    num_cores_for_dbscan : int, default LOGICAL_CORES-2 if LOGICAL_CORES > 3 else 1
        Number of cores to use in DBSCAN
    messages : bool, default True
        If `True`, print messages indicating which columns were cleaned

    Returns
    -------
    df : pandas DataFrame
        DataFrame with outliers cleaned

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
        >>> df_without_outliers = clean_outlier_by_all_columns(df)

        Author
    ------
        tunprimus
    """
    import joblib
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import time

    try:
        import fireducks.pandas as pd
    except ImportError:
        import pandas as pd
    from sklearn import preprocessing
    from sklearn.cluster import DBSCAN

    # Monkey patching NumPy >= 1.24 in order to successfully import model from sklearn and other libraries
    np.float = np.float64
    np.int = np.int_
    np.object = np.object_
    np.bool = np.bool_

    pd.set_option("mode.copy_on_write", True)

    # Clean the DataFrame first
    num_cols_with_missing_values = df.shape[1] - df.dropna(axis="columns").shape[1]
    df.dropna(axis="columns", inplace=True)
    if messages:
        print(f"{num_cols_with_missing_values} column(s) with missing values dropped.")
    num_rows_with_missing_values = df.shape[0] - df.dropna(axis="columns").shape[0]
    df.dropna(inplace=True)
    if messages:
        print(f"{num_rows_with_missing_values} row(s) with missing values dropped.")
    # Handle basic wrangling, binning and outliers
    df_temp = df.copy()
    df_temp = bin_categories(df_temp, features=df_temp.columns, messages=False)
    df_temp = basic_wrangling(df_temp, features=df_temp.columns, messages=False)
    df_temp = pd.get_dummies(df_temp, drop_first=True)
    # Normalise the data
    df_temp = pd.DataFrame(
        preprocessing.MinMaxScaler().fit_transform(df_temp),
        columns=df_temp.columns,
        index=df_temp.index,
    )
    # Calculate outliers based on a range of eps values
    outliers_per_eps = []
    outliers_per_eps_history = {}
    outliers = df_temp.shape[0]
    eps_loop = 0
    counter = 0
    row_count = df_temp.shape[0]
    if row_count < 500:
        INCREMENT_VAL = 0.010
    elif row_count < 1000:
        INCREMENT_VAL = 0.025
    elif row_count < 2000:
        INCREMENT_VAL = 0.050
    elif row_count < 10000:
        INCREMENT_VAL = 0.075
    elif row_count < 25000:
        INCREMENT_VAL = 0.100
    elif row_count < 50000:
        INCREMENT_VAL = 0.200
    elif row_count < 100000:
        INCREMENT_VAL = 0.250
    elif row_count < 250000:
        INCREMENT_VAL = 0.350
    else:
        INCREMENT_VAL = 0.500

    db_scan_time_start = time.time_ns()

    while outliers > 0:
        loop_start_time = time.time_ns()
        eps_loop += INCREMENT_VAL
        db_loop = DBSCAN(
            eps=eps_loop,
            metric=distance_method,
            min_samples=min_samples,
            n_jobs=num_cores_for_dbscan,
        ).fit(df_temp)
        outliers = np.count_nonzero(db_loop.labels_ == -1)
        outliers_per_eps.append(outliers)
        outliers_percent = (outliers / row_count) * 100
        outliers_per_eps_history[f"{counter}_trial"] = {}
        outliers_per_eps_history[f"{counter}_trial"]["eps_val"] = round(
            eps_loop, num_dp
        )
        outliers_per_eps_history[f"{counter}_trial"]["outliers"] = outliers
        outliers_per_eps_history[f"{counter}_trial"]["outliers_percent"] = round(
            outliers_percent, num_dp
        )
        loop_end_time = time.time_ns()
        loop_time_diff_ns = loop_end_time - loop_start_time
        loop_time_diff_s = (loop_end_time - loop_start_time) / 1000000000
        outliers_per_eps_history[f"{counter}_trial"][
            "loop_duration_ns"
        ] = loop_time_diff_ns
        outliers_per_eps_history[f"{counter}_trial"][
            "loop_duration_s"
        ] = loop_time_diff_s
        counter += 1
        if messages:
            print(
                f"eps = {round(eps_loop, num_dp)}, (outliers: {outliers}, percent: {round(outliers_percent, num_dp)}% in {round(loop_time_diff_s, num_dp)}s)"
            )

    to_drop = min(
        outliers_per_eps,
        key=lambda x: abs(x - round((drop_proportion * row_count), num_dp)),
    )

    # Find the optimal eps value
    eps = (outliers_per_eps.index(to_drop) + 1) * INCREMENT_VAL
    outliers_per_eps_history["optimal_eps"] = eps
    db_scan_time_end = time.time_ns()
    db_scan_time_diff_s = (db_scan_time_end - db_scan_time_start) / 1000000000
    outliers_per_eps_history["db_scan_duration_s"] = db_scan_time_diff_s
    outliers_per_eps_history["distance_metric_used"] = distance_method
    outliers_per_eps_history["min_samples_used"] = min_samples
    outliers_per_eps_history["drop_proportion_used"] = drop_proportion
    outliers_per_eps_history["timestamp"] = pd.Timestamp.now()
    if messages:
        print(f"Optimal eps value: {round(eps, num_dp)}")

        print(f"\nHistory:")
        extended_print_report_object(outliers_per_eps_history)

    db = DBSCAN(
        eps=eps,
        metric=distance_method,
        min_samples=min_samples,
        n_jobs=num_cores_for_dbscan,
    ).fit(df_temp)

    df["outlier"] = db.labels_
    if messages:
        print(
            f"{df[df['outlier'] == -1].shape[0]} row(s) of outliers found for removal."
        )
        sns.lineplot(x=range(1, len(outliers_per_eps) + 1), y=outliers_per_eps)
        sns.scatterplot(x=[eps / INCREMENT_VAL], y=[to_drop])
        plt.xlabel(f"eps (divided by {INCREMENT_VAL})")
        plt.ylabel("Number of outliers")
        plt.show()

    # Drop rows that are outliers
    df = df[df["outlier"] != -1]

    return df


## Skewness
##*********************##

def skew_correct(df, feature, max_power=103, messages=True):
    """
    Corrects the skew of a given feature in a DataFrame by applying a transformation to achieve normality.

    Parameters
    ----------
    - df : pandas DataFrame
        DataFrame containing the feature to correct
    - feature : str
        Name of the feature to correct
    - max_power : int, optional
        Maximum power to use for transformation. Default: 103
    - messages : bool, optional
        If `True`, print messages about the transformation. Default: True

    Returns
    -------
    df : pandas DataFrame
        DataFrame with the corrected feature

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
        >>> skew_correct(df, "charges")

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
    from os.path import realpath as realpath

    # Monkey patching NumPy >= 1.24 in order to successfully import model from sklearn and other libraries
    np.float = np.float64
    np.int = np.int_
    np.object = np.object_
    np.bool = np.bool_

    pd.set_option("mode.copy_on_write", True)
    GOLDEN_RATIO = 1.618033989
    FIG_WIDTH = 20
    FIG_HEIGHT = FIG_WIDTH / GOLDEN_RATIO
    FIG_SIZE = (FIG_WIDTH, FIG_HEIGHT)
    FIG_DPI = 72

    # rcParams for Plotting
    plt.rcParams["figure.figsize"] = FIG_SIZE
    plt.rcParams["figure.dpi"] = FIG_DPI

    # Check to use only numerical features
    if not pd.api.types.is_numeric_dtype(df[feature]):
        if messages:
            print(
                f"The feature '{feature}' is not numerical. No transformation performed."
            )
        return df

    # Clean out missing data
    df = basic_wrangling(df, messages=False)
    if messages:
        print(
            f"{df.shape[0] - df.dropna().shape[0]} row(s) with missing values dropped."
        )
    df.dropna(inplace=True)

    # In case the dataset is too big, use a subsample
    df_temp = df.copy()
    if df_temp.memory_usage(deep=True).sum() > 1_000_000:
        df_temp = df.sample(frac=round((5000 / df_temp.shape[0]), 2))

    # Identify the proper transformation exponent
    exp_val = 1
    skew = df_temp[feature].skew()
    if messages:
        print(f"Starting skew: {round(skew, 5)}")

    while (round(skew, 2) != 0) and (exp_val <= max_power):
        exp_val += 0.01
        if skew > 0:
            skew = np.power(df_temp[feature], 1 / exp_val).skew()
        else:
            skew = np.power(df_temp[feature], exp_val).skew()
    if messages:
        print(f"Final skew: {round(skew, 5)} (using exponent: {round(exp_val, 5)})")

    # Make the transformed version of the feature in the df DataFrame
    if (skew > -0.1) and (skew < 0.1):
        if skew > 0:
            corrected = np.power(df[feature], 1 / round(exp_val, 3))
            name = f"{feature}_1/{round(exp_val, 3)}"
        else:
            corrected = np.power(df[feature], round(exp_val, 3))
            name = f"{feature}_{round(exp_val, 3)}"
        # Add the corrected feature to the original DataFrame
        df[name] = corrected
    else:
        name = f"{feature}_binary"
        df[name] = df[feature]
        if skew > 0:
            df.loc[df[name] == df[name].value_counts().index[0], name] = 0
            df.loc[df[name] != df[name].value_counts().index[0], name] = 1
        else:
            df.loc[df[name] == df[name].value_counts().index[0], name] = 1
            df.loc[df[name] != df[name].value_counts().index[0], name] = 0
        if messages:
            print(
                f"The feature {feature} could not be transformed into a normal distribution."
            )
            print("Instead, it has been transformed into a binary (0/1) distribution.")

    # Plot visualisations
    if messages:
        fig, axs = plt.subplots(1, 2, figsize=FIG_SIZE, dpi=FIG_DPI)
        sns.despine(left=True)
        sns.histplot(df_temp[feature], color="b", ax=axs[0], kde=True)
        if (skew > -0.1) and (skew < 0.1):
            if skew > 0:
                corrected = np.power(df_temp[feature], 1 / round(exp_val, 3))
            else:
                corrected = np.power(df_temp[feature], round(exp_val, 3))
            df_temp["corrected"] = corrected
            sns.histplot(df_temp["corrected"], color="g", ax=axs[1], kde=True)
        else:
            df_temp["corrected"] = df_temp[feature]
            if skew > 0:
                df_temp.loc[
                    df_temp["corrected"] == df_temp["corrected"].min(), "corrected"
                ] = 0
                df_temp.loc[
                    df_temp["corrected"] > df_temp["corrected"].min(), "corrected"
                ] = 1
            else:
                df_temp.loc[
                    df_temp["corrected"] == df_temp["corrected"].max(), "corrected"
                ] = 1
                df_temp.loc[
                    df_temp["corrected"] < df_temp["corrected"].max(), "corrected"
                ] = 0
            sns.countplot(data=df_temp, x="corrected", color="g", ax=axs[1])
        plt.suptitle(f"Skew of {feature} before and after transformation", fontsize=29)
        plt.setp(axs, yticks=[])
        plt.tight_layout()
        plt.show()
    return df


## Missing Data
##*********************##

def missing_drop(
    df, label="", features=[], row_threshold=0.90, col_threshold=0.50, messages=True
):
    """
    Drop all columns and rows that have more missing values than the given threshold.

    Parameters
    ----------
    - df : pandas DataFrame
        DataFrame to drop missing values from
    - label : str, default ""
        Label to drop columns with
    - features : list of str, default []
        Columns to consider for dropping
    - row_threshold : float, default 0.90
        Threshold to determine if a row is missing too many values
    - col_threshold : float, default 0.50
        Threshold to determine if a column is missing too many values
    - messages : bool, default True
        If `True`, print messages indicating which columns were dropped and how many missing values were dropped

    Returns
    -------
    df : pandas DataFrame
        DataFrame with columns and rows dropped

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
        >>> missing_drop(df.copy())

        Author
    ------
        tunprimus
    """
    try:
        import fireducks.pandas as pd
    except ImportError:
        import pandas as pd

    pd.set_option("mode.copy_on_write", True)

    start_count = df.count().sum()
    # Drop all columns that have less data than the proportion col_threshold requires
    col_thresh_val = round((col_threshold * df.shape[0]), 0)
    missing_col_thresh = df.shape[0] - col_thresh_val
    if messages:
        print(
            start_count,
            "out of",
            df.shape[0] * df.shape[1],
            "in",
            df.shape[0],
            "rows(s)",
        )
        print(
            f"Going to drop any column with more than {missing_col_thresh} missing value(s)."
        )
    df.dropna(axis=1, thresh=col_thresh_val, inplace=True)

    # Drop all rows that have less data than the proportion row_threshold requires
    row_thresh_val = round((row_threshold * df.shape[1]), 0)
    missing_row_thresh = df.shape[1] - row_thresh_val
    if messages:
        print(
            start_count,
            "out of",
            df.shape[0] * df.shape[1],
            "in",
            df.shape[1],
            "column(s)",
        )
        print(
            f"Going to drop any row with more than {missing_row_thresh} missing value(s)."
        )
    df.dropna(axis=0, thresh=row_thresh_val, inplace=True)

    # Drop all column(s) of given label(s)
    if label != "":
        df.dropna(axis=0, subset=[label], inplace=True)
        if messages:
            print(f"Dropped all column(s) with {label} feature(s).")

    # Function to generate table of residuals if rows/columns with missing values are dropped
    def generate_missing_table():
        df_results = pd.DataFrame(
            columns=["num_missing", "after_column_drop", "after_rows_drop"]
        )
        for feat in df:
            missing = df[feat].isna().sum()
            if missing > 0:
                rem_col = df.drop(columns=[feat]).count().sum()
                rem_rows = df.dropna(subset=[feat]).count().sum()
                df_results.loc[feat] = [missing, rem_col, rem_rows]
        return df_results

    df_results = generate_missing_table()
    while df_results.shape[0] > 0:
        max_val = df_results[["after_column_drop", "after_rows_drop"]].max(axis=1)[0]
        max_val_axis = df_results.columns[df_results.isin([max_val]).any()][0]
        print(max_val, max_val_axis)
        df_results.sort_values(by=[max_val_axis], ascending=False, inplace=True)
        if messages:
            print("\n", df_results)
        if max_val_axis == "after_rows_drop":
            df.dropna(axis=0, subset=[df_results.index[0]], inplace=True)
        else:
            df.drop(columns=[df_results.index[0]], inplace=True)
        df_results = generate_missing_table()
    if messages:
        print(
            f"{round(((df.count().sum() / start_count) * 100), 2)}% ({df.count().sum()} out of {start_count}) of non-null cells were kept after dropping."
        )
    # Return the final DataFrame
    return df


def check_target_balance(df, target="target"):
    """
    Plot the bar chart of target classes count.

    Parameters
    ----------
    - df : pandas.core.frame.DataFrame
        The DataFrame containing the target column.
    - target : str, default "target"
        The name of the target column.

    Returns
    -------
    None

    Notes
    -----
    This function will show a bar chart of the target classes count.
    The x-axis is the unique values of the target column and the y-axis is the count of each target class.
    The bar colours are red for class 0 and green for class 1.

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
        >>> check_target_balance(df, target="smoker")

        Author
    ------
        tunprimus
    """
    import matplotlib.pyplot as plt

    try:
        import fireducks.pandas as pd
    except ImportError:
        import pandas as pd

    colours = ["red", "green", "yellow", "blue", "orange"]
    font = {"size": 17}

    target_unique = df[target].unique()
    target_values = df[target].value_counts()
    target_values_pct = (df[target].value_counts(normalize=True) * 100).round(1)

    print(target_values)

    fig, ax = plt.subplots()

    bar_container = ax.bar(target_unique, target_values, color=colours)
    plt.xticks([val for _, val in enumerate(target_unique)])
    ax.bar_label(bar_container, fmt="{:,.0f}")
    plt.xlabel(f"Target `{target}´ Classes")
    plt.ylabel("Count")
    plt.title(f"Count of each `{target}´ Class", fontdict=font)
    plt.show()
    print(f"`{target}` Class Percentages")
    print(target_values_pct)


# Function to test and print statistics reports
def check_and_print_stats_gen(report_obj, alpha_val=ALPHA_VALUE, messages=False):
    """
    Evaluate and print statistical test results, highlighting significance.

    This function processes a dictionary of statistical test results, checking
    each for significance against a provided alpha value. If `messages` is True,
    it prints the results, indicating whether each test is significant.

    Parameters
    ----------
    report_obj : dict
        A nested dictionary containing the results of statistical tests,
        where keys are test names and values are either another dictionary
        or a single test result.
    alpha_val : float, optional
        The alpha value (significance level) to compare against p-values.
        Default is ALPHA_VALUE.
    messages : bool, optional
        If True, prints the results of statistical tests. Default is False.

    Returns
    -------
    None

    Notes
    -----
    Each statistical test is checked for significance by comparing its p-value
    against the provided alpha value. If the p-value is less than alpha, the
    test is considered significant.

    Author
    ------
        tunprimus
    """

    def get_stats(report_obj):
        for key, val in report_obj.items():
            if isinstance(val, dict):
                for pair in get_stats(val):
                    yield (key, *pair)
            else:
                yield (key, val)

    if messages:
        for item in get_stats(stat_results):
            if item[1] == "p_value" and item[-1] < alpha_val:
                print(
                    f"The {item[0].capitalize()} statistics for `{item[2].capitalize()} content` is `significant` with a p-value of: {item[-1]:.4f}.\n"
                )
            elif item[1] == "p_value" and item[-1] > alpha_val:
                print(
                    f"The {item[0].capitalize()} statistics for `{item[2].capitalize()} content` has a p-value of: {item[-1]:.4f} - not significant.\n"
                )
            else:
                print(
                    f"The {item[0].capitalize()} statistics for `{item[2].capitalize()} content` is: {item[-1]:.4f}.\n"
                )


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
