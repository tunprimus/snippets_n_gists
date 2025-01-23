import pandas as pd
import numpy as np
import boto3
import zipfile
import pyarrow.parquet as pq
import yaml
import gspread
from os.path import realpath as realpath
from os.path import splitext as splitext
from io import StringIO, BytesIO
from fastavro import reader
from scipy.io import loadmat
from google.cloud import bigquery

np.float = np.float64
np.int = np.int_
np.object = np.object_
np.bool = np.bool_

pd.set_option("mode.copy_on_write", True)

def generate_df_from_data_source(data_source,from_aws=False, aws_access_key=None, aws_secret_key=None, aws_bucket_name=None, aws_file_key=None, from_googlesheet=False, path_to_googlesheet_cred=None, googlesheet_name=None, from_bigquery=False, path_to_bigquery_cred=None, bigquery_id=None, bigquery_dataset_id=None, bigquery_table_id=None):
    """
    Automatically detect the type of the file and use the appropriate pandas method to create a DataFrame.

    Parameters
    ----------
    data_source : str
        The path to the data source file.
    from_aws : bool, default False
        Load the data from AWS S3.
    aws_access_key : str, optional
        The access key for AWS S3.
    aws_secret_key : str, optional
        The secret key for AWS S3.
    aws_bucket_name : str, optional
        The name of the AWS S3 bucket.
    aws_file_key : str, optional
        The key of the file in the AWS S3 bucket.
    from_googlesheet : bool, default False
        Load the data from Google Sheets.
    path_to_googlesheet_cred : str, optional
        The path to the Google Sheets credentials file.
    googlesheet_name : str, optional
        The name of the Google Sheets document.
    from_bigquery : bool, default False
        Load the data from Google BigQuery.
    path_to_bigquery_cred : str, optional
        The path to the Google BigQuery credentials file.
    bigquery_id : str, optional
        The ID of the BigQuery project.
    bigquery_dataset_id : str, optional
        The ID of the BigQuery dataset.
    bigquery_table_id : str, optional
        The ID of the BigQuery table.

    Returns
    -------
    df : pd.DataFrame
        A DataFrame containing the data from the data source.
    """
    real_path_to_data_source = realpath(data_source)
    file_path_name, ext_buffer = splitext(real_path_to_data_source)
    ext = ext_buffer.lstrip(".")
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
    elif ext == "db" or ext == "sqlite" or ext == "sqlite3" or ext == "sql":
        return pd.read_sql(real_path_to_data_source, con=real_path_to_data_source)
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
    elif ext == "yaml" or ext == "yml":
        with open(real_path_to_data_source, "r") as f:
            return pd.json_normalize(yaml.safe_load(f))
    elif from_aws:
        s3 = boto3.client("s3", aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key)
        obj = s3.get_object(Bucket=aws_bucket_name, Key=aws_file_key)
        return pd.read_csv(BytesIO(obj["Body"].read()))
    elif from_googlesheet:
        gc = gspread.service_account(filename=realpath(path_to_googlesheet_cred))
        worksheet = gc.open(googlesheet_name)
        return pd.DataFrame(worksheet.get_worksheet(0).get_all_records())
    elif from_bigquery:
        client = bigquery.Client.from_service_account_json(realpath(path_to_bigquery_cred))
        query = f"SELECT * FROM {bigquery_id}.{bigquery_dataset_id}.{bigquery_table_id}"
        try:
            return client.query(query).to_dataframe()
        except Exception as exc:
            print(exc)
            return pd.read_gbq(query, dialect="standard", project_id=bigquery_id)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


