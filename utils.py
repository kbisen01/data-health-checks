import os
from io import BytesIO, StringIO
import pandas as pd
import boto3
import numpy as np
import botocore


class S3Utils:
    @staticmethod
    def get_s3_obj_body(
        key: str, bucket_name: str = os.getenv("BUCKET_NAME")
    ) -> botocore.response.StreamingBody:
        # s3_client = boto3.client("s3")
        # obj = s3_client.get_object(Bucket=bucket_name, Key=key)
        # body = obj["Body"]

        # Modified code for testing in local
        with open(key, 'rb') as f:
            local_csv_content = f.read()
        body = BytesIO(local_csv_content)
        return body

    @staticmethod
    def upload_csv_to_s3(
        df: pd.DataFrame, key: str, bucket_name: str = os.getenv("BUCKET_NAME")
    ) -> None:
        # s3_client = boto3.client("s3")
        # csv_buffer = StringIO()
        # df.to_csv(csv_buffer, index=False)
        # s3_client.put_object(Bucket=bucket_name, Key=key, Body=csv_buffer.getvalue())

        # Modified code for testing in local
        df.to_csv(key, index=False)
        print(f"Upload successful to {bucket_name}/{key}")


def extract_rma_from_wo(workorder_column) -> list:
    rma_number = workorder_column.str.extract("RMA(.*)")[
        0
    ]  # Extracting RMA number from workorder
    rma_number = rma_number.str.split("-").str[0]
    rma_number = rma_number.str.split("T").str[0]
    rma_number = rma_number.replace("", np.nan)
    rma_number = rma_number.replace(r"[^0-9]", np.nan, regex=True)
    return rma_number


# Set Serial Numbers DataType
def dtype_sn(sn_column):
    sn_column = sn_column.astype(str)
    sn_column = sn_column.str.split(".").str[0]
    return sn_column


# Set RMA Numbers DataType
def dtype_rma(rma_column):
    rma_column = pd.to_numeric(rma_column)
    rma_column = rma_column.astype("Int64")
    return rma_column


def get_current_repair_status(
    base_data,
    repair_status,
    status_column_name="Current_Repair_Status",
    date_column_name=None,
):
    """Returns base data with current repair status column with values cooresponding to corresponsing to
    RMA_dim_id(preferred) or serial_number/rma_number
    Parameters
    ----------
    base_data: Pandas DataFrame

    repair_status: Pandas DataFrame

    status_column_name: string, None
        If None status_column won`t be added to base added.
    date_column_name: string, None
        If None column with current status date won`t be added
    """
    if (status_column_name == None) & (date_column_name == None):
        return "No column select to add in base data"

    if any(
        x in base_data.columns.tolist() for x in (status_column_name, date_column_name)
    ):
        raise ValueError(
            f"{status_column_name} or {date_column_name} already exists in base_data"
        )

    if ("RMA_dim_id" in repair_status.columns.tolist()) & (
        "RMA_dim_id" in base_data.columns.tolist()
    ):
        merge_col = ["RMA_dim_id"]
    elif (
        all(
            x in repair_status.columns.tolist() for x in ["serial_number", "rma_number"]
        )
    ) & (all(x in base_data.columns.tolist() for x in ["serial_number", "rma_number"])):
        merge_col = ["serial_number", "rma_number"]
    else:
        raise ValueError(
            "Neither RMA_dim_id nor serial_number and rma_number found in base_data or repair_status!"
        )

    required_cols = ["id", "date", "status"] + merge_col
    repair_status = repair_status[required_cols]
    repair_status["date"] = pd.to_datetime(
        repair_status["date"], infer_datetime_format=True
    )
    repair_status = repair_status.sort_values(
        by=["date", "id"], ascending=True
    ).reset_index(drop=True)
    repair_status = repair_status.drop(columns=["id"], axis=1)
    repair_status = repair_status.drop_duplicates(subset=merge_col, keep="last")
    if status_column_name == None:
        repair_status.drop(columns=["status"], axis=1, inplace=True)
    if date_column_name == None:
        repair_status.drop(columns=["date"], axis=1, inplace=True)

    repair_status = repair_status.rename(
        columns={"date": date_column_name, "status": status_column_name}
    )

    if merge_col == ["RMA_dim_id"]:
        base_data["RMA_dim_id"] = base_data["RMA_dim_id"].astype(int)
        base_data["RMA_dim_id"] = base_data["RMA_dim_id"].astype(int)
    else:
        base_data["serial_number"] = dtype_sn(base_data, "serial_number")
        base_data["rma_number"] = dtype_rma(base_data, "rma_number")
        repair_status["serial_number"] = dtype_sn(repair_status, "serial_number")
        repair_status["rma_number"] = dtype_rma(repair_status, "rma_number")

    base_data = base_data.merge(repair_status, on=merge_col, how="left")

    return base_data


def get_current_rma_status(
    base_data,
    rma_operation,
    status_column_name="Current_RMA_status",
    date_column_name=None,
):
    if (status_column_name == None) & (date_column_name == None):
        return "No column select to add in base data"

    if any(
        x in base_data.columns.tolist() for x in (status_column_name, date_column_name)
    ):
        raise ValueError(
            f"{status_column_name} or {date_column_name} already exists in base_data"
        )

    if ("RMA_dim_id" in rma_operation.columns.tolist()) & (
        "RMA_dim_id" in base_data.columns.tolist()
    ):
        merge_col = ["RMA_dim_id"]
    elif (
        all(
            x in rma_operation.columns.tolist() for x in ["serial_number", "rma_number"]
        )
    ) & (all(x in base_data.columns.tolist() for x in ["serial_number", "rma_number"])):
        merge_col = ["serial_number", "rma_number"]
    else:
        raise ValueError(
            "Neither RMA_dim_id nor serial_number and rma_number found in base_data or rma_operation!"
        )

    required_cols = ["id", "date", "process_status"] + merge_col
    rma_operation = rma_operation[required_cols]
    rma_operation["date"] = pd.to_datetime(
        rma_operation["date"], infer_datetime_format=True
    )
    rma_operation = rma_operation.sort_values(
        by=["date", "id"], ascending=True
    ).reset_index(drop=True)
    rma_operation = rma_operation.drop(columns=["id"], axis=1)
    rma_operation = rma_operation.drop_duplicates(subset=merge_col, keep="last")

    if status_column_name == None:
        rma_operation.drop(columns=["process_status"], axis=1, inplace=True)
    if date_column_name == None:
        rma_operation.drop(columns=["date"], axis=1, inplace=True)

    rma_operation = rma_operation.rename(
        columns={"date": date_column_name, "process_status": status_column_name}
    )

    if merge_col == ["RMA_dim_id"]:
        base_data["RMA_dim_id"] = base_data["RMA_dim_id"].astype(int)
        base_data["RMA_dim_id"] = base_data["RMA_dim_id"].astype(int)
    else:
        base_data["serial_number"] = dtype_sn(base_data, "serial_number")
        base_data["rma_number"] = dtype_rma(base_data, "rma_number")
        rma_operation["serial_number"] = dtype_sn(rma_operation, "serial_number")
        rma_operation["rma_number"] = dtype_rma(rma_operation, "rma_number")

    base_data = base_data.merge(rma_operation, on=merge_col, how="left")

    return base_data


def add_rma_status_date(base_data, rma_operation, process_status: str, new_column: str):
    """Adds a new column to your base data with rma status (Created/Receipt/Shipped) date
    for the corresponding RMA and serial number.

    Args:
        base_data (pandas.DataFrame): DataFrame on which you want to add the column
        rma_operation (pandas.DataFrame): rma_operation table from database
        process_status (str): [Created or Receipt or Shipped] whichever status you need the date for.
        new_column (str): New Column Name

    Raises:
        KeyError: When the common columns between base_data and rma_operation is not found
        ValueError:Value provided in process_status is not valid, i.e. not part of process_status column in rma_operations

    Returns:
        pandas.DataFrame: base_data with new_column added to right.
    """
    if ("RMA_dim_id" in base_data.columns.tolist()) & (
        "RMA_dim_id" in rma_operation.columns.tolist()
    ):
        merge_col = ["RMA_dim_id"]
    elif all(
        x in base_data.columns.tolist() for x in ["serial_number", "rma_number"]
    ) & all(
        x in rma_operation.columns.tolist() for x in ["serial_number", "rma_number"]
    ):
        merge_col = ["serial_number", "rma_number"]
    else:
        raise KeyError(
            "Neither RMA_dim_id nor serial_number or rma_number found in base_data or rma_operation. Confirm column names."
        )

    valid_process_status = rma_operation["process_status"].unique().tolist()
    if process_status not in valid_process_status:
        raise ValueError(
            f"results: process_status must be one of {valid_process_status}."
        )

    rma_operation = rma_operation[rma_operation["process_status"] == process_status]
    rma_operation["date"] = pd.to_datetime(rma_operation["date"])
    required_cols = merge_col + ["date"]
    rma_operation = rma_operation[required_cols].rename(columns={"date": new_column})

    if merge_col == ["RMA_dim_id"]:
        base_data["RMA_dim_id"] = base_data["RMA_dim_id"].astype(int)
        rma_operation["RMA_dim_id"] = rma_operation["RMA_dim_id"].astype(int)
    else:
        base_data["serial_number"] = dtype_sn(base_data, "serial_number")
        rma_operation["serial_number"] = dtype_sn(rma_operation, "serial_number")
        base_data["rma_number"] = dtype_rma(base_data, "rma_number")
        rma_operation["rma_number"] = dtype_rma(rma_operation, "rma_number")

    base_data = base_data.merge(rma_operation, on=merge_col, how="left")
    return base_data

def filter_new_cases(df:pd.DataFrame, existing_cases:pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        print("Data output is empty!!!")
        return df

    if existing_cases.empty:
        print("No existing cases!!!")
        return df
    
    # Verify if both dataframe have same columns.
    # If not then log the list of columns not in df or addtional in df and raise error
    if set(df.columns) != set(existing_cases.columns):
        new_cols = set(df.columns) - set(existing_cases.columns)
        missing_cols = set(existing_cases.columns) - set(df.columns)
        print(f"New columns in data: {new_cols}")
        print(f"Missing columns in data: {missing_cols}")
        raise ValueError(" and existing cases have different columns!")

    if set(df['dh_failure']) != set(existing_cases['dh_failure']):
        print("Data Health failure message is not same!!!")
        # return empty dataframe with same columns as df
        return pd.DataFrame(columns= df.columns)
    
    df = df.merge(existing_cases, how='left', indicator=True)
    df = df.loc[df['_merge'] == 'left_only']
    df = df.drop(columns=['_merge'])
    return df
