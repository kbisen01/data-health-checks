# Writie a script to check if all data health checks are satisfied.

from copy import copy
import os
import dotenv
import numpy as np
import pandas as pd
from utils import S3Utils
from dataclasses import dataclass

dotenv.load_dotenv()

tod = pd.Timestamp(year=2023, month=8, day=24, hour=15, minute=17)  # placeholder


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

@dataclass
class BaseData(frozen = True):
    def load_process_table(self) -> pd.DataFrame:
        """
        load process data from s3 bucket.
        """

        process_file_location = "input/Process all.csv"
        opn_map_location = "input/operations_mapping.csv"

        process_columns: list = [
            "step_id",
            "serial_no",
            "workorder",
            "trans_seq",
            "sn_attr_code",
            "operation",
            "status",
            "datetime_checkin",
            "datetime_checkout",
        ]
        process_rename: dict = {
            "serial_no": "serial_number",
        }
        process_types: dict = {
            "step_id": int,
            "serial_no": str,
            "workorder": str,
            "trans_seq": int,
            "sn_attr_code": int,
            "operation": str,
            "status": str,
        }

        process_parse_dates: list = ["datetime_checkin", "datetime_checkout"]

        process_df = pd.read_csv(
            S3Utils.get_s3_obj_body(process_file_location),
            usecols=process_columns,
            dtype=process_types,
            parse_dates=process_parse_dates,
        )
        process_df["serial_no"] = (
            process_df["serial_no"].str.split(".").str[0]
        )
        process_df = process_df.rename(columns=process_rename)

        operation_columns: list = ["operation", "OPN_Map"]
        operation_types: dict = {"operation": str, "OPN_Map": str}

        # Add OPN_Map column to process_df
        opn_map_df = pd.read_csv(
            S3Utils.get_s3_obj_body(opn_map_location),
            usecols=operation_columns,
            dtype=operation_types,
        )
        process_df = process_df.merge(
            opn_map_df, on="operation", how="left"
        )

        process_df["rma_number"] = extract_rma_from_wo(
            process_df["workorder"]
        )
        process_df["rma_number"] = dtype_rma(process_df["rma_number"])
        return process_df

    def load_current_fits_operation(self) -> pd.DataFrame:
        """
        load current FITS operation from s3 bucket.
        """

        current_fits_operation_location = "input/process_current_operation.csv"
        opn_map_location = "input/operations_mapping.csv"

        current_fits_operation_columns: list = [
            "step_id",
            "serial_no",
            "workorder",
            "trans_seq",
            "sn_attr_code",
            "operation",
            "status",
            "datetime_checkin",
            "datetime_checkout",
        ]
        current_fits_operation_rename: dict = {
            "serial_no": "serial_number",
        }
        current_fits_operation_types: dict = {
            "step_id": int,
            "serial_no": str,
            "workorder": str,
            "trans_seq": int,
            "sn_attr_code": int,
            "operation": str,
            "status": str,
        }

        current_fits_operation_parse_dates: list = [
            "datetime_checkin",
            "datetime_checkout",
        ]

        current_fits_operation_df = pd.read_csv(
            S3Utils.get_s3_obj_body(current_fits_operation_location),
            usecols=current_fits_operation_columns,
            dtype=current_fits_operation_types,
            parse_dates=current_fits_operation_parse_dates,
        )
        current_fits_operation_df["serial_no"] = (
            current_fits_operation_df["serial_no"].str.split(".").str[0]
        )
        current_fits_operation_df = current_fits_operation_df.rename(
            columns=current_fits_operation_rename
        )

        operation_columns: list = ["operation", "OPN_Map"]
        operation_types: dict = {"operation": str, "OPN_Map": str}

        # Add OPN_Map column to process_df
        opn_map_df = pd.read_csv(
            S3Utils.get_s3_obj_body(opn_map_location),
            usecols=operation_columns,
            dtype=operation_types,
        )
        current_fits_operation_df = current_fits_operation_df.merge(
            opn_map_df, on="operation", how="left"
        )

        current_fits_operation_df["rma_number"] = extract_rma_from_wo(
            current_fits_operation_df["workorder"]
        )
        current_fits_operation_df["rma_number"] = dtype_rma(
            current_fits_operation_df["rma_number"]
        )
        # self._current_fits_operation_df = self._current_fits_operation_df.rename(
        #     columns={
        #         "operation": "current_operation",
        #         "OPN_Map": "current_OPN_map",
        #         "status": "current_status",
        #     }
        # )
        return current_fits_operation_df

    def load_parameter_table(self) -> pd.DataFrame:
        """
        load parameter data from s3 bucket.
        """
        parameter_file_location = "input/Parameter all.csv"

        parameter_columns: list = [
            "parameter_id",
            "serial_no",
            "trans_seq",
            "sn_attr_code",
            "attribute_code",
            "parameter_name",
            "parameter_value",
            "datetime_script",
        ]
        parameter_rename: dict = {"serial_no": "serial_number"}

        parameter_types: dict = {
            "parameter_id": int,
            "serial_no": str,
            "trans_seq": int,
            "sn_attr_code": int,
            "attribute_code": int,
            "parameter_name": str,
            "parameter_value": str,
        }

        parameter_parse_dates: list = ["datetime_script"]

        parameter_df = pd.read_csv(
            S3Utils.get_s3_obj_body(parameter_file_location),
            usecols=parameter_columns,
            dtype=parameter_types,
            parse_dates=parameter_parse_dates,
        )
        parameter_df["serial_no"] = (
            parameter_df["serial_no"].str.split(".").str[0]
        )
        parameter_df = parameter_df.rename(columns=parameter_rename)
        return parameter_df

    def load_rma_data(self) -> pd.DataFrame:
        """
        load index RMA dimension from s3 bucket.
        """
        _FABRINET_LOCATION_CODE: list[str] = ["FBN", "FPT"]
        _NON_RMA_CREATED_FROM_CODE = "NONRMA"

        index_rma_dim_file_location = "input/RMA_dim.csv"
        serial_file_location = "input/serialship.csv"

        rma_dim_columns: list = [
            "id",
            "rma_number",
            "FA_location",
            "Serial_detail_id",
            "created_from",
        ]
        rma_dim_rename: dict = {"id": "RMA_dim_id"}
        rma_dim_types: dict = {
            "id": int,
            "rma_number": "Int64",
            "FA_location": str,
            "Serial_detail_id": int,
        }

        rma_dim_parse_dates = []

        # serialship columns
        serialship_columns: list = ["id", "serial_number"]
        serialship_rename: dict = {"id": "Serial_detail_id"}
        serialship_types: dict = {
            "id": int,
            "serial_number": str,
        }

        serialship_parse_dates = []

        rma_dim_df = pd.read_csv(
            S3Utils.get_s3_obj_body(index_rma_dim_file_location),
            usecols=rma_dim_columns,
            dtype=rma_dim_types,
            parse_dates=rma_dim_parse_dates,
        )
        rma_dim_df = rma_dim_df.rename(columns=rma_dim_rename)
        rma_dim_df = rma_dim_df[
            rma_dim_df["FA_location"].isin(_FABRINET_LOCATION_CODE)
        ]
        rma_dim_df = rma_dim_df[
            rma_dim_df["created_from"] != _NON_RMA_CREATED_FROM_CODE
        ]

        serialship_df = pd.read_csv(
            S3Utils.get_s3_obj_body(serial_file_location),
            usecols=serialship_columns,
            dtype=serialship_types,
            parse_dates=serialship_parse_dates,
        )
        serialship_df = serialship_df.rename(columns=serialship_rename)
        serialship_df["serial_number"] = (
            serialship_df["serial_number"].str.split(".").str[0]
        )

        rma_data_df = rma_dim_df.merge(
            serialship_df, on="Serial_detail_id", how="left"
        )
        rma_data_df = rma_data_df.drop(
            columns=["FA_location", "Serial_detail_id", "created_from"]
        )
        return rma_data_df

    def load_rma_operations(self) -> pd.DataFrame:
        """
        load RMA operations from s3 bucket.
        """
        rma_operations_file_location = "input/rma_operations.csv"

        rma_operations_columns: list = ["id", "process_status", "date", "RMA_dim_id"]

        rma_operations_types: dict = {
            "process_status": str,
            "RMA_dim_id": int,
        }

        rma_operation_parse_dates: list = ["date"]

        rma_operations_df = pd.read_csv(
            S3Utils.get_s3_obj_body(rma_operations_file_location),
            usecols=rma_operations_columns,
            dtype=rma_operations_types,
            parse_dates=rma_operation_parse_dates,
        )
        return rma_operations_df

    def load_repair_status(self) -> pd.DataFrame:
        """
        load repair status from s3 bucket.
        """
        repair_status_file_location = "input/repair_status.csv"

        repair_status_columns: list = ["id", "status", "date", "time", "RMA_dim_id"]

        repair_status_types: dict = {
            "status": str,
            "RMA_dim_id": int,
        }

        repair_status_parse_dates: list = ["date"]

        repair_status_df = pd.read_csv(
            S3Utils.get_s3_obj_body(repair_status_file_location),
            usecols=repair_status_columns,
            dtype=repair_status_types,
            parse_dates=repair_status_parse_dates,
        )
        # Change time column
        repair_status_df["time"] = repair_status_df["time"].astype(
            "timedelta64"
        )
        repair_status_df["date"] = repair_status_df[
            "date"
        ] + repair_status_df["time"].fillna(pd.Timedelta(0))
        repair_status_df = repair_status_df.drop(columns=["time"])
        return repair_status_df

    process_df = load_process_table()
    current_fits_operation_df = load_current_fits_operation()
    parameter_df = load_parameter_table()
    rma_data = load_rma_data()
    rma_operations = load_rma_operations()
    repair_status = load_repair_status()

class PortalDataHealthChecks:
    _DIAGNOSIS_OPERATION_CODE = ["RMA08", "RMA09"]
    _CLOSED_FITS_STATUS_CODE = ["SALVAGE", "SHIPPED", "SCRAPED"]
    output_dir = f"output/{tod.strftime('%Y%m%d')}"
    captured_cases_dir = "output/captured_cases"

    _dh_rma_data_columns: list = [
            "RMA_dim_id",
            "serial_number",
            "rma_number",
            "dh_failure",
        ]

    dh_rma_data_df: pd.DataFrame = pd.DataFrame(columns=_dh_rma_data_columns)

    def __init__(self, BaseData):
        BaseData = BaseData
        self._load_dh_rma_data()
        self._setup_output_directory()

    def _load_dh_rma_data(self) -> None:
        _dh_rma_data_file_location = "output/captured_cases/DH_RMA_Data.csv"
        _dh_rma_data_columns: list = [
            "RMA_dim_id",
            "serial_number",
            "rma_number",
            "dh_failure",
        ]

        _dh_rma_data_types: dict = {
            "RMA_dim_id": int,
            "serial_number": str,
            "rma_number": "Int64",
            "dh_failure": str,
        }

        try:
            self._captured_dh_rma_data_df = pd.read_csv(
                S3Utils.get_s3_obj_body(_dh_rma_data_file_location),
                usecols=_dh_rma_data_columns,
                dtype=_dh_rma_data_types,
            )
            self._captured_dh_rma_data_df["serial_number"] = (
                self._captured_dh_rma_data_df["serial_number"].str.split(".").str[0]
            )
        except FileNotFoundError:
            self._captured_dh_rma_data_df = pd.DataFrame(columns=_dh_rma_data_columns)

    # Does the file pulled in has info on all the RMA that was recieveed atleast 3 days ago and isn't repair closed?
    def check_all_repair_rma_present(self) -> "RMADataHealthOutput":
        temp = copy(self)

        rma_data_wip = get_current_repair_status(
            temp._rma_data_df,
            temp._repair_status_df,
            status_column_name="current_repair_status",
        )
        # WIP RMAs cannot be in Reapir Closed or empty
        rma_data_wip = rma_data_wip.dropna(subset=["current_repair_status"])
        rma_data_wip = rma_data_wip.loc[
            rma_data_wip["current_repair_status"] != "Repair Close"
        ]

        process_df = temp._process_df[
            ["serial_number", "rma_number"]
        ].drop_duplicates()
        rma_data_wip = rma_data_wip.merge(
            process_df,
            on=["serial_number", "rma_number"],
            how="left",
            indicator=True,
        )
        temp.output = rma_data_wip.loc[
            rma_data_wip["_merge"] == "left_only"
        ]
        temp.output = temp.output.drop(columns=["_merge"])
        temp.output["dh_failure"] = "Absent from FITS"
        return temp

class ProcessDataHealthOutput:
    # output_dir = f"output/{tod.strftime('%Y%m%d')}"
    # captured_cases_dir = "output/captured_cases"

    def __init__(self, output_file:str, captured_cases_file:str) -> None:
        self.output_file = output_file
        self.captured_cases_file = captured_cases_file
        self._load_dh_process()
        self._setup_output_directory()

    _dh_process_columns: list = [
            "step_id",
            "serial_number",
            "rma_number",
            "workorder",
            "trans_seq",
            "sn_attr_code",
            "operation",
            "OPN_Map",
            "status",
            "datetime_checkin",
            "datetime_checkout",
            "dh_failure",
        ]

    dh_process_df: pd.DataFrame = pd.DataFrame(columns=_dh_process_columns)

    def _load_dh_process(self) -> None:
        _dh_process_file_location = "output/captured_cases/DH_Process.csv"
        _dh_process_columns: list = [
            "step_id",
            "serial_number",
            "rma_number",
            "workorder",
            "trans_seq",
            "sn_attr_code",
            "operation",
            "OPN_Map",
            "status",
            "datetime_checkin",
            "datetime_checkout",
            "dh_failure",
        ]

        _dh_process_types: dict = {
            "step_id": int,
            "serial_number": str,
            "rma_number": "Int64",
            "workorder": str,
            "trans_seq": int,
            "sn_attr_code": int,
            "operation": str,
            "OPN_Map": str,
            "status": str,
            "datetime_checkin": str,
            "datetime_checkout": str,
            "dh_failure": str,
        }

        _dh_process_parse_dates: list = ["datetime_checkin", "datetime_checkout"]

        try:
            self._captured_dh_process_df = pd.read_csv(
                S3Utils.get_s3_obj_body(_dh_process_file_location),
                usecols=_dh_process_columns,
                dtype=_dh_process_types,
                parse_dates=_dh_process_parse_dates,
            )
            self._captured_dh_process_df["serial_number"] = (
                self._captured_dh_process_df["serial_number"].str.split(".").str[0]
            )
        except FileNotFoundError:
            self._captured_dh_process_df = pd.DataFrame(columns=_dh_process_columns)

    def append_with_dh_process(self):
        """
        Append the data health output to dh_process
        """

        # Copy DataFrame
        temp = copy(self)

        # Check quality of output to be included in dh_process
        # Check if dh_process_df is empty
        if temp.output.empty:
            print("Data health output is empty")

        # Append new data
        try:
            temp.output = temp.output.loc[:, temp._dh_process_columns]
        except KeyError as e:
            print(e)
            raise KeyError("Data health output does not have all the columns required in dh_process. You could be trying to append some other dh to process dh")
        
        temp.dh_process_df = pd.concat(
            [temp.dh_process_df, temp.output], ignore_index=True
        )
        return temp

    def _setup_output_directory(self) -> None:        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        if not os.path.exists(self.captured_cases_dir):
            os.makedirs(self.captured_cases_dir)

    def count(self) -> int:
        """
        Return number of rows in current selection
        """

        return self.output.shape[0]

    def fetch(self, limit: int = None, sample: bool = False) -> pd.DataFrame:
        """
        Fetch all weather stations or a (sampled) subset
        """

        # Copy DataFrame
        temp = self.output.copy()

        # Return limited number of sampled entries
        if sample and limit:
            return temp.sample(limit)

        # Return limited number of entries
        if limit:
            return temp.head(limit)

        # Return all entries
        return temp

class RMADataHealthOutput:
    _dh_rma_data_columns: list = [
            "RMA_dim_id",
            "serial_number",
            "rma_number",
            "dh_failure",
        ]

    dh_rma_data_df: pd.DataFrame = pd.DataFrame(columns=_dh_rma_data_columns)

    def __init__(self, output_file:str, captured_cases_file:str) -> None:
        self.output_file = output_file
        self.captured_cases_file = captured_cases_file
        self._load_dh_rma_data()
        self._setup_output_directory()

    def _load_dh_rma_data(self) -> None:
        _dh_rma_data_file_location = "output/captured_cases/DH_RMA_Data.csv"
        _dh_rma_data_columns: list = [
            "RMA_dim_id",
            "serial_number",
            "rma_number",
            "dh_failure",
        ]

        _dh_rma_data_types: dict = {
            "RMA_dim_id": int,
            "serial_number": str,
            "rma_number": "Int64",
            "dh_failure": str,
        }

        try:
            self._captured_dh_rma_data_df = pd.read_csv(
                S3Utils.get_s3_obj_body(_dh_rma_data_file_location),
                usecols=_dh_rma_data_columns,
                dtype=_dh_rma_data_types,
            )
            self._captured_dh_rma_data_df["serial_number"] = (
                self._captured_dh_rma_data_df["serial_number"].str.split(".").str[0]
            )
        except FileNotFoundError:
            self._captured_dh_rma_data_df = pd.DataFrame(columns=_dh_rma_data_columns)
    
    def _setup_output_directory(self) -> None:        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        if not os.path.exists(self.captured_cases_dir):
            os.makedirs(self.captured_cases_dir)

    def count(self) -> int:
        """
        Return number of rows in current selection
        """

        return self.output.shape[0]

    def fetch(self, limit: int = None, sample: bool = False) -> pd.DataFrame:
        """
        Fetch all or a (sampled) subset
        """

        # Copy DataFrame
        temp = self.output.copy()

        # Return limited number of sampled entries
        if sample and limit:
            return temp.sample(limit)

        # Return limited number of entries
        if limit:
            return temp.head(limit)

        # Return all entries
        return temp

class FITSDataHealthChecks:
    _critical_error: bool = False
    _DIAGNOSIS_OPERATION_CODE = ["RMA08", "RMA09"]
    _CLOSED_FITS_STATUS_CODE = ["SALVAGE", "SHIPPED", "SCRAPED"]

    def __init__(self, BaseData):
        BaseData = BaseData

    # Are all the columns present in Process table?
    def check_process_columns(self) -> None:
        return "Not required because usecols would have returned an error during class initialization"

    # serial_no, trans_seq, sn_attr_code together should be unique in Process table.
    def check_unique_serial_no_trans_seq(self, df) -> str:
        if df.duplicated(["serial_no", "trans_seq", "sn_attr_code"]).any():
            self._critical_error = True
            return "Critical error set True"
        else:
            return "All serial_no, trans_seq, sn_attr_code are unique"

    # Unknown operation found in process table.
    def check_unknown_operation(self) -> "ProcessDataHealthOutput":
        temp = copy(self)
        temp.output = temp._process_df.loc[temp._process_df["OPN_Map"].isnull()]
        temp.output["dh_failure"] = "Unknown operation"
        return temp

    # Does the RMA in WO matches Portal RMA?
    def check_rma_portal_match(self) -> "ProcessDataHealthOutput":
        temp = copy(self)
        all_rma_in_portal = temp._rma_data_df[
            ["serial_number", "rma_number"]
        ].drop_duplicates()
        temp.output = temp._process_df.merge(
            all_rma_in_portal,
            on=["serial_number", "rma_number"],
            how="left",
            indicator=True,
        )
        temp.output = temp.output.loc[temp.output["_merge"] == "left_only"]
        temp.output = temp.output.drop(columns=["_merge"])
        temp.output["dh_failure"] = "RMA not found in Portal"
        return temp

    # Is the RMA number format correct in WO? RMA3000010000RMA will raise an alert here.
    def check_rma_format(self) -> "ProcessDataHealthOutput":
        temp = copy(self)
        process_df = temp._process_df
        process_df["wo_extract"] = process_df["workorder"].str.extract(
            "RMA(.*)"
        )[0]
        process_df["wo_extract"] = process_df["wo_extract"].str.split("-")
        process_df["wo_serial_number"] = process_df["wo_extract"].str[1]
        process_df["wo_rma_number"] = process_df["wo_extract"].str[0]

        is_rma_numeric = process_df["wo_rma_number"].str.isnumeric()
        is_wo_sn_match_sn = (
            process_df["wo_serial_number"] == process_df["serial_number"]
        )

        temp.output = process_df.loc[(~is_rma_numeric) | (~is_wo_sn_match_sn)]
        temp.output["dh_failure"] = "Incorrect workorder format"
        return temp

    # Oracle Shipped units that are still open in Debug.(Is Adv. Replc. an exception here?)
    def check_oracle_shipped_units(self) -> "ProcessDataHealthOutput":
        temp = copy(self)
        rma_data_shipped = add_rma_status_date(
            temp._rma_data_df,
            temp._rma_operations_df,
            process_status="Shipped",
            new_column="shipped_date",
        )
        rma_data_shipped = rma_data_shipped.loc[
            rma_data_shipped["shipped_date"].notnull()
        ]
        rma_data_shipped = rma_data_shipped[
            ["serial_number", "rma_number"]
        ].drop_duplicates()
        temp.output = temp._process_df.merge(
            rma_data_shipped, on=["serial_number", "rma_number"], how="inner"
        )
        temp.output["dh_failure"] = "Oracle shipped units still in WIP"
        return temp

    # Current status of a Repair Close unit should be Salvage, Shipped, or Scraped(spelled incorrectly in FITS as well), Module replacement (status check in)
    def check_repair_close_fits_status(self) -> "ProcessDataHealthOutput":
        temp = copy(self)
        rma_data_repair_closed = get_current_repair_status(
            temp._rma_data_df,
            temp._repair_status_df,
            status_column_name="current_repair_status",
        )
        rma_data_repair_closed = rma_data_repair_closed.loc[rma_data_repair_closed["current_repair_status"] == "Repair Close"]
        rma_data_repair_closed = rma_data_repair_closed.loc[:, ["serial_number", "rma_number"]].drop_duplicates()
        temp.output = temp._current_fits_operation_df.merge(
            rma_data_repair_closed,
            on=["serial_number", "rma_number"],
            how="inner",
        )
        temp.output = temp.output.loc[~temp.output["status"].str.upper().isin(temp._CLOSED_FITS_STATUS_CODE)]
        temp.output["dh_failure"] = "Repair Close status incorrect"
        return temp

    # Is the trans_seq of Debug/Verification step present in Parameter?
    def check_trans_seq_in_param(self) -> "ProcessDataHealthOutput":
        temp = copy(self)
        process_df = temp._process_df.loc[
            temp._process_df["operation"].isin(temp._DIAGNOSIS_OPERATION_CODE)
        ]
        parameter_df = temp._parameter_df[
            ["serial_number", "trans_seq"]
        ].drop_duplicates()
        temp.output = process_df.merge(
            parameter_df,
            on=["serial_number", "trans_seq"],
            how="left",
            indicator=True,
        )
        temp.output = temp.output.loc[temp.output["_merge"] == "left_only"]
        temp.output = temp.output.drop(columns=["_merge"])
        temp.output["dh_failure"] = "trans_seq not found in Parameter"
        return temp

    # Is the Parameter data entry closed before checkout datetime of that trans_seq?
    def check_param_entry_closed_before_process_checkout(self) -> "ProcessDataHealthOutput":
        temp = copy(self)
        process_df = temp._process_df.loc[
            temp._process_df["operation"].isin(temp._DIAGNOSIS_OPERATION_CODE)
        ]
        parameter_df = temp._parameter_df.groupby(
            by=["serial_number", "trans_seq"], as_index=False
        ).agg({"datetime_script": max})
        parameter_df = parameter_df.rename(
            columns={"datetime_script": "max_parameter_entry_datetime"}
        )
        temp.output = process_df.merge(
            parameter_df, on=["serial_number", "trans_seq"], how="inner"
        )
        temp.output = temp.output.loc[
            temp.output["max_parameter_entry_datetime"]
            > temp.output["datetime_checkout"]
        ]
        temp.output["dh_failure"] = "Parameter entry made after process checkout"
        return temp

    # No New data added in process today.
    # def check_last_update_in_process(self) -> "FITSDataHealth":
    #     temp = copy(self)


    def append_with_dh_rma_data(self) -> "FITSDataHealth":
        """
        Append the data health output to dh_rma_data
        """

        # Copy DataFrame
        temp = copy(self)

        # Check quality of output to be included in dh_rma_data
        # Check if dh_rma_data_df is empty
        if temp.count() == 0:
            print("Data health output is empty")
            return temp

        # Append new data
        try:
            temp.output = temp.output.loc[:, temp._dh_rma_data_columns]
        except KeyError as e:
            print(e)
            raise KeyError("Data health output does not have all the columns required in dh_rma_data. You could be trying to append some other dh to rma_data dh")

        temp.dh_rma_data_df = pd.concat(
            [temp.dh_rma_data_df, temp.output], ignore_index=True
        )
        # Return new selection
        return temp

    def group_dh_failures(self) -> "FITSDataHealth":
        """
        Group together the data health failures for same key by concating all the values of dh_failure for that key.
        """
        temp = copy(self)
        
        temp.output = temp.output.groupby(by = [col for col in temp.output.columns if col != "dh_failure"]).agg(
            {"dh_failure": lambda x: " | ".join(x)}
        )
        return temp

    def remove_previous_dh_process_cases(self) -> "FITSDataHealth":
        """
        Remove all previously captured cases in dh_process
        """
        temp = copy(self)
        temp.dh_process_df = temp.dh_process_df.merge(
            temp._captured_dh_process_df,
            on= temp._captured_dh_process_df.columns.tolist(),
            how="left",
            indicator=True,
        )
        temp.dh_process_df = temp.dh_process_df.loc[temp.dh_process_df["_merge"] == "left_only"]
        temp.dh_process_df = temp.dh_process_df.drop(columns=["_merge"])
        temp.output = temp.dh_process_df
        return temp

    def remove_previous_dh_rma_data_cases(self) -> "FITSDataHealth":
        """
        Remove all previously captured cases in dh_rma_data
        """
        temp = copy(self)
        temp.new_dh_rma_data_df = temp.dh_rma_data_df.merge(
            temp._captured_dh_rma_data_df,
            on=temp._captured_dh_rma_data_df.columns.tolist(),
            how="left",
            indicator=True,
        )
        temp.new_dh_rma_data_df = temp.new_dh_rma_data_df.loc[temp.new_dh_rma_data_df["_merge"] == "left_only"]
        temp.new_dh_rma_data_df = temp.new_dh_rma_data_df.drop(columns=["_merge"])
        temp.output = temp.new_dh_rma_data_df
        return temp

class HistoricalFITSDH:
    def __init__(self, df):
        ...

    # Do we have current FITS operation for all existing RMA?
    def check_current_fits_operation(self, df):
        ...
