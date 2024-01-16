from copy import copy
import os
import pandas as pd
from utils import add_rma_status_date, get_current_repair_status
from utils import S3Utils

s3_client = S3Utils()

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


class ProcessDHLogic:
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
    def check_unknown_operation(self) -> pd.DataFrame:
        temp = copy(self)
        output = temp.process_df.loc[temp.process_df["OPN_Map"].isnull()]
        output["dh_failure"] = "Unknown operation"
        return output

    # Does the RMA in WO matches Portal RMA?
    def check_rma_portal_match(self) -> pd.DataFrame:
        temp = copy(self)
        all_rma_in_portal = temp._rma_data_df[
            ["serial_number", "rma_number"]
        ].drop_duplicates()
        output = temp.process_df.merge(
            all_rma_in_portal,
            on=["serial_number", "rma_number"],
            how="left",
            indicator=True,
        )
        output = output.loc[output["_merge"] == "left_only"]
        output = output.drop(columns=["_merge"])
        output["dh_failure"] = "RMA not found in Portal"
        return output

    # Is the RMA number format correct in WO? RMA3000010000RMA will raise an alert here.
    def check_rma_format(self) -> pd.DataFrame:
        temp = copy(self)
        process_df = temp.process_df
        process_df["wo_extract"] = process_df["workorder"].str.extract("RMA(.*)")[0]
        process_df["wo_extract"] = process_df["wo_extract"].str.split("-")
        process_df["wo_serial_number"] = process_df["wo_extract"].str[1]
        process_df["wo_rma_number"] = process_df["wo_extract"].str[0]

        is_rma_numeric = process_df["wo_rma_number"].str.isnumeric()
        is_wo_sn_match_sn = (
            process_df["wo_serial_number"] == process_df["serial_number"]
        )

        output = process_df.loc[(~is_rma_numeric) | (~is_wo_sn_match_sn)]
        output["dh_failure"] = "Incorrect workorder format"
        return output

    # Oracle Shipped units that are still open in Debug.(Is Adv. Replc. an exception here?)
    def check_oracle_shipped_units(self) -> pd.DataFrame:
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
        output = temp.process_df.merge(
            rma_data_shipped, on=["serial_number", "rma_number"], how="inner"
        )
        output["dh_failure"] = "Oracle shipped units still in WIP"
        return output

    # Current status of a Repair Close unit should be Salvage, Shipped, or Scraped(spelled incorrectly in FITS as well), Module replacement (status check in)
    def check_repair_close_fits_status(self) -> pd.DataFrame:
        temp = copy(self)
        rma_data_repair_closed = get_current_repair_status(
            temp._rma_data_df,
            temp._repair_status_df,
            status_column_name="current_repair_status",
        )
        rma_data_repair_closed = rma_data_repair_closed.loc[
            rma_data_repair_closed["current_repair_status"] == "Repair Close"
        ]
        rma_data_repair_closed = rma_data_repair_closed.loc[
            :, ["serial_number", "rma_number"]
        ].drop_duplicates()
        output = temp._current_fits_operation_df.merge(
            rma_data_repair_closed,
            on=["serial_number", "rma_number"],
            how="inner",
        )
        output = output.loc[
            ~output["status"].str.upper().isin(temp._CLOSED_FITS_STATUS_CODE)
        ]
        output["dh_failure"] = "Repair Close status incorrect"
        return output

    # Is the trans_seq of Debug/Verification step present in Parameter?
    def check_trans_seq_in_param(self) -> pd.DataFrame:
        temp = copy(self)
        process_df = temp.process_df.loc[
            temp.process_df["operation"].isin(temp._DIAGNOSIS_OPERATION_CODE)
        ]
        parameter_df = temp._parameter_df[
            ["serial_number", "trans_seq"]
        ].drop_duplicates()
        output = process_df.merge(
            parameter_df,
            on=["serial_number", "trans_seq"],
            how="left",
            indicator=True,
        )
        output = output.loc[output["_merge"] == "left_only"]
        output = output.drop(columns=["_merge"])
        output["dh_failure"] = "trans_seq not found in Parameter"
        return output

    # Is the Parameter data entry closed before checkout datetime of that trans_seq?
    def check_param_entry_closed_before_process_checkout(self) -> pd.DataFrame:
        temp = copy(self)
        process_df = temp.process_df.loc[
            temp.process_df["operation"].isin(temp._DIAGNOSIS_OPERATION_CODE)
        ]
        parameter_df = temp._parameter_df.groupby(
            by=["serial_number", "trans_seq"], as_index=False
        ).agg({"datetime_script": max})
        parameter_df = parameter_df.rename(
            columns={"datetime_script": "max_parameter_entry_datetime"}
        )
        output = process_df.merge(
            parameter_df, on=["serial_number", "trans_seq"], how="inner"
        )
        output = output.loc[
            output["max_parameter_entry_datetime"]
            > output["datetime_checkout"]
        ]
        output["dh_failure"] = "Parameter entry made after process checkout"
        return output

    # No New data added in process today.
    # def check_last_update_in_process(self) -> "FITSDataHealth":
    #     temp = copy(self)


class ProcessDHOutput:
    # output_dir = f"output/{tod.strftime('%Y%m%d')}"
    # captured_cases_dir = "output/captured_cases"

    def __init__(self) -> None:
        self._load_dh_process()

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
            "file_date",
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

    _dh_process_parse_dates: list = ["datetime_checkin", "datetime_checkout", "file_date",]

    dh_process_df: pd.DataFrame = pd.DataFrame(columns=_dh_process_columns)

    def _load_dh_process(self) -> None:
        _dh_process_file_location = "output/captured_cases/DH_Process.csv"

        try:
            self._captured_dh_process_df = pd.read_csv(
                s3_client.get_s3_obj_body(_dh_process_file_location),
                usecols= self._dh_process_columns,
                dtype= self._dh_process_types,
                parse_dats= self._dh_process_parse_dates,
            )
            self._captured_dh_process_df["serial_number"] = (
                self._captured_dh_process_df["serial_number"].str.split(".").str[0]
            )
        except FileNotFoundError:
            self._captured_dh_process_df = pd.DataFrame(columns= self._dh_process_columns)

    def file_date(self):
        return pd.Timestamp.today().date()
    
    def filter_new_cases(self, df, existing_cases) -> pd.DataFrame:
        df = df.loc[:, self._dh_process_columns]
        existing_cases = existing_cases.loc[:, self._dh_process_columns]
        # drop file_date column
        existing_cases = existing_cases.drop(columns=['file_date'])

        if df.empty:
            print("Data output is empty!!!")
            return df

        if existing_cases.empty:
            print("No existing cases!!!")
            return df
        
        if set(df['dh_failure']) != set(existing_cases['dh_failure']):
            print("Data Health failure message is not same!!!")
            # return empty dataframe with same columns as df
            return pd.DataFrame(columns= df.columns)
        
        df = df.merge(existing_cases, how='left', indicator=True)
        df = df.loc[df['_merge'] == 'left_only']
        df = df.drop(columns=['_merge'])
        return df
    
    # Write a function to update the existing_cases file in S3.
    def update_existing_cases_file(self, new_cases, existing_cases, existing_cases_location) -> None:
        # Check if new cases is empty
        if new_cases.empty:
            print("No new cases!!!")
            return
        
        # Check if new cases have a file_date column that shows today's date, if not then add
        if 'file_date' not in new_cases.columns:
            new_cases['file_date'] = self.file_date()
        else:
            if ((len(new_cases['file_date'].unique()) > 1) | new_cases['file_date'].iloc[0] != self.file_date()):
                print("File date is not today's date!!!")
                return
            

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
            raise KeyError(
                "Data health output does not have all the columns required in dh_process. You could be trying to append some other dh to process dh"
            )

        temp.dh_process_df = pd.concat(
            [temp.dh_process_df, temp.output], ignore_index=True
        )
        return temp

    def setup_output_directory(self) -> None:
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
