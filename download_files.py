import boto3
import botocore

BUCKET_NAME = 'aca-qdap-production-transfer'

s3 = boto3.resource('s3')

SAVE_PATH = "input/"

# LOC_PROCESS_ALL = "FITS_incoming/Process all.csv"
# LOC_PARAMETER_ALL = "FITS_incoming/Parameter all.csv"
LOC_CURRENT_OPERATION = "FITS_incoming/process_current_operation.csv"
# LOC_PROCESS_CURRENT_OPERATION = "FITS_incoming/process_current_operation.csv"
# LOC_INDEX_RMA_DIM = "oracle_data_incoming/RMA_dim.csv"
# LOC_REPAIR_STATUS = "oracle_data_incoming/repair_status.csv"
# LOC_SERIALSHIP = "oracle_data_incoming/serialship.csv"
# LOC_RMA_OPERATIONS = "oracle_data_incoming/rma_operations.csv"

LIST_OF_KEY = [eval(x) for x in dir() if x.startswith("LOC")]

for KEY in LIST_OF_KEY:
    filename = KEY.split("/")[-1]
    print(KEY)
    try:
        s3.Bucket(BUCKET_NAME).download_file(KEY, SAVE_PATH+filename)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise