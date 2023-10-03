import os
from io import BytesIO, StringIO
import pandas as pd
import boto3
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
