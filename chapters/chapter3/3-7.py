import os
import re

import tensorflow as tf
import pandas as pd
from pathlib import Path


def _bytes_feature(value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[value.encode()])
    )


def _float_feature(value):
    return tf.train.Feature(
        float_list=tf.train.FloatList(value=[value])
    )


def _int64_feature(value):
    return tf.train.Feature(
        int64_list=tf.train.Int64List(value=[value])
    )


def clean_rows(row):
    if pd.isna(row["zip_code"]):
        row["zip_code"] = "99999"
    return row


def convert_zipcode_to_int(zipcode):
    nums = re.findall(r'\d+', zipcode)
    if len(nums) > 0:
        int_zipcode = int(nums[0])
    else:
        int_zipcode = 99999
    return int_zipcode


dir_path = Path(__file__).parent.absolute()
data_dir = os.path.join(dir_path, "..", "..", "data")
tfrecord_dir = os.path.join(dir_path, "..", "..", "data", "tfrecord")
df = pd.read_csv(os.path.join(data_dir, "processed-complaints.csv"))

tfrecord_filename = "consumer-complaints.tfrecord"
tfrecord_filepath = os.path.join(tfrecord_dir, tfrecord_filename)
# tfrecord_filename에 지정된 경로에 저장하는 TFRecordWriter 객체를 만듭니다.
tf_record_writer = tf.io.TFRecordWriter(tfrecord_filepath)

for index, row in df.iterrows():
    row = clean_rows(row)
    # 모든 데이터 레코드를 tf.train.Example로 변환
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "product": _bytes_feature(str(row["product"])),
                "sub_product": _bytes_feature(str(row["sub_product"])),
                "issue": _bytes_feature(str(row["issue"])),
                "sub_issue": _bytes_feature(str(row["sub_issue"])),
                "state": _bytes_feature(str(row["state"])),
                "zip_code": _int64_feature(convert_zipcode_to_int(row["zip_code"])),
                "company": _bytes_feature(str(row["company"])),
                "company_response": _bytes_feature(str(row["company_response"])),
                "timely_response": _bytes_feature(str(row["timely_response"])),
                "consumer_disputed": _float_feature(
                    row["consumer_disputed"]
                ),
            }
        )
    )
    # 데이터 구조를 직렬화
    tf_record_writer.write(example.SerializeToString())
tf_record_writer.close()
