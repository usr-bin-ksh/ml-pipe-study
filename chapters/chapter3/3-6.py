import os
import tensorflow as tf
import pandas as pd
import numpy as np
import shutil
from pathlib import Path

# 웹 상에서 원시 데이터를 다운 받습니다.
filepath = tf.keras.utils.get_file(
    "complaints.csv.zip",
    "http://files.consumerfinance.gov/ccdb/complaints.csv.zip")

dir_path = Path(__file__).parent.absolute()
data_dir = os.path.join(dir_path, "..", "..", "data")
processed_dir = os.path.join(dir_path, "..", "..", "data", "processed")
Path(processed_dir).mkdir(parents=True, exist_ok=True)

# 압축을 해제합니다.
shutil.unpack_archive(filepath, data_dir)
# pandas로 csv 파일을 읽어 들입니다.
df = pd.read_csv(os.path.join(data_dir, "complaints.csv"))

df.columns = [
    "date_received", "product", "sub_product", "issue", "sub_issue",
    "consumer_complaint_narrative", "company_public_response",
    "company", "state", "zip_code", "tags",
    "consumer_consent_provided", "submitted_via",
    "date_sent_to_company", "company_response",
    "timely_response", "consumer_disputed", "complaint_id"]

df.loc[df["consumer_disputed"] == "", "consumer_disputed"] = np.nan

# 주요한 필드가 비어있는 경우 레코드를 제외합니다.
df = df.dropna(subset=["consumer_complaint_narrative", "consumer_disputed"])

# Label 필드인 consumer_disputed를 Yes, No에서 1, 0 으로 변경합니다.
df.loc[df["consumer_disputed"] == "Yes", "consumer_disputed"] = 1
df.loc[df["consumer_disputed"] == "No", "consumer_disputed"] = 0

df.loc[df["zip_code"] == "", "zip_code"] = "000000"
df.loc[pd.isna(df["zip_code"]), "zip_code"] = "000000"

df = df[df['zip_code'].str.len() == 5]
df["zip_code"] = df['zip_code'].str.replace('XX', '00')
df = df.reset_index(drop=True)
df["zip_code"] = pd.to_numeric(df["zip_code"], errors='coerce')

# 판다스 DataFrame을 csv 파일로 다시 저장합니다.
df.to_csv(os.path.join(processed_dir, "processed-complaints.csv"), index=False)
