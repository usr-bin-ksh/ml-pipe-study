from typing import Union

import tensorflow as tf
import tensorflow_transform as tft

# 피처 이름, 피처 차원
ONE_HOT_FEATURES = {
    "product": 11,
    "sub_product": 45,
    "company_response": 5,
    "state": 60,
    "issue": 90,
}

# 피처 이름, 버킷 개수
BUCKET_FEATURES = {"zip_code": 10}

# 피처 이름, 값은 정의되지 않음
TEXT_FEATURES = {"consumer_complaint_narrative": None}

LABEL_KEY = "consumer_disputed"


def transformed_name(key):
    return key + "_xf"


def fill_in_missing(x: Union[tf.Tensor, tf.SparseTensor]) -> tf.Tensor:
    if isinstance(x, tf.sparse.SparseTensor):
        default_value = "" if x.dtype == tf.string else 0
        x = tf.sparse.to_dense(
            tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
            default_value,
        )
    return tf.squeeze(x, axis=1)


def convert_num_to_one_hot(label_tensor, num_labels=2):
    one_hot_tensor = tf.one_hot(label_tensor, num_labels)
    return tf.reshape(one_hot_tensor, [-1, num_labels])


def convert_zip_code(zipcode):
    if zipcode == "":
        zipcode = "00000"
    zipcode = tf.strings.regex_replace(zipcode, r"X{0,5}", "0")
    zipcode = tf.strings.to_number(zipcode, out_type=tf.float32)
    return zipcode


def preprocessing_fn(inputs):
    outputs = {}

    for key in ONE_HOT_FEATURES.keys():
        dim = ONE_HOT_FEATURES[key]
        int_value = tft.compute_and_apply_vocabulary(
            fill_in_missing(inputs[key]), top_k=dim + 1
        )
        outputs[transformed_name(key)] = convert_num_to_one_hot(
            int_value, num_labels=dim + 1
        )

    for key, bucket_count in BUCKET_FEATURES.items():
        temp_feature = tft.bucketize(
            convert_zip_code(fill_in_missing(inputs[key])),
            bucket_count,
            always_return_num_quantiles=False,
        )
        outputs[transformed_name(key)] = convert_num_to_one_hot(
            temp_feature, num_labels=bucket_count + 1
        )

    for key in TEXT_FEATURES.keys():
        outputs[transformed_name(key)] = fill_in_missing(inputs[key])

    outputs[transformed_name(LABEL_KEY)] = fill_in_missing(inputs[LABEL_KEY])

    return outputs
