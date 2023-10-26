import pprint
import tempfile

import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_transform.beam as tft_beam
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import schema_utils


def preprocessing_fn(inputs):
    """Preprocess input columns into transformed columns."""
    print(inputs)
    x = inputs['x']
    x_xf = tft.scale_to_0_1(x)
    return {
        'x_xf': x_xf,
    }


raw_data = [
    {'x': 1.20},
    {'x': 2.99},
    {'x': 100.0}
]

raw_data_metadata = dataset_metadata.DatasetMetadata(
    schema_utils.schema_from_feature_spec({
        'x': tf.io.FixedLenFeature([], tf.float32),
    }))

with tft_beam.Context(temp_dir=tempfile.mkdtemp()):
    transformed_dataset, transform_fn = (  # pylint: disable=unused-variable
            (raw_data, raw_data_metadata) | tft_beam.AnalyzeAndTransformDataset(
        preprocessing_fn))

transformed_data, transformed_metadata = transformed_dataset  # pylint: disable=unused-variable

pprint.pprint(transformed_data)