from typing import Text, List

import tensorflow as tf
import tensorflow_hub as hub
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


def transformed_name(key: Text) -> Text:
    """Generate the name of the transformed feature from original name."""
    return key + "_xf"


def vocabulary_name(key: Text) -> Text:
    """Generate the name of the vocabulary feature from original name."""
    return key + "_vocab"


def transformed_names(keys: List[Text]) -> List[Text]:
    """Transform multiple feature names at once."""
    return [transformed_name(key) for key in keys]


def _gzip_reader_fn(filenames):
    """Small utility returning a record reader that can read gzip'ed files."""
    return tf.data.TFRecordDataset(filenames, compression_type="GZIP")


def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Returns a function that parses a serialized tf.Example and applies TFT.
    """
    # 전처리 그래프를 로드합니다.
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        """Returns the output to be used in the serving signature."""
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)
        # 요청에서 원시tf.Example 레코드를 구문 분석합니다.
        parsed_features = tf.io.parse_example(
            serialized_tf_examples, feature_spec
        )

        # 전처리 변환을 원시 데이터에 적용합니다.
        transformed_features = model.tft_layer(parsed_features)

        return model(transformed_features)

    return serve_tf_examples_fn


def input_fn(file_pattern, tf_transform_output, batch_size=200):
    """Generates features and label for tuning/training.

  Args:
    file_pattern: input tfrecord file pattern.
    tf_transform_output: A TFTransformOutput.
    batch_size: representing the number of consecutive elements of returned
      dataset to combine in a single batch

  Returns:
    A dataset that contains (features, indices) tuple where features is a
      dictionary of Tensors, and indices is a single Tensor of label indices.
  """
    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy()
    )

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=_gzip_reader_fn,
        label_key=transformed_name(LABEL_KEY),
    )

    return dataset


def get_model():

    # 원-핫 범주형 피처
    input_features = []
    # 피처를 루프 돌리며 각 피처에 대한 input_feature를 작성합니다.
    for key, dim in ONE_HOT_FEATURES.items():
        input_features.append(
            tf.keras.Input(shape=(dim + 1,),
                           name=transformed_name(key)))

    # 버킷화 피처를 추가
    for key, dim in BUCKET_FEATURES.items():
        input_features.append(
            tf.keras.Input(shape=(dim + 1,),
                           name=transformed_name(key)))

    # 문자열 피처를 추가
    input_texts = []
    for key in TEXT_FEATURES.keys():
        input_texts.append(
            tf.keras.Input(shape=(1,),
                           name=transformed_name(key),
                           dtype=tf.string))

    inputs = input_features + input_texts

    # 문자열 피처를 임베딩
    MODULE_URL = "https://tfhub.dev/google/universal-sentence-encoder/4"
    # 범용 문장 인코더 모델의 tf.hub 모듈을 로드합니다.
    embed = hub.KerasLayer(MODULE_URL)
    # 케라스 입력은 2차원이지만 인코더는 1차원 입력을 기대합니다.
    reshaped_narrative = tf.reshape(input_texts[0], [-1])
    embed_narrative = embed(reshaped_narrative)
    deep_ff = tf.keras.layers.Reshape((512, ), input_shape=(1, 512))(embed_narrative)

    deep = tf.keras.layers.Dense(256, activation='relu')(deep_ff)
    deep = tf.keras.layers.Dense(64, activation='relu')(deep)
    deep = tf.keras.layers.Dense(16, activation='relu')(deep)

    wide_ff = tf.keras.layers.concatenate(input_features)
    wide = tf.keras.layers.Dense(16, activation='relu')(wide_ff)

    both = tf.keras.layers.concatenate([deep, wide])

    output = tf.keras.layers.Dense(1, activation='sigmoid')(both)
    # 피처 API로 모델 그래프를 조립합니다.
    keras_model = tf.keras.models.Model(inputs, output)

    keras_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                        loss='binary_crossentropy',
                        metrics=[
                            tf.keras.metrics.BinaryAccuracy(),
                            tf.keras.metrics.TruePositives()
                        ])
    return keras_model


def run_fn(fn_args):

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    # input_fn 을 호출하여 데이터 생성기를 가져옵니다.
    train_dataset = input_fn(fn_args.train_files, tf_transform_output)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output)

    # get_model함수를 호출하여 컴파일된 케라스 모델을 가져옵니다.
    model = get_model()
    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        # Trainer 컴포넌트가 통과한 학습 및 평가 단계 수를 사용하여 모델을 학습합니다.
        validation_steps=fn_args.eval_steps)

    # 나중에 설명할 서빙 피처를 포함하는 모델 서명을 정의합니다.
    signatures = {
        'serving_default':
            _get_serve_tf_examples_fn(
                model,
                tf_transform_output).get_concrete_function(
                tf.TensorSpec(
                    shape=[None],
                    dtype=tf.string,
                    name='examples')
            )
    }
    model.save(fn_args.serving_model_dir,
               save_format='tf', signatures=signatures)
