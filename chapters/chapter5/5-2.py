import tensorflow as tf


def process_image(raw_image):
    raw_image = tf.reshape(raw_image, [-1])
    # JPEG 이미지 형식을 디코딩합니다.
    img_rgb = tf.io.decode_jpeg(raw_image, channels=3)
    # 로드된 RGB 영상을 회색조로 변환합니다.
    img_gray = tf.image.rgb_to_grayscale(img_rgb)
    img = tf.image.convert_image_dtype(img_gray, tf.float32)
    # 이미지 크기를 300 × 300 픽셀로 조정합니다.
    resized_img = tf.image.resize_with_pad(
        img,
        target_height=300,
        target_width=300
    )
    # 이미지를 회색조로 변환합니다.
    img_grayscale = tf.image.rgb_to_grayscale(resized_img)
    return tf.reshape(img_grayscale, [-1, 300, 300, 1])
