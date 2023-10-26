import tensorflow_transform as tft


def preprocessing_fn(inputs):
    x = inputs['x']
    x_normalized = tft.scale_to_0_1(x)
    return {
        'x_xf': x_normalized
    }
