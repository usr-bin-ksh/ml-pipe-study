import tensorflow as tf

with tf.io.TFRecordWriter("test.tfrecord") as w:
    w.write(b"First record")
    w.write(b"Second record")

for record in tf.data.TFRecordDataset("test.tfrecord"):
    print(record)

# tf.Tensor(b'First record', shape=(), dtype=string)
# tf.Tensor(b'Second record', shape=(), dtype=string)