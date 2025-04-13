import os
import tensorflow as tf
import numpy as np

from store import Configs

configs = Configs()

def read_tfrecord(file_path):
    with tf.device('/GPU:0'):
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return np.zeros(configs.EMBEDDING_DIM, dtype=np.float32)

        try:
            feature_description = {
                'embedding': tf.io.VarLenFeature(tf.float32),
                'image/id': tf.io.FixedLenFeature([], tf.string, default_value=''),
                'image/format': tf.io.FixedLenFeature([], tf.string, default_value='')
            }

            dataset = tf.data.TFRecordDataset(file_path)

            def _parse_example(example_proto):
                parsed_features = tf.io.parse_single_example(example_proto, feature_description)
                embedding = tf.sparse.to_dense(parsed_features['embedding'])
                return embedding

            parsed_dataset = dataset.map(_parse_example)

            for embedding_tensor in parsed_dataset.take(1):
                embedding_array = embedding_tensor.numpy()
                if len(embedding_array) >= configs.EMBEDDING_DIM:
                    return embedding_array[:configs.EMBEDDING_DIM]
                else:
                    print(f"Warning: Embedding dimension mismatch. Expected {configs.EMBEDDING_DIM}, got {len(embedding_array)}")
                    padded = np.zeros(configs.EMBEDDING_DIM, dtype=np.float32)
                    padded[:len(embedding_array)] = embedding_array
                    return padded
            print(f"Warning: No embedding found in TFRecord: {file_path}")
            return np.zeros(configs.EMBEDDING_DIM, dtype=np.float32)

        except Exception as e:
            print(f"Error reading TFRecord: {e}")
            try:
                feature_description = {
                    'embedding': tf.io.VarLenFeature(tf.float32)}
                dataset = tf.data.TFRecordDataset(file_path)
                parsed_dataset = dataset.map(lambda x: tf.io.parse_single_example(x, feature_description))

                for embedding_tensor in parsed_dataset.take(1):
                    embedding_array = tf.sparse.to_dense(embedding_tensor['embedding']).numpy()
                    if len(embedding_array) >= configs.EMBEDDING_DIM:
                        return embedding_array[:configs.EMBEDDING_DIM]
                    else:
                        padded = np.zeros(configs.EMBEDDING_DIM, dtype=np.float32)
                        padded[:len(embedding_array)] = embedding_array
                        return padded
            except Exception as fallback_error:
                print(f"Fallback approach also failed: {fallback_error}")
                return np.zeros(configs.EMBEDDING_DIM, dtype=np.float32)