import tensorflow as tf
import sys
import numpy as np


def inspect_tfrecord(file_path):
    """
    Inspect a TFRecord file to understand its structure
    """
    print(f"Inspecting TFRecord file: {file_path}")

    try:
        # Create a TFRecordDataset for the file
        raw_dataset = tf.data.TFRecordDataset(file_path)

        # Try to print raw examples
        print("\nRaw record content:")
        for i, raw_record in enumerate(raw_dataset.take(1)):
            print(f"Record {i + 1} (raw bytes, first 100 bytes):")
            print(raw_record.numpy()[:100])

            # Try to parse as Example
            try:
                example = tf.train.Example()
                example.ParseFromString(raw_record.numpy())
                print("\nParsed as tf.train.Example:")
                print(example)

                print("\nFeature keys:")
                for key in example.features.feature:
                    feature = example.features.feature[key]
                    # Check which field is set
                    if feature.HasField('bytes_list'):
                        print(f"Key: {key}, Type: bytes_list, Length: {len(feature.bytes_list.value)}")
                        if len(feature.bytes_list.value) > 0:
                            # Try to decode bytes as various formats
                            bytes_val = feature.bytes_list.value[0]
                            print(f"  First few bytes: {bytes_val[:20]}")
                            try:
                                # Try to interpret as float32 array
                                float_array = np.frombuffer(bytes_val, dtype=np.float32)
                                print(f"  As float32 array: shape={float_array.shape}, values={float_array[:5]}")
                            except:
                                pass
                    elif feature.HasField('float_list'):
                        print(f"Key: {key}, Type: float_list, Length: {len(feature.float_list.value)}")
                        if len(feature.float_list.value) > 0:
                            print(f"  First few values: {feature.float_list.value[:5]}")
                    elif feature.HasField('int64_list'):
                        print(f"Key: {key}, Type: int64_list, Length: {len(feature.int64_list.value)}")
                        if len(feature.int64_list.value) > 0:
                            print(f"  First few values: {feature.int64_list.value[:5]}")
            except Exception as e:
                print(f"\nFailed to parse as tf.train.Example: {e}")

                # Try other TensorFlow record formats
                print("\nTrying other TensorFlow record formats...")

                # Try parsing as SequenceExample
                try:
                    seq_example = tf.train.SequenceExample()
                    seq_example.ParseFromString(raw_record.numpy())
                    print("\nParsed as tf.train.SequenceExample:")
                    print(seq_example)
                except Exception as e:
                    print(f"Failed to parse as tf.train.SequenceExample: {e}")

                # Try parsing as raw tensor (TFRecord containing direct tensor data)
                try:
                    # Try different data types
                    for dtype in [np.float32, np.float64, np.int32, np.int64]:
                        tensor = np.frombuffer(raw_record.numpy(), dtype=dtype)
                        print(f"\nAs raw {dtype} tensor: Shape={tensor.shape}, First few values={tensor[:5]}")
                except Exception as e:
                    print(f"Failed to parse as raw tensor: {e}")

    except Exception as e:
        print(f"Error inspecting TFRecord: {e}")


if __name__ == "__main__":
    inspect_tfrecord("generalized-image-embeddings-for-the-mimic-chest-x-ray-dataset-1.0/files/p11/p11000183/s51967845/3b8571b4-1418c4eb-ddf2b4bc-5cb96d9b-3b99df84.tfrecord")